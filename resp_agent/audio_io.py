# -*- coding: utf-8 -*-
"""音频输入统一处理: wav直接用; m4a/mp3先转码成OPERA-CT期望的16kHz单声道wav
临时文件(用后自动清理)。不修改`extract_opera_feature`内部任何逻辑, 只保证
喂给它的路径永远是它认识的.wav。"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator, Union

import librosa
import soundfile as sf

OPERA_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.1
SUPPORTED_SUFFIXES = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}


class AudioLoadError(RuntimeError):
    """空音频/损坏文件/过短音频/不支持的格式统一抛这个, 消息里写清楚原因,
    方便Gradio界面直接把message展示给用户, 不需要再解析异常类型。"""


def _validate_path(path: Path) -> None:
    if not path.exists():
        raise AudioLoadError(f"音频文件不存在: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioLoadError(
            f"不支持的音频格式 {path.suffix!r}(支持: {sorted(SUPPORTED_SUFFIXES)}): {path}"
        )
    if path.stat().st_size == 0:
        raise AudioLoadError(f"音频文件为空(0字节): {path}")


def _check_wav_readable_and_duration(path: Path) -> None:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise AudioLoadError(f"无法读取wav文件头, 文件可能已损坏: {path} ({exc})") from exc
    if info.frames == 0:
        raise AudioLoadError(f"wav文件不含任何音频帧(空音频): {path}")
    duration = info.frames / info.samplerate
    if duration < MIN_DURATION_SECONDS:
        raise AudioLoadError(f"音频过短({duration:.3f}秒 < {MIN_DURATION_SECONDS}秒下限): {path}")


@contextlib.contextmanager
def as_standard_wav(input_path: Union[str, Path]) -> Iterator[Path]:
    """返回一个16kHz单声道wav文件的绝对路径, 供`opera_encoder`直接使用。

    如果输入本来就是.wav, 原样使用(不做无谓的重新编码, 交给OPERA自己的
    librosa.load去做重采样, 和训练数据处理路径完全一致), 只做基本可读性/
    时长校验。只有m4a/mp3等其他格式才会先转码成临时wav文件, 用完在
    with块结束后自动删除。
    """
    path = Path(input_path).resolve()
    _validate_path(path)

    if path.suffix.lower() == ".wav":
        _check_wav_readable_and_duration(path)
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="resp_agent_audio_") as tmp_dir:
        tmp_wav = Path(tmp_dir) / (path.stem + ".wav")
        try:
            wav_data, sr = librosa.load(str(path), sr=OPERA_SAMPLE_RATE, mono=True)
        except Exception as exc:  # 常见于损坏文件, 或缺少ffmpeg导致m4a/mp3解码失败
            raise AudioLoadError(
                f"无法解码音频文件 {path} ({exc}); 如果是m4a/mp3格式, "
                "请确认系统已安装ffmpeg并加入PATH。"
            ) from exc
        if wav_data.size == 0:
            raise AudioLoadError(f"音频文件解码后为空: {path}")
        duration = len(wav_data) / sr
        if duration < MIN_DURATION_SECONDS:
            raise AudioLoadError(f"音频过短({duration:.3f}秒 < {MIN_DURATION_SECONDS}秒下限): {path}")
        sf.write(str(tmp_wav), wav_data, sr, subtype="PCM_16")
        yield tmp_wav
