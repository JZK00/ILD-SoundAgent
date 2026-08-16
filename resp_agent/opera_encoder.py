# -*- coding: utf-8 -*-
"""OPERA-CT特征提取薄封装, 原样调用
`D:\\OPERA\\src\\benchmark\\model_util.py::extract_opera_feature`,
不改变其内部Log-Mel参数(n_mels=64, fmin=50, fmax=8000, nfft=1024, hop=512)、
16kHz单声道重采样、静音裁剪、重复填充到input_sec=8秒的任何逻辑。

该函数内部用相对路径`cks/model/encoder-operaCT.ckpt`定位checkpoint, 且这个
相对路径解析发生在*调用时*而不是import时, 所以每次实际调用都必须让当前
工作目录是`D:\\OPERA`, 而不是只在import的时候chdir一次——已经用真实音频文件
从非D:\\OPERA工作目录验证过这个前提。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .audio_io import as_standard_wav

OPERA_ROOT = Path(
    os.environ.get("OPERA_ROOT", r"D:\OPERA")
).expanduser().resolve()
if str(OPERA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERA_ROOT))

PRETRAIN_NAME = "operaCT"
FEATURE_DIM = 768
INPUT_SEC = 8  # 与extract_all_features.py/训练数据处理完全一致, 不得改动
CKPT_PATH = OPERA_ROOT / "cks" / "model" / "encoder-operaCT.ckpt"

CACHE_DIR = (
    Path(__file__).resolve().parent.parent
    / ".opera_feature_cache"
)
# 换提取逻辑/换checkpoint文件本身以外的"提取代码版本"时手动改这个字符串,
# 让旧缓存自动失效(即使音频文件和checkpoint文件指纹都没变)。
CACHE_VERSION = "resp_agent_app_v1"

_extract_opera_feature = None  # 懒加载, 避免import这个模块就强制触发checkpoint相关逻辑

# extract_opera_feature内部靠"当前工作目录=D:\OPERA"解析相对路径checkpoint,
# 这是整个进程共享的可变状态。Gradio(或任何多线程调用方)如果同时处理两个
# 请求, 两个线程各自的_opera_cwd()调用会互相踩到对方改过的cwd。用一把可重入
# 全局锁把"改cwd -> 调用OPERA代码 -> 还原cwd"这段完整包起来, 串行化所有实际
# 触碰cwd的调用(懒加载import和真正的特征提取调用都必须在锁内, 见下方两处
# 用法), 而不只是import那一次。
_OPERA_CALL_LOCK = threading.RLock()


@contextlib.contextmanager
def _opera_cwd():
    with _OPERA_CALL_LOCK:
        prev = os.getcwd()
        os.chdir(OPERA_ROOT)
        try:
            yield
        finally:
            os.chdir(prev)


def _load_extractor():
    global _extract_opera_feature
    if _extract_opera_feature is None:
        with _opera_cwd():
            from src.benchmark.model_util import extract_opera_feature
        _extract_opera_feature = extract_opera_feature
    return _extract_opera_feature


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_fingerprint() -> str:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"OPERA-CT checkpoint未找到: {CKPT_PATH}; 请确认D:\\OPERA\\cks\\model\\目录完整。"
        )
    stat = CKPT_PATH.stat()
    # 用大小+mtime做轻量指纹, 避免每次都对~340MB文件重新算完整sha256;
    # 如果以后要百分之百防止"文件内容变了但mtime没变"的边角情况, 可以改成
    # 完整哈希, 目前这个粒度足够区分"checkpoint被替换过"。
    return f"{stat.st_size}-{int(stat.st_mtime)}"


def _cache_key(audio_path: Path) -> str:
    payload = {
        "audio_sha256": _file_hash(audio_path),
        "opera_ckpt_fingerprint": _checkpoint_fingerprint(),
        "pretrain": PRETRAIN_NAME,
        "input_sec": INPUT_SEC,
        "dim": FEATURE_DIM,
        "cache_version": CACHE_VERSION,
    }
    key_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _read_cache_if_valid(cache_file: Path) -> Optional[np.ndarray]:
    """读缓存并校验; 文件损坏/形状不对/含NaN-Inf就删掉重来, 不静默返回坏数据。"""
    if not cache_file.exists():
        return None
    try:
        arr = np.load(cache_file)
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None
    if arr.shape != (FEATURE_DIM,) or arr.dtype.kind != "f" or not np.isfinite(arr).all():
        cache_file.unlink(missing_ok=True)
        return None
    return arr.astype(np.float32)


def _atomic_cache_write(cache_file: Path, feature: np.ndarray) -> None:
    """先写同目录临时文件, 再os.replace原子替换, 避免并发请求同时写同一个
    缓存文件时读到半写的坏数据。传文件对象给np.save(而不是路径字符串), 绕开
    numpy会自动给不以.npy结尾的文件名追加.npy后缀这个坑。"""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(cache_file.parent), prefix=cache_file.stem + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, feature)
        os.replace(tmp_path, cache_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def extract_opera_features(audio_path: Union[str, Path], model_name: str = "opera_ct") -> np.ndarray:
    """audio_path -> 768维OPERA-CT特征(np.float32, shape=(768,))。

    `model_name`目前只支持"opera_ct"(锁定的患者级checkpoint只用OPERA-CT
    特征, 加OPERA-GT是超出当前MVP范围的新功能, 不在这里假装实现)。
    """
    if model_name != "opera_ct":
        raise NotImplementedError(
            f"model_name={model_name!r} 尚未实现, 目前只支持 'opera_ct' "
            "(锁定的patient级checkpoint只用OPERA-CT特征)"
        )

    with as_standard_wav(audio_path) as wav_path:
        key = _cache_key(wav_path)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{key}.npy"

        cached = _read_cache_if_valid(cache_file)
        if cached is not None:
            return cached

        extractor = _load_extractor()
        with _opera_cwd():
            features = extractor(
                [str(wav_path)], pretrain=PRETRAIN_NAME, input_sec=INPUT_SEC, dim=FEATURE_DIM
            )
        feature = np.asarray(features[0], dtype=np.float32)
        if feature.shape != (FEATURE_DIM,):
            raise RuntimeError(
                f"extract_opera_feature返回了意外的形状 {feature.shape}, 期望 ({FEATURE_DIM},)"
            )
        _atomic_cache_write(cache_file, feature)
        return feature


def extract_opera_features_batch(
    audio_paths: list, model_name: str = "opera_ct"
) -> np.ndarray:
    """批量版`extract_opera_features`。逐个先查缓存, 只把未命中的音频一次性
    传给底层`extract_opera_feature`(只加载一次~340MB checkpoint, 一次前向
    完成全部未命中项), 输出顺序和`audio_paths`严格一致。时间段定位(一段
    10秒音频~17个窗口x3次确定性噪声重复~51次调用)靠这个函数避免51次重复
    加载模型。"""
    if model_name != "opera_ct":
        raise NotImplementedError(
            f"model_name={model_name!r} 尚未实现, 目前只支持 'opera_ct'"
        )
    if not audio_paths:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    with contextlib.ExitStack() as stack:
        wav_paths = [stack.enter_context(as_standard_wav(p)) for p in audio_paths]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_files = [CACHE_DIR / f"{_cache_key(wp)}.npy" for wp in wav_paths]

        results: list = [None] * len(audio_paths)
        miss_indices = []
        for i, cf in enumerate(cache_files):
            cached = _read_cache_if_valid(cf)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)

        if miss_indices:
            miss_wav_paths = [str(wav_paths[i]) for i in miss_indices]
            extractor = _load_extractor()
            with _opera_cwd():
                batch_features = extractor(
                    miss_wav_paths, pretrain=PRETRAIN_NAME, input_sec=INPUT_SEC, dim=FEATURE_DIM
                )
            for local_idx, global_idx in enumerate(miss_indices):
                feature = np.asarray(batch_features[local_idx], dtype=np.float32)
                if feature.shape != (FEATURE_DIM,):
                    raise RuntimeError(
                        f"extract_opera_feature返回了意外的形状 {feature.shape} "
                        f"(第{global_idx}个音频: {audio_paths[global_idx]}), 期望 ({FEATURE_DIM},)"
                    )
                _atomic_cache_write(cache_files[global_idx], feature)
                results[global_idx] = feature

    return np.stack(results, axis=0)
