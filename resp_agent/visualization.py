# -*- coding: utf-8 -*-
"""Static visualizations for patient-, site-, and interval-level model evidence.

只做渲染, 不做任何新的分类判据/阈值/峰值检测计算——所有画在图上的数值都
直接来自`localization.py::localize_suspected_abnormal_segments()`的返回值
(或按同样方法用合成window_stats跑`_classify_evidence_pattern`得到的等价
结构, 供单元测试用), 不在本模块里重新推导。

时间轴对齐说明: 频谱图和有符号时间证据曲线共用同一个`waveform`/`sr`——
调用方必须传入`localize_suspected_abnormal_segments`内部实际用于滑窗定位
的那份**未经OPERA特征提取管线裁剪/静音裁剪/重复填充到8秒**的原始波形
(即`localization.py`里`sf.read(...)`读出来的那份, 不是喂给
`opera_encoder.extract_opera_features`前的那份8秒定长输入)。本模块提供
`load_localization_waveform()`按和`localization.py`完全一致的方式重新读取
这份波形, 避免调用方各自实现一遍读取逻辑导致两边不一致。

四类顶层证据模式(focal/diffuse/minimal/unstable)本身不因为画图而改变；
只有`evidence_pattern == "focal"`时`suspected_abnormal_segments`才非空,
diffuse/minimal/unstable不会有任何声段框——这是`localization.py`分类逻辑
决定的, 本模块只是如实按segments列表画, 不强加。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能存图, 不依赖本机是否有GUI后端

import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import soundfile as sf
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Optional, Tuple, Union

from .audio_io import as_standard_wav
from .schemas import (
    DIRECTION_MINIMAL,
    DIRECTION_OPPOSES,
    DIRECTION_SUPPORTS,
    FORBIDDEN_TERMS,
    SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
)

# 这组Log-Mel参数是专门为*本模块的展示频谱图*选的, 不是opera_encoder.py
# 内部特征提取管线参数的复制(那条管线输出768维池化向量, 本来就不暴露中间
# 频谱图给可视化层选择frame粒度的自由度; 也不套用它的静音裁剪/重复填充到
# 8秒逻辑——定位算法本身也是在未裁剪的原始波形上做滑窗, 见localization.py:
# `waveform, sr = sf.read(...)`, 画图应该和它看到的是同一份数据, 但frame
# hop完全是画图层自己的选择)。
# fmin原来用50Hz, n_fft=1024/hop=512时会出现"Empty filters detected in mel
# frequency basis"(部分低频mel band在这个FFT频率分辨率下覆盖不到任何频点,
# 频谱图对应频段会缺信息)——改成fmin=20, hop=256后不再出现, 已经用
# `librosa.filters.mel(...).sum(axis=1)`验证64个mel band没有空滤波器
# (见`_mel_spectrogram_db`里的显式assert, 不是靠肉眼看警告消失就当修好了)。
MEL_N_MELS = 64
MEL_FMIN = 20
MEL_FMAX = 8000
MEL_N_FFT = 1024
MEL_HOP = 256

THEMES = {
    "paper": {
        "figure_facecolor": "#ffffff",
        "axes_facecolor": "#ffffff",
        "text_color": "#1a1a1a",
        "muted_text_color": "#5a5a5a",
        "grid_color": "#d9d9d9",
        "spine_color": "#333333",
        "supports_color": "#c0392b",
        "opposes_color": "#1f6fb2",
        "minimal_color": "#7f7f7f",
        "na_color": "#bfbfbf",
        "borderline_color": "#c8790a",
        "std_band_alpha": 0.22,
        "segment_fill_alpha": 0.16,
        "cmap": "magma",
    },
    "dark": {
        "figure_facecolor": "#1b1b1b",
        "axes_facecolor": "#1b1b1b",
        "text_color": "#e8e8e8",
        "muted_text_color": "#a8a8a8",
        "grid_color": "#3a3a3a",
        "spine_color": "#8f8f8f",
        "supports_color": "#e57373",
        "opposes_color": "#64b5f6",
        "minimal_color": "#a8a8a8",
        "na_color": "#5c5c5c",
        "borderline_color": "#f0a83c",
        "std_band_alpha": 0.28,
        "segment_fill_alpha": 0.22,
        "cmap": "magma",
    },
}

_DIRECTION_COLOR_KEY = {
    DIRECTION_SUPPORTS: "supports_color",
    DIRECTION_OPPOSES: "opposes_color",
    DIRECTION_MINIMAL: "minimal_color",
}


class VisualizationInputError(ValueError):
    pass


class VisualizationFontError(RuntimeError):
    """找不到任何支持中文的字体时抛这个, 不静默retreat成方框。本模块的
    免责声明(`SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER`)和diffuse/minimal/
    unstable的说明文案(`_PATTERN_NOTES`)都是中文——`FORBIDDEN_TERMS`检查
    只验证字符串*内容*不含禁用措辞, 不代表渲染环境真的有字体能把这些字符
    画出来而不是方框, 这是两件事, 需要分开检查(见`_resolve_cjk_font`和
    对应的`test_cjk_font_available`)。"""


# 依次尝试的中文字体候选, Windows/Linux都覆盖, 不硬编码单一平台的字体路径。
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK",
    "Noto Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)

_cjk_font_name: Optional[str] = None  # 模块级缓存, 避免每次画图都重新扫描字体列表


def _resolve_cjk_font(candidates=CJK_FONT_CANDIDATES) -> str:
    global _cjk_font_name
    if _cjk_font_name is not None:
        return _cjk_font_name
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            _cjk_font_name = name
            return name
    raise VisualizationFontError(
        f"未找到任何支持中文的字体(尝试过: {list(candidates)}); 免责声明等中文"
        "文案会渲染成方框而不是可读文字, 不能静默出图。请安装其中一种字体后"
        "重试(Windows通常已预装Microsoft YaHei/SimHei; Linux: "
        "sudo apt install fonts-noto-cjk), 或把本机已安装的中文字体名称加入"
        "resp_agent/visualization.py::CJK_FONT_CANDIDATES。"
    )


def _configure_cjk_font() -> str:
    """把找到的中文字体加进matplotlib的sans-serif字体列表最前面(不是整体
    替换——图上英文标题/坐标轴仍然优先用matplotlib默认字体渲染, 只是遇到
    中文字符时会回退用这个字体, 不会出方框), 并关掉unicode_minus(部分中文
    字体没有unicode负号字形, 不关掉会导致坐标轴负数刻度显示成方框)。"""
    font_name = _resolve_cjk_font()
    current = list(matplotlib.rcParams.get("font.sans-serif", []))
    if font_name in current:
        current.remove(font_name)
    matplotlib.rcParams["font.sans-serif"] = [font_name] + current
    matplotlib.rcParams["axes.unicode_minus"] = False
    return font_name


def _get_theme(theme: str) -> dict:
    if theme not in THEMES:
        raise VisualizationInputError(f"未知theme={theme!r}, 可选: {list(THEMES)}")
    return THEMES[theme]


def _direction_color(direction: Optional[str], theme: dict) -> str:
    if direction is None:
        return theme["na_color"]
    key = _DIRECTION_COLOR_KEY.get(direction)
    if key is None:
        return theme["na_color"]
    return theme[key]


def _style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["axes_facecolor"])
    for spine in ax.spines.values():
        spine.set_color(theme["spine_color"])
    ax.tick_params(colors=theme["text_color"], labelsize=9)
    ax.xaxis.label.set_color(theme["text_color"])
    ax.yaxis.label.set_color(theme["text_color"])
    ax.title.set_color(theme["text_color"])
    ax.grid(True, color=theme["grid_color"], linewidth=0.6, alpha=0.7)


def _add_disclaimer_footer(fig, theme: dict) -> None:
    """固定限定说明, 每张图都带, 不因为theme/病例不同而省略或改写措辞
    (措辞本身直接复用`schemas.SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER`, 不在
    本模块里重新写一遍类似但不完全一致的句子)。"""
    fig.text(
        0.01, 0.005, SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
        ha="left", va="bottom", fontsize=7.5, color=theme["muted_text_color"], wrap=True,
    )


def load_localization_waveform(audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """按和`localization.py::localize_suspected_abnormal_segments`内部完全
    一致的方式读取波形(先经`audio_io.as_standard_wav`, 再`soundfile.read`,
    多声道取平均), 不经过OPERA特征提取管线的静音裁剪/重复填充到8秒/内部
    重采样到16kHz逻辑, 保证和`evidence_curve`/`suspected_abnormal_segments`
    的时间轴严格对应同一份数据。

    采样率不强制假设/改写成16kHz: `as_standard_wav`对已经是.wav格式的输入
    完全不重采样(只做可读性/时长校验), 原样保留文件本身的原生采样率;
    只有m4a/mp3才会在格式转换过程中被重采样到`audio_io.OPERA_SAMPLE_RATE`
    (16000)。本项目实际使用的患者录音本身就是.wav格式, 真实原生采样率
    需要以`sf.read`实际返回的`sr`为准, 之前版本的注释/画图标题里假设
    "统一是16kHz"是不准确的, 已改正——不应该在这里悄悄把返回的`sr`换成
    16000, 那会让画出来的频谱和`localization.py`真正处理的那份波形不是
    同一份数据, 而不是修好了"频谱应该用未裁剪波形"这条要求。
    """
    with as_standard_wav(audio_path) as wav_path:
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.float32)
    return waveform, int(sr)


def _boundary_marker(ax, x: float, direction_symbol: str, theme: dict) -> None:
    """boundary_limited标记: 在贴着录音起始/结束的那一侧画一个指向"录音之外"
    的三角形marker(不是重新画一个新声段框, 只是给已有声段框的那一条边加
    一个额外的视觉提示), y位置用axes分数坐标(跟着子图走, 不受具体数据量纲
    影响)。"""
    ax.plot(
        [x], [0.93], transform=ax.get_xaxis_transform(), marker=direction_symbol,
        markersize=8, color=theme["spine_color"], clip_on=False, zorder=6,
    )


def _draw_segment_overlays(ax, segments: list, theme: dict) -> int:
    """把`suspected_abnormal_segments`列表叠加到一个已有的time-axis子图上。
    borderline用橙黄虚线边框(而不是和clear声段一样的实线), boundary_type
    为left/right/both时在对应一侧加截断marker。返回实际画了几个声段框,
    供调用方/测试核对(diffuse/minimal/unstable传进来的segments本来就是
    空列表, 这里自然画0个, 不是本函数额外判断"要不要画")。"""
    n_drawn = 0
    for seg in segments:
        color = _direction_color(seg.get("direction"), theme)
        is_borderline = bool(seg.get("is_borderline"))
        edge_color = theme["borderline_color"] if is_borderline else color
        ax.axvspan(
            seg["start_seconds"], seg["end_seconds"],
            facecolor=to_rgba(color, theme["segment_fill_alpha"]),
            edgecolor=edge_color,
            linewidth=1.8 if is_borderline else 1.4,
            linestyle="--" if is_borderline else "-",
            zorder=5,
        )
        boundary_type = seg.get("boundary_type", "none")
        if boundary_type in ("left", "both"):
            _boundary_marker(ax, seg["start_seconds"], "<", theme)
        if boundary_type in ("right", "both"):
            _boundary_marker(ax, seg["end_seconds"], ">", theme)
        n_drawn += 1
    return n_drawn


def plot_site_evidence_bars(prediction: dict, theme: str = "paper", figsize=(9, 4)):
    """患者级`predict_patient()`返回值里的`site_contributions` -> 六位点
    有符号贡献条形图。标题固定为"Site-level model evidence"(不称"异常
    程度"——这不是一个经过验证的异常度量, 只是遮挡该位点后患者级概率的
    有符号变化量)。

    `available=False`(未采集)和`available=True`但没能算出delta(通常是
    只提供了1个位点、遮掉它会让mask全零因此跳过贡献度计算, 见
    `inference.py::predict_patient`)两种"没有真实signed_probability_delta
    可画"的情况, 都画成0高度的浅灰网点纹理柱子并标"n/a", 不和真正算出来的
    minimal_influence(有真实数值, 只是很小, 实心灰色无纹理)混在一起。
    """
    _configure_cjk_font()
    th = _get_theme(theme)
    site_contributions = prediction["site_contributions"]
    n = len(site_contributions)
    fig, ax = plt.subplots(figsize=figsize, facecolor=th["figure_facecolor"])
    _style_axes(ax, th)

    xs = np.arange(n)
    for i, c in enumerate(site_contributions):
        delta = c.get("signed_probability_delta")
        if delta is None:
            ax.bar(
                i, 0.0, width=0.6, facecolor=th["na_color"], edgecolor=th["spine_color"],
                hatch="....", linewidth=0.8, alpha=0.6,
            )
            ax.text(
                i, 0.0, "n/a", ha="center", va="bottom", fontsize=8, color=th["muted_text_color"],
            )
            continue
        color = _direction_color(c.get("direction"), th)
        ax.bar(i, delta, width=0.6, facecolor=color, edgecolor=th["spine_color"], linewidth=0.8)

    ax.axhline(0.0, color=th["spine_color"], linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["site"] for c in site_contributions])
    ax.set_ylabel("Δ fibrosis probability (occlusion)")
    ax.set_title("Site-level model evidence")
    _add_disclaimer_footer(fig, th)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_site_localization(
    localization_result: dict,
    waveform: np.ndarray,
    sr: int,
    theme: str = "paper",
    figsize=(10, 7),
    duration_tolerance_seconds: float = 0.25,
):
    """频谱(上) + 有符号时间证据曲线(下, mean±std用fill_between画, 不用
    虚线模拟)共用同一条时间轴, `suspected_abnormal_segments`同时叠加在
    两个子图上。返回`(fig, meta)`, `meta["n_segments_drawn"]`供测试核对
    "diffuse/minimal/unstable不应该画出任何声段框"这类断言, 不用去读图。

    `waveform`/`sr`必须是`load_localization_waveform()`(或定位算法内部
    同样的读取方式)得到的那份未裁剪原始波形, 不接受经OPERA特征提取管线
    裁剪/填充过的版本——下面会用`evidence_curve`的时长做一次粗校验, 时长
    差超过`duration_tolerance_seconds`就直接报错, 不静默画一张时间轴对不上
    的图。
    """
    _configure_cjk_font()
    th = _get_theme(theme)
    curve = localization_result["evidence_curve"]
    time_grid = np.asarray(curve["time_seconds"])
    signed = np.asarray(curve["signed_evidence"])
    std = np.asarray(curve["evidence_std"])
    time_resolution = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.1
    curve_duration_estimate = float(time_grid[-1]) + time_resolution
    waveform_duration = len(waveform) / sr
    if abs(curve_duration_estimate - waveform_duration) > duration_tolerance_seconds:
        raise VisualizationInputError(
            f"传入的waveform时长({waveform_duration:.3f}秒)和localization_result"
            f"的evidence_curve推算时长({curve_duration_estimate:.3f}秒)对不上"
            f"(容差{duration_tolerance_seconds}秒); 请用load_localization_waveform()"
            "重新按定位算法内部同样的方式读取音频, 不要传入经过额外裁剪/重采样的版本"
        )

    mel, effective_fmax = _mel_spectrogram_db(waveform, sr)

    import librosa.display  # noqa: E402  (延迟import, 避免仅用条形图函数时也强制拉起librosa.display)

    fig, (ax_spec, ax_curve) = plt.subplots(
        2, 1, figsize=figsize, facecolor=th["figure_facecolor"], sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    _style_axes(ax_spec, th)
    _style_axes(ax_curve, th)

    img = librosa.display.specshow(
        mel, sr=sr, hop_length=MEL_HOP, x_axis="time", y_axis="mel",
        fmin=MEL_FMIN, fmax=effective_fmax, ax=ax_spec, cmap=th["cmap"],
    )
    cbar = fig.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=th["text_color"], labelsize=8)
    plt.setp(cbar.ax.get_yticklabels(), color=th["text_color"])
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("")
    # `localization_result["site"]`本身已经是完整位点标签(比如"Site 6"), 不
    # 是裸编号——这里直接用, 不再拼一遍"Site "前缀(之前版本拼出过
    # "Site Site 6"这个重复bug)。标题里的采样率是`sr`实际值(真实数据里
    # patient 001这批录音原生就是4000Hz, 不是之前硬编码假设的16kHz——见
    # `load_localization_waveform`docstring), 不再写死"16 kHz"。
    ax_spec.set_title(
        f"{localization_result['site']} · log-mel spectrogram "
        f"(unmodified {sr / 1000:g} kHz waveform, no additional cropping)"
    )

    ax_curve.plot(time_grid, signed, color=th["text_color"], linewidth=1.3, zorder=4)
    ax_curve.fill_between(
        time_grid, signed - std, signed + std,
        color=th["text_color"], alpha=th["std_band_alpha"], linewidth=0, zorder=3,
        label="mean ± std (3 masking repeats × 3 model initializations)",
    )
    ax_curve.axhline(0.0, color=th["spine_color"], linewidth=0.8)
    ax_curve.set_xlabel("Time (s)")
    ax_curve.set_ylabel("Δ fibrosis probability (signed)")
    ax_curve.legend(loc="upper right", fontsize=7.5, facecolor=th["figure_facecolor"],
                     edgecolor=th["grid_color"], labelcolor=th["text_color"])

    n_drawn_spec = _draw_segment_overlays(ax_spec, localization_result["suspected_abnormal_segments"], th)
    n_drawn_curve = _draw_segment_overlays(ax_curve, localization_result["suspected_abnormal_segments"], th)
    assert n_drawn_spec == n_drawn_curve

    ax_spec.set_xlim(0, waveform_duration)
    ax_curve.set_xlim(0, waveform_duration)

    pattern = localization_result["evidence_pattern"]
    note = localization_result.get("note")
    caption = f"evidence pattern: {pattern}"
    if note:
        caption += f" — {note}"
    fig.text(0.01, 0.965, caption, ha="left", va="top", fontsize=9, color=th["text_color"])

    _add_disclaimer_footer(fig, th)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    return fig, {"n_segments_drawn": n_drawn_spec, "waveform_duration_seconds": waveform_duration}


def _mel_spectrogram_db(waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """返回(log-mel频谱dB, 实际使用的fmax)。

    `MEL_FMAX`(8000)是给16kHz量级音频准备的上限, 不能对任何`sr`都原样使用
    ——采样率的奈奎斯特频率是`sr/2`, `fmax`超过它会导致对应的mel band在这个
    `sr`下必然覆盖不到任何真实FFT频点(空滤波器), 这正是之前"Empty filters"
    报错的根本原因(patient 001 Site 4这批录音本身是4000Hz原生采样率, 不是
    之前文档假设的16kHz——`load_localization_waveform`不做强制重采样, 见
    该函数docstring)。这里按实际传入的`sr`把`fmax`钳制到`min(MEL_FMAX,
    sr/2)`, 而不是简单调低`n_mels`——n_mels调低治标不治本, 只要fmax仍然
    超过奈奎斯特频率, 高mel band依然会是空的。"""
    import librosa  # noqa: E402  (延迟import, 原因同上)

    effective_fmax = min(float(MEL_FMAX), sr / 2.0)

    # 先独立算一次mel滤波器组本身、显式检查有没有空滤波器, 不是等
    # librosa.feature.melspectrogram内部打"Empty filters"警告就当作可以
    # 忽略静默出图——warning很容易在批量跑图/日志里被淹没, 这里改成直接
    # 报错, 强制调用方(或本模块以后改MEL_*参数的人)正视这个问题。
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=MEL_N_FFT, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=effective_fmax,
    )
    empty_filters = np.where(mel_basis.sum(axis=1) == 0)[0]
    if len(empty_filters) > 0:
        raise VisualizationInputError(
            f"mel滤波器组存在{len(empty_filters)}个空滤波器(索引{empty_filters.tolist()}), "
            f"当前参数sr={sr}, n_fft={MEL_N_FFT}, n_mels={MEL_N_MELS}, fmin={MEL_FMIN}, "
            f"fmax={effective_fmax}(已按sr/2钳制); 说明这组参数在这个采样率下仍然不合理, "
            "不能带着这个问题静默出图。请调低n_mels、或调大n_fft、或检查传入的sr是否正确。"
        )
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=effective_fmax, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max), effective_fmax


def save_figure(fig, path: Union[str, Path], dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def collect_figure_texts(fig) -> list:
    """取出图上所有文字(免责声明、标题、坐标轴标签、刻度、图例), 供测试
    检查禁用措辞没有被本模块新加的任何文案带进来。逐一显式遍历
    `fig.texts`/`ax.get_title()`/`get_xlabel()`/`get_ylabel()`/`ax.texts`/
    刻度标签/图例, 而不是用`findobj`笼统扫全部Text子类——`findobj`对
    Axis(刻度标签的实际归属对象)子树的递归行为在不同matplotlib版本上不够
    稳定, 显式遍历更可靠, 也让"这个函数到底查了哪些地方的文字"一目了然。"""
    texts = []
    texts.extend(t.get_text() for t in fig.texts)
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        texts.extend(t.get_text() for t in ax.texts)
        texts.extend(t.get_text() for t in ax.get_xticklabels())
        texts.extend(t.get_text() for t in ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(t.get_text() for t in legend.get_texts())
    return [t for t in texts if t]


def assert_disclaimer_present(fig) -> None:
    """免责声明必须逐字、原样出现在图上——不是"包含关系"意义上的宽松检查,
    而是图上确实存在完全等于`SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER`原文的
    一条文字, 防止以后有人为了绕开下面的禁用措辞扫描而悄悄改写措辞。"""
    if SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER not in collect_figure_texts(fig):
        raise AssertionError("图上没有找到完整、逐字的免责声明原文(SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER)")


def assert_no_forbidden_wording(fig) -> None:
    """禁用措辞扫描, 跳过和免责声明原文*逐字完全相同*的那一条文字——免责
    声明本身在否定意义上引用"病理异常声段"做澄清("...尚不等同于经临床
    医生确认的病理异常声段"), 这是`schemas.py`里已经明确的既定豁免(见该
    模块顶部注释和`tests/test_localization.py::test_no_forbidden_wording`
    的先例)。要求逐字完全等于免责声明原文才跳过
    (不是"文字里包含免责声明"就跳过), 避免免责声明被意外拼接进其他自定义
    文案、把那条文案也一起放过检查。"""
    texts = collect_figure_texts(fig)
    for text in texts:
        if text == SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER:
            continue
        for term in FORBIDDEN_TERMS:
            if term in text:
                raise AssertionError(f"图中出现禁用措辞: {term}; text={text!r}")
