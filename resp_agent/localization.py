# -*- coding: utf-8 -*-
"""Localize respiratory-sound intervals that influence patient-level predictions.

术语与免责声明——首次出现必须带(见`schemas.SUSPECTED_ABNORMAL_SEGMENT_
DISCLAIMER`, 本模块所有输出都会把它放进`term_disclaimer`字段):
    可疑异常声段是指遮挡后会显著改变患者级预测概率的呼吸音时间段, 属于
    模型定位结果, 尚不等同于经临床医生确认的病理异常声段。
禁止使用"confirmed abnormal segment"/"病理异常声段"/"确诊异常声段"这类
措辞(`schemas.FORBIDDEN_TERMS`), 有专门测试检查本模块产出的所有字符串
不含这些词。

Method:

    1) 先跑一次患者级预测(`inference.predict_patient`, 内含位点贡献度), 用
       来选定目标位点、拿到用于报告的ensemble级`fibrosis_probability`。
    2) 默认选absolute_importance最大的可用位点, 或用`site=`手动指定。
    3) 对该位点音频做2秒窗/0.5秒步长滑窗, 每个窗口用确定性低电平噪声遮挡
       (种子由(音频哈希, 窗口起止时间, 重复序号)派生), 重复3次, 且对3个
       患者模型初始化*分别*取原始概率(不在这一步平均), 得到每个窗口
       3(repeat)x3(init)=9个原始signed_delta样本并完整保留(不只存均值),
       用来算evidence_std/direction_consistency和"峰值位置是否稳定"这类
       跨初始化的稳定性判断。
    4) 用window-level(每个2秒窗一个值)数组做峰值检测——注意不是在0.1秒
       细分辨率曲线上直接找峰: 因为hop=0.5秒意味着相邻0.1秒采样点在同一个
       窗口覆盖集合不变的区间内取值完全相同(平台), scipy.signal.find_peaks
       默认不认平台为峰, 所以峰值检测在window级(13个点/8秒音频)数组上做,
       0.1秒分辨率的连续曲线只用于数据导出和可视化, 不参与分类
       判断逻辑。
    5) 按四个预设工程判据(峰值绝对影响>=下限、prominence>=全曲线最大值的
       25%、峰值/中位数>=1.5、局部方向一致率>=80%; 这四个阈值都是未经临床
       时间段标注验证的工程参数, 不是医学意义上的诊断标准)筛focal可疑
       异常声段; 否则按稳定性/整体幅度归入unstable/minimal/diffuse。

Visualization and report generation are handled by separate modules.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .audio_io import as_standard_wav
from .inference import build_site_features, predict_patient
from .opera_encoder import extract_opera_features_batch
from .patient_model import DeploymentModel, score_segment_classifier
from .schemas import (
    DIRECTION_MINIMAL,
    DIRECTION_OPPOSES,
    DIRECTION_SUPPORTS,
    SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
    classify_direction,
)

CONFIG_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "localization.yaml"

TARGET_CLASS = "fibrotic"
MASKING_METHOD = "deterministic_low_energy_noise"

EVIDENCE_PATTERN_FOCAL = "focal"
EVIDENCE_PATTERN_DIFFUSE = "diffuse"
EVIDENCE_PATTERN_MINIMAL = "minimal"
EVIDENCE_PATTERN_UNSTABLE = "unstable"

# focal声段的文字等级(见_classify_evidence_pattern里的minimum_relative_
# criterion_margin/is_borderline计算): "borderline"=4项预设工程判据里最
# 紧的那项相对余量低于cfg["borderline_relative_margin"], "clear"=否。
# 不是医学意义上的分级, 只是把is_borderline翻译成一个人可读的文字等级。
FOCALITY_STRENGTH_BORDERLINE = "borderline"
FOCALITY_STRENGTH_CLEAR = "clear"

_PATTERN_NOTES = {
    EVIDENCE_PATTERN_DIFFUSE: "模型证据分布在整段录音中, 未发现明确局部集中的可疑异常声段。",
    EVIDENCE_PATTERN_MINIMAL: "未发现对患者级预测产生明显影响的可疑异常声段。",
    EVIDENCE_PATTERN_UNSTABLE: "当前时间段定位结果稳定性不足, 暂不输出明确可疑异常声段。",
}

DEFAULT_CONFIG = {
    "window_seconds": 2.0,
    "hop_seconds": 0.5,
    "time_resolution_seconds": 0.1,
    "masking_repeats": 3,
    "minimal_absolute_effect": 0.02,
    "peak_prominence_ratio": 0.25,
    "peak_to_median_ratio": 1.5,
    "minimum_direction_consistency": 0.8,
    "maximum_segments": 3,
    "max_relative_std": 0.5,
    "max_peak_location_spread_seconds": 2.0,
    "minimum_tail_window_seconds": 0.5,
    "background_frame_seconds": 0.05,
    "peak_width_rel_height": 0.5,
    # focal候选峰的4项判据(absolute_effect/prominence/peak_to_median_ratio/
    # direction_consistency)里, 任意一项相对阈值的"相对余量" =
    # (raw_value - threshold) / threshold 低于这个值就标记is_borderline=True。
    # 不影响focal/diffuse/minimal/unstable四类顶层判定本身(那部分逻辑不变),
    # 只是给已经判成focal的声段附加"离阈值有多近"这个额外说明字段。初始
    # 工程参数(0.2 = 20%), 未经临床验证, 修改前请理解下方is_borderline的
    # 计算方式。不得为了让某个具体病例呈现特定结果而反向调整这个值。
    "borderline_relative_margin": 0.2,
}
REQUIRED_CONFIG_KEYS = tuple(DEFAULT_CONFIG.keys())


class LocalizationInputError(ValueError):
    pass


class LocalizationConfigError(RuntimeError):
    """配置文件缺失或缺配置项时抛这个。正式调用(`localize_suspected_
    abnormal_segments`不显式传`config=`时)禁止静默回退到内置默认值——
    `DEFAULT_CONFIG`只是文档/测试用的参考值, 不是生产兜底。"""


def load_localization_config(config_path: Optional[Path] = None, allow_fallback: bool = False) -> dict:
    """所有工程参数都从这里读, 不硬编码在函数体里(见configs/localization.yaml)。

    `allow_fallback=False`(默认, 正式调用路径用这个): 配置文件不存在, 或
    存在但缺任何一个必需字段, 都直接抛`LocalizationConfigError`, 不静默
    退回`DEFAULT_CONFIG`。`allow_fallback=True`仅供开发/测试脚本主动选择
    "拿一份参考默认值"时使用, 不是正式推理路径会走到的分支。
    """
    path = Path(config_path) if config_path else CONFIG_PATH_DEFAULT
    if not path.exists():
        if allow_fallback:
            return dict(DEFAULT_CONFIG)
        raise LocalizationConfigError(
            f"定位配置文件不存在: {path}; 正式调用不允许静默回退到内置默认值, "
            "请确认configs/localization.yaml存在, 或显式传入config=dict(...)"
        )
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in loaded]
    if missing:
        if allow_fallback:
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(loaded)
            return cfg
        raise LocalizationConfigError(
            f"{path} 缺少必需的配置项: {missing}; 正式调用不允许用内置默认值静默补齐"
        )
    return {k: loaded[k] for k in REQUIRED_CONFIG_KEYS}


# Deterministic audio masking
def _audio_hash(wav_path: Path) -> str:
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _generate_windows(
    duration_sec: float, window_seconds: float, hop_seconds: float, minimum_tail_window_seconds: float,
) -> List[Tuple[float, float]]:
    windows = []
    start = 0.0
    while start < duration_sec:
        end = min(start + window_seconds, duration_sec)
        if end - start >= min(minimum_tail_window_seconds, window_seconds):
            windows.append((start, end))
        if end >= duration_sec:
            break
        start += hop_seconds
    return windows


def _estimate_background_noise_level(waveform: np.ndarray, sr: int, frame_seconds: float) -> float:
    frame_len = max(1, int(sr * frame_seconds))
    n_frames = max(1, len(waveform) // frame_len)
    if n_frames <= 1:
        return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)) * 0.1 + 1e-6)
    frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return float(max(np.percentile(rms, 10), 1e-6))


def _mask_window_with_noise(
    waveform: np.ndarray, sr: int, start_sec: float, end_sec: float, noise_level: float, seed: int,
) -> np.ndarray:
    masked = waveform.copy()
    start_idx = int(round(start_sec * sr))
    end_idx = min(int(round(end_sec * sr)), len(waveform))
    if end_idx <= start_idx:
        return masked
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_level, size=end_idx - start_idx).astype(waveform.dtype)
    masked[start_idx:end_idx] = noise
    return masked


def _deterministic_seed(audio_hash: str, start_sec: float, end_sec: float, repeat_idx: int) -> int:
    key = f"{audio_hash}:{start_sec:.3f}:{end_sec:.3f}:{repeat_idx}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ============ 逐初始化(不平均)前向推断 ============
def _run_per_init_probs(deployment: DeploymentModel, combined: np.ndarray, mask: np.ndarray) -> List[float]:
    """返回3个患者模型初始化各自的(未平均)纤维化概率。

    这里特意在本模块内复刻`inference.run_ensemble`的前向传播逻辑, 而不是
    import它并在外面做`[run_ensemble(...) for m in models]`——因为定位结果的
    执行范围明确只允许改`localization.py`/`schemas.py`, 不允许改
    `inference.py`, 而这里需要的是*没有*被`run_ensemble`内部`np.mean()`掉
    的逐初始化原始值, 用来算evidence_std/direction_consistency和跨初始化
    的峰值位置稳定性, `inference.py`当前的公开接口不提供这个粒度。
    """
    if not np.any(mask):
        raise LocalizationInputError("模型推理至少需要一个有效位点(mask全为0, 没有任何可用输入)")
    feat_t = torch.from_numpy(combined).unsqueeze(0).to(deployment.device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(deployment.device)
    probs = []
    with torch.no_grad():
        for model in deployment.models:
            logits = model(feat_t, mask_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs.append(float(torch.softmax(logits, dim=-1)[0, 1].item()))
    return probs


def _select_target_site(
    prediction: dict, available_sites: List[str], site: Optional[str]
) -> Tuple[str, str]:
    if site is not None:
        if site not in available_sites:
            raise LocalizationInputError(
                f"指定的site={site!r}不在已采集位点里(available_sites={available_sites})"
            )
        return site, "user_specified"
    if len(available_sites) == 1:
        return available_sites[0], "only_available_site"
    ranked = [
        c for c in prediction["site_contributions"]
        if c["available"] and c["absolute_importance"] is not None
    ]
    if not ranked:
        raise LocalizationInputError(
            "没有任何位点算出了有效的absolute_importance, 无法自动选择要定位的位点, "
            "请通过site参数手动指定"
        )
    ranked.sort(key=lambda c: c["absolute_importance"], reverse=True)
    return ranked[0]["site"], "top_absolute_importance"


def _build_continuous_curve(
    window_stats: List[dict], duration: float, time_resolution: float
) -> dict:
    """把每个窗口的(signed_delta, std, direction_consistency)按覆盖时间分配到
    固定分辨率的连续时间轴上, 重叠部分取平均。只用于数据导出和可视化,
    不参与本阶段的分类判断——判断逻辑见`_classify_evidence_pattern`, 用的是
    window级(未插值)数组, 原因见模块docstring第4点。"""
    n_points = int(np.floor(duration / time_resolution)) + 1
    time_grid = np.round(np.arange(n_points) * time_resolution, 6)
    signed_evidence = np.zeros(n_points)
    evidence_std = np.zeros(n_points)
    # 必须从0初始化, 不能用1.0"假装完全一致"当默认值——direction_consistency
    # 是"覆盖该时间点的窗口里样本符号一致的比例", 定义域是[0,1], 用1.0初始化
    # 会导致累加平均之后的值可能超出[0,1](例如被1个consistency=0.5的窗口
    # 覆盖时, 旧写法算出(1.0+0.5)/1=1.5, 明显错误)。
    direction_consistency = np.zeros(n_points)
    coverage = np.zeros(n_points, dtype=int)
    for w in window_stats:
        covered = (time_grid >= w["start_seconds"]) & (time_grid < w["end_seconds"])
        signed_evidence[covered] += w["signed_delta"]
        evidence_std[covered] += w["std"]
        direction_consistency[covered] += w["direction_consistency"]
        coverage[covered] += 1
    valid = coverage > 0
    signed_evidence[valid] /= coverage[valid]
    evidence_std[valid] /= coverage[valid]
    direction_consistency[valid] /= coverage[valid]
    return {
        "time_seconds": time_grid.tolist(),
        "signed_evidence": signed_evidence.tolist(),
        "absolute_evidence": np.abs(signed_evidence).tolist(),
        "evidence_std": evidence_std.tolist(),
        "direction_consistency": direction_consistency.tolist(),
    }


def _boundary_peak_prominence_and_bounds(values: np.ndarray, idx: int, rel_height: float):
    """手动计算边界峰(idx=0或idx=len(values)-1)的prominence和半高宽边界,
    只用存在的那一侧真实数据。

    scipy的`peak_prominences`/`peak_widths`在算法设计上假设peak两侧都存在
    真实数据; 边界点缺一侧, 直接喂给它们会把"缺失的那一侧"退化处理成
    "以peak自己为界", 导致prominence恒为0(用真实数据验证过, 不是猜测)。
    一开始尝试过"在数组两端垫一个哨兵值再整体交给scipy"的办法, 但验证
    发现这会连带把*所有*会自然搜索到数组边缘的内部峰(比如信号里只有1个
    窗口明显突出、其余全是低平baseline那种情况, 这在本模块的实际数据里
    很常见)的prominence也一起算错——哨兵不管垫多低, 都会被内部峰的谷值
    搜索当成"更深的谷底"用上, 使prominence被人为拉高、half-height阈值被
    拉到负数、宽度算成整个数组。所以改成两条路径分开处理: 真正的内部峰
    两侧都有数据的内部峰交给scipy原生函数处理;
    只有数组首尾这两个特例, 才用这个函数手动按"向存在的那一侧走, 直到
    遇到更高点或数组边界, 记录途中最低点当谷底; half-height阈值处线性
    插值找边界"的方式单独算, 和scipy内部算法思路一致, 只是省去了"两侧都
    要有效"这个scipy没有为这种输入设计过的前提。
    """
    n = len(values)
    peak_val = float(values[idx])
    if idx == 0:
        walk = range(1, n)
    else:
        walk = range(n - 2, -1, -1)
    valley = peak_val
    for j in walk:
        if values[j] >= peak_val:
            break
        valley = min(valley, float(values[j]))
    prominence = peak_val - valley

    height_threshold = peak_val - rel_height * prominence
    if idx == 0:
        inner_ip = float(n - 1)
        for j in range(1, n):
            if values[j] < height_threshold:
                prev = float(values[j - 1])
                cur = float(values[j])
                inner_ip = (j - 1) + ((prev - height_threshold) / (prev - cur) if prev != cur else 0.0)
                break
            inner_ip = float(j)
        left_ip, right_ip = 0.0, inner_ip
    else:
        inner_ip = 0.0
        for j in range(n - 2, -1, -1):
            if values[j] < height_threshold:
                nxt = float(values[j + 1])
                cur = float(values[j])
                inner_ip = (j + 1) - ((nxt - height_threshold) / (nxt - cur) if nxt != cur else 0.0)
                break
            inner_ip = float(j)
        left_ip, right_ip = inner_ip, float(n - 1)
    return float(prominence), left_ip, right_ip


def _find_window_peaks(window_abs: np.ndarray, height: float, rel_height: float):
    """window级绝对值数组上找峰。内部峰(两侧都有数据)完全走scipy原生
    `find_peaks`+`peak_prominences`+`peak_widths`, 不做任何改动。数组首尾
    两个边界点单独检查是否满足"比唯一的那个邻居更高"的局部极大条件, 满足
    则用`_boundary_peak_prominence_and_bounds`手动算(原因见该函数docstring:
    scipy的prominence/width函数没有为只有一侧数据的输入设计过, 边界点直接
    退化成prominence=0, 会漏掉真实存在的边界峰——patient 001 Site 4的真实
    数据里, `strongest_window`就正好落在第0个窗口, 不是假设性场景)。"""
    n = len(window_abs)
    peak_idx_list: List[int] = []
    prominences_list: List[float] = []
    left_ips_list: List[float] = []
    right_ips_list: List[float] = []

    peaks_interior, _ = find_peaks(window_abs, height=height)
    if len(peaks_interior) > 0:
        prom_i, left_bases, right_bases = peak_prominences(window_abs, peaks_interior)
        _, _, left_ips_i, right_ips_i = peak_widths(
            window_abs, peaks_interior, rel_height=rel_height,
            prominence_data=(prom_i, left_bases, right_bases),
        )
        for k in range(len(peaks_interior)):
            peak_idx_list.append(int(peaks_interior[k]))
            prominences_list.append(float(prom_i[k]))
            left_ips_list.append(float(left_ips_i[k]))
            right_ips_list.append(float(right_ips_i[k]))

    if n >= 2:
        if window_abs[0] > window_abs[1] and window_abs[0] >= height and 0 not in peak_idx_list:
            prom, li, ri = _boundary_peak_prominence_and_bounds(window_abs, 0, rel_height)
            peak_idx_list.append(0)
            prominences_list.append(prom)
            left_ips_list.append(li)
            right_ips_list.append(ri)
        if window_abs[-1] > window_abs[-2] and window_abs[-1] >= height and (n - 1) not in peak_idx_list:
            prom, li, ri = _boundary_peak_prominence_and_bounds(window_abs, n - 1, rel_height)
            peak_idx_list.append(n - 1)
            prominences_list.append(prom)
            left_ips_list.append(li)
            right_ips_list.append(ri)
    elif n == 1 and window_abs[0] >= height:
        peak_idx_list.append(0)
        prominences_list.append(float(window_abs[0]))
        left_ips_list.append(0.0)
        right_ips_list.append(0.0)

    if not peak_idx_list:
        empty = np.array([], dtype=np.intp)
        return empty, np.array([]), np.array([]), np.array([])

    order = np.argsort(peak_idx_list)
    peaks = np.asarray(peak_idx_list, dtype=np.intp)[order]
    prominences = np.asarray(prominences_list)[order]
    left_ips = np.asarray(left_ips_list)[order]
    right_ips = np.asarray(right_ips_list)[order]
    return peaks, prominences, left_ips, right_ips


def _peak_location_spread_seconds(
    raw_samples: Dict[int, List[float]], windows: List[Tuple[float, float]],
    masking_repeats: int, window_seconds: float,
) -> float:
    """跨初始化的峰值位置稳定性: 分别对3个初始化各自的window级曲线(只跨
    masking_repeats取平均, 不跨init平均)找各自的绝对值最大窗口, 比较3个
    初始化找到的窗口起始时间彼此相差多少。**只应该在已经存在>=1个通过基础
    4项判据的候选峰之后才调用这个函数**(见`_classify_evidence_pattern`)——
    对一条本来就没有明显峰、证据分散在整段录音里的diffuse曲线, 3个初始化
    各自的"全局绝对值最大窗口"本来就容易因为噪声落在不同位置, 这不代表
    "定位结果不稳定", 只是"这条曲线本来就没有一个真正的峰"; 如果不加这个
    前置条件, 会把正常的diffuse曲线误判成unstable。"""
    n_windows = len(windows)
    peak_time_per_init: List[float] = []
    for init_idx in range(3):
        curve_i = np.zeros(n_windows)
        for w_i in range(n_windows):
            samples = np.array(raw_samples[w_i]).reshape(masking_repeats, 3)
            curve_i[w_i] = samples[:, init_idx].mean()
        if np.abs(curve_i).max() < 1e-12:
            continue
        peak_w = int(np.argmax(np.abs(curve_i)))
        peak_time_per_init.append(windows[peak_w][0] + window_seconds / 2)
    if len(peak_time_per_init) < 2:
        return 0.0
    return float(max(peak_time_per_init) - min(peak_time_per_init))


_DIRECTION_NOTE_ZH = {
    DIRECTION_SUPPORTS: "支持纤维化预测方向(遮挡该声段后患者级纤维化概率下降)",
    DIRECTION_OPPOSES: "反对纤维化预测方向(遮挡该声段后患者级纤维化概率上升)",
    DIRECTION_MINIMAL: "方向性较弱, 遮挡该声段对患者级概率影响很小",
}


def _relative_margin(value: float, threshold: float) -> float:
    """(value - threshold) / threshold。value/threshold都已经是"判据满足即
    通过"方向(越大越容易通过focal候选), 调用方保证传入时value已经满足
    value>=threshold(候选峰能进入这个函数说明4项判据都已经通过, 见
    `_classify_evidence_pattern`), 所以正常情况下返回值>=0。threshold<=0
    是本模块配置里不会出现的退化情况(minimal_absolute_effect/
    peak_prominence_ratio*max_abs/peak_to_median_ratio/
    minimum_direction_consistency按配置定义都应为正), 这里仍做防御性处理,
    不让除零直接抛异常中断整个定位流程。"""
    if threshold > 0:
        return (value - threshold) / threshold
    return float("inf") if value > threshold else 0.0


def _segment_boundary_info(li: int, ri: int, n_windows: int) -> Tuple[bool, str]:
    touches_left = li <= 0
    touches_right = ri >= n_windows - 1
    if touches_left and touches_right:
        return True, "both"
    if touches_left:
        return True, "left"
    if touches_right:
        return True, "right"
    return False, "none"


def _segment_caveats(is_borderline: bool, direction: str, boundary_type: str) -> List[str]:
    """给已经判定为focal的候选声段附加说明性文字, 不参与任何分类判断,
    纯粹是把is_borderline/boundary信息翻译成人可读的一句话, 内容完全由
    传入的参数决定, 不针对任何特定病例硬编码。"""
    caveats: List[str] = []
    if is_borderline:
        caveats.append(
            "该可疑异常声段的局灶性证据接近预设工程判据阈值(borderline; "
            "该阈值未经临床时间段标注验证), 结果对参数选择可能较敏感, "
            "建议结合其他证据谨慎解读。"
        )
    if boundary_type in ("left", "both"):
        caveats.append(
            "该声段左侧紧贴录音起始时间, 无法排除相关声学证据的真实起点"
            "早于当前录音开始时间。"
        )
    if boundary_type in ("right", "both"):
        caveats.append(
            "该声段右侧紧贴录音结束时间, 无法排除相关声学证据的真实终点"
            "晚于当前录音结束时间。"
        )
    direction_note = _DIRECTION_NOTE_ZH.get(direction)
    if direction_note and (is_borderline or boundary_type != "none"):
        caveats.append(f"方向说明: {direction_note}。")
    return caveats


def _classify_evidence_pattern(
    window_stats: List[dict], raw_samples: Dict[int, List[float]],
    windows: List[Tuple[float, float]], masking_repeats: int, window_seconds: float, cfg: dict,
) -> Tuple[str, List[dict], dict]:
    """window级(未插值)数组上做全部分类判断。判定顺序(已按审查意见调整):

        1) minimal优先——效应本身微小时, 不再看方向一致性/方差这些在小
           数值上天然更容易被噪声放大的指标, 直接归minimal, 不下"不稳定"
           这种更强的结论。
        2) 全局不稳定(方向一致率/方差比例, 不含峰值位置——峰值位置稳定性
           只在下面第4步"已经有候选峰"时才检查, 见`_peak_location_spread_
           seconds`的说明)。
        3) 用window级数组(含首尾边界峰, 见`_find_window_peaks`)找
           focal候选峰, 4项基础判据(峰值绝对值/prominence占比/峰值中位数
           比/局部方向一致率)。
        4) 如果找到>=1个候选峰, 再检查这些候选峰的位置跨3个初始化是否
           稳定, 不稳定则整体降级为unstable; 稳定则focal。
        5) 没有任何候选峰(基础4项判据都不满足)则diffuse——"方向基本一致
           但没有局灶峰"的默认归类。

    返回(pattern, suspected_segments, stability_stats)。"""
    n_windows = len(window_stats)
    window_signed = np.array([w["signed_delta"] for w in window_stats])
    window_abs = np.abs(window_signed)
    window_std = np.array([w["std"] for w in window_stats])
    window_consistency = np.array([w["direction_consistency"] for w in window_stats])

    max_abs = float(window_abs.max()) if n_windows else 0.0
    median_abs = float(np.median(window_abs)) if n_windows else 0.0
    overall_direction_consistency = float(np.mean(window_consistency)) if n_windows else 1.0
    overall_relative_std = float(np.mean(window_std) / (max_abs + 1e-9)) if n_windows else 0.0

    stability = {
        "overall_direction_consistency": overall_direction_consistency,
        "overall_relative_std": overall_relative_std,
        "peak_location_spread_seconds": None,  # 只有走到"已有候选峰"分支才会被填上具体数值
    }

    if max_abs < cfg["minimal_absolute_effect"]:
        return EVIDENCE_PATTERN_MINIMAL, [], stability

    is_globally_unstable = (
        overall_direction_consistency < cfg["minimum_direction_consistency"]
        or overall_relative_std > cfg["max_relative_std"]
    )
    if is_globally_unstable:
        return EVIDENCE_PATTERN_UNSTABLE, [], stability

    peaks, prominences, left_ips, right_ips = _find_window_peaks(
        window_abs, cfg["minimal_absolute_effect"], cfg["peak_width_rel_height"]
    )
    candidates: List[dict] = []
    if len(peaks) > 0:
        for p_i, peak_idx in enumerate(peaks):
            peak_val = float(window_abs[peak_idx])
            if peak_val < cfg["minimal_absolute_effect"]:
                continue
            if float(prominences[p_i]) < cfg["peak_prominence_ratio"] * max_abs:
                continue
            if median_abs <= 0 or peak_val / median_abs < cfg["peak_to_median_ratio"]:
                continue
            local_consistency = float(window_consistency[peak_idx])
            if local_consistency < cfg["minimum_direction_consistency"]:
                continue

            peak_sign = np.sign(window_signed[peak_idx])
            left_bound = max(int(np.floor(left_ips[p_i])), 0)
            right_bound = min(int(np.ceil(right_ips[p_i])), n_windows - 1)
            # 半高宽边界和"不跨越符号翻转点"取交集(更窄的那个), 保证相反
            # 方向的峰值不会被强行并进同一个区间。
            li = peak_idx
            while li > left_bound and np.sign(window_signed[li - 1]) == peak_sign:
                li -= 1
            ri = peak_idx
            while ri < right_bound and np.sign(window_signed[ri + 1]) == peak_sign:
                ri += 1

            # ---- 以下只是把上面已经用来做accept/reject判断的4项原始值/
            # 阈值重新打包成可读字段 + 附加is_borderline/focality_strength/
            # 边界信息, 不改变上面任何一个continue/accept分支的判断逻辑。
            prominence_value = float(prominences[p_i])
            prominence_threshold = cfg["peak_prominence_ratio"] * max_abs
            peak_to_median_value = peak_val / median_abs
            criteria = {
                "absolute_effect": {
                    "value": peak_val,
                    "threshold": cfg["minimal_absolute_effect"],
                    "margin": peak_val - cfg["minimal_absolute_effect"],
                    "relative_margin": _relative_margin(peak_val, cfg["minimal_absolute_effect"]),
                },
                "prominence": {
                    "value": prominence_value,
                    "threshold": prominence_threshold,
                    "margin": prominence_value - prominence_threshold,
                    "relative_margin": _relative_margin(prominence_value, prominence_threshold),
                },
                "peak_to_median_ratio": {
                    "value": peak_to_median_value,
                    "threshold": cfg["peak_to_median_ratio"],
                    "margin": peak_to_median_value - cfg["peak_to_median_ratio"],
                    "relative_margin": _relative_margin(peak_to_median_value, cfg["peak_to_median_ratio"]),
                },
                "direction_consistency": {
                    "value": local_consistency,
                    "threshold": cfg["minimum_direction_consistency"],
                    "margin": local_consistency - cfg["minimum_direction_consistency"],
                    "relative_margin": _relative_margin(local_consistency, cfg["minimum_direction_consistency"]),
                },
            }
            # minimum_relative_criterion_margin: 4项预设工程判据里relative_
            # margin最小的那一项(即"离被拒绝最近的那道判据还剩多少余量"),
            # 数值型, 供需要精确数字的调用方使用。focality_strength是把这个
            # 数字翻译成的文字等级(borderline/clear), 供只需要粗粒度判断的
            # 调用方使用——两者定义完全一致, 只是数值/文字两种呈现形式,
            # 不是两套独立逻辑。
            minimum_relative_criterion_margin = min(item["relative_margin"] for item in criteria.values())
            is_borderline = minimum_relative_criterion_margin < cfg["borderline_relative_margin"]
            focality_strength = FOCALITY_STRENGTH_BORDERLINE if is_borderline else FOCALITY_STRENGTH_CLEAR

            touches_recording_boundary, boundary_type = _segment_boundary_info(li, ri, n_windows)
            direction = classify_direction(float(window_signed[peak_idx]))

            candidates.append(
                {
                    "start_seconds": round(float(windows[li][0]), 3),
                    "end_seconds": round(float(windows[ri][1]), 3),
                    "peak_time_seconds": round(float(windows[peak_idx][0] + window_seconds / 2), 3),
                    "peak_signed_probability_delta": float(window_signed[peak_idx]),
                    "importance_score": peak_val,
                    "direction": direction,
                    "direction_consistency": local_consistency,
                    "evidence_std": float(window_std[peak_idx]),
                    "criteria": criteria,
                    "minimum_relative_criterion_margin": minimum_relative_criterion_margin,
                    "focality_strength": focality_strength,
                    "is_borderline": is_borderline,
                    "touches_recording_boundary": touches_recording_boundary,
                    "boundary_type": boundary_type,
                    "boundary_limited": touches_recording_boundary,
                    "caveats": _segment_caveats(is_borderline, direction, boundary_type),
                }
            )

    if not candidates:
        return EVIDENCE_PATTERN_DIFFUSE, [], stability

    candidates.sort(key=lambda c: c["importance_score"], reverse=True)

    # 只有已经找到候选峰时才评估跨初始化的峰值位置稳定性(见
    # `_peak_location_spread_seconds`的说明), 不对diffuse曲线做这个检查。
    peak_location_spread = _peak_location_spread_seconds(
        raw_samples, windows, masking_repeats, window_seconds
    )
    stability["peak_location_spread_seconds"] = peak_location_spread
    if peak_location_spread > cfg["max_peak_location_spread_seconds"]:
        return EVIDENCE_PATTERN_UNSTABLE, [], stability

    return EVIDENCE_PATTERN_FOCAL, candidates[: cfg["maximum_segments"]], stability


def localize_suspected_abnormal_segments(
    deployment: DeploymentModel,
    site_audio: Dict[str, str],
    patient_info: Optional[dict] = None,
    site: Optional[str] = None,
    config: Optional[dict] = None,
) -> dict:
    """Localize influential intervals for one selected recording site."""
    # 正式调用路径(config=None时)严格要求配置文件存在且字段齐全, 不静默
    # 回退到DEFAULT_CONFIG(见load_localization_config的allow_fallback说明)。
    # 显式传config=dict(...)的调用方(比如测试)自己负责config内容, 不走
    # 文件加载/校验逻辑。
    cfg = config if config is not None else load_localization_config()
    window_seconds = cfg["window_seconds"]
    hop_seconds = cfg["hop_seconds"]
    time_resolution = cfg["time_resolution_seconds"]
    masking_repeats = cfg["masking_repeats"]

    prediction = predict_patient(deployment, site_audio, patient_info=patient_info)
    combined, mask, available_sites, missing_sites = build_site_features(deployment, site_audio)
    target_site, selection_reason = _select_target_site(prediction, available_sites, site)
    target_idx = deployment.site_order.index(target_site)

    original_per_init = _run_per_init_probs(deployment, combined, mask)

    limitations = ["可疑异常声段为模型定位结果", "尚未经过临床医生时间段标注确认"]
    if len(available_sites) == 1:
        limitations.append(
            "当前只提供了1个位点音频, 本结果只能做segment层面的时间定位分析, "
            "不能替代完整6位点的患者级诊断"
        )

    with as_standard_wav(site_audio[target_site]) as wav_path:
        audio_hash = _audio_hash(wav_path)
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1).astype(np.float32)
        duration = len(waveform) / sr
        noise_level = _estimate_background_noise_level(waveform, sr, cfg["background_frame_seconds"])
        windows = _generate_windows(duration, window_seconds, hop_seconds, cfg["minimum_tail_window_seconds"])
        if not windows:
            raise LocalizationInputError(f"{target_site}音频时长{duration:.2f}秒, 生成不出任何窗口")

        with tempfile.TemporaryDirectory(prefix="resp_agent_localization_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            variant_paths: List[Path] = []
            variant_meta: List[Tuple[int, int]] = []
            for w_idx, (start_sec, end_sec) in enumerate(windows):
                for r_idx in range(masking_repeats):
                    seed = _deterministic_seed(audio_hash, start_sec, end_sec, r_idx)
                    masked_waveform = _mask_window_with_noise(waveform, sr, start_sec, end_sec, noise_level, seed)
                    variant_path = tmp_dir_path / f"w{w_idx}_r{r_idx}.wav"
                    sf.write(str(variant_path), masked_waveform, sr, subtype="PCM_16")
                    variant_paths.append(variant_path)
                    variant_meta.append((w_idx, r_idx))

            # 所有未缓存的遮挡音频通过extract_opera_features_batch一次批量提取,
            # 已缓存的直接命中, 其余5个位点的特征完全不重新提取(见combined,
            # 只在下面替换target_idx这一行)。
            variant_features = extract_opera_features_batch([str(p) for p in variant_paths])

        raw_samples: Dict[int, List[float]] = {i: [] for i in range(len(windows))}
        for (w_idx, r_idx), raw_feature in zip(variant_meta, variant_features):
            aux_prob = score_segment_classifier(deployment, raw_feature)
            standardized = (raw_feature - deployment.feature_mean) / deployment.feature_std
            variant_row = np.concatenate([standardized, [aux_prob]]).astype(np.float32)

            perturbed = combined.copy()
            perturbed[target_idx] = variant_row
            masked_per_init = _run_per_init_probs(deployment, perturbed, mask)
            for init_idx, masked_prob in enumerate(masked_per_init):
                raw_samples[w_idx].append(original_per_init[init_idx] - masked_prob)

    window_stats = []
    for w_idx, (start_sec, end_sec) in enumerate(windows):
        samples = np.array(raw_samples[w_idx])
        mean_delta = float(samples.mean())
        std_delta = float(samples.std())
        sign_mean = np.sign(mean_delta)
        consistency = 1.0 if sign_mean == 0 else float(np.mean(np.sign(samples) == sign_mean))
        window_stats.append(
            {
                "start_seconds": round(start_sec, 3),
                "end_seconds": round(end_sec, 3),
                "signed_delta": mean_delta,
                "std": std_delta,
                "direction_consistency": consistency,
                "raw_signed_delta_samples": samples.tolist(),  # 9个原始值, 不只存均值
            }
        )

    evidence_pattern, suspected_segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, masking_repeats, window_seconds, cfg
    )

    strongest = max(window_stats, key=lambda w: abs(w["signed_delta"]))
    strongest_window = {
        "start_seconds": strongest["start_seconds"],
        "end_seconds": strongest["end_seconds"],
        "signed_probability_delta": strongest["signed_delta"],
    }

    evidence_curve = _build_continuous_curve(window_stats, duration, time_resolution)

    result = {
        "site": target_site,
        "target_site_selection": selection_reason,
        "target_class": TARGET_CLASS,
        "evidence_pattern": evidence_pattern,
        "suspected_abnormal_segment_detected": bool(
            evidence_pattern == EVIDENCE_PATTERN_FOCAL and len(suspected_segments) > 0
        ),
        "suspected_abnormal_segments": suspected_segments,
        "strongest_window": strongest_window,
        "method": {
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "masking_repeats": masking_repeats,
            "masking_method": MASKING_METHOD,
            "target_class": TARGET_CLASS,
            "time_resolution_seconds": time_resolution,
        },
        "stability": stability,
        "evidence_curve": evidence_curve,
        "raw_window_samples": {
            str(w_idx): {
                "start_seconds": window_stats[w_idx]["start_seconds"],
                "end_seconds": window_stats[w_idx]["end_seconds"],
                "raw_signed_delta_samples": window_stats[w_idx]["raw_signed_delta_samples"],
            }
            for w_idx in range(len(windows))
        },
        "patient_prediction": prediction,
        "term_disclaimer": SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
        "limitations": limitations,
    }
    if evidence_pattern in _PATTERN_NOTES:
        result["note"] = _PATTERN_NOTES[evidence_pattern]
    return result
