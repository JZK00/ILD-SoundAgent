# -*- coding: utf-8 -*-
"""阶段3测试。没有pytest的环境下也能跑:
    D:\\conda_envs\\resp\\python.exe tests\\test_localization.py
(pytest如果装了也能直接`pytest tests/test_localization.py`, 函数是标准
test_*命名, 用assert, 不依赖pytest专有API。)

真实音频/真实模型的测试(patient 001 Site 4 diffuse回归、短合成音频确定性
测试)会真的调GPU跑OPERA-CT+patient模型, 比纯逻辑单测慢, 已经在函数名里
标注`_real_`前缀方便区分。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resp_agent.localization import (  # noqa: E402
    CONFIG_PATH_DEFAULT,
    DEFAULT_CONFIG,
    EVIDENCE_PATTERN_DIFFUSE,
    EVIDENCE_PATTERN_FOCAL,
    EVIDENCE_PATTERN_MINIMAL,
    EVIDENCE_PATTERN_UNSTABLE,
    LocalizationConfigError,
    LocalizationInputError,
    _build_continuous_curve,
    _find_window_peaks,
    _classify_evidence_pattern,
    _run_per_init_probs,
    load_localization_config,
)
from resp_agent.schemas import FORBIDDEN_TERMS  # noqa: E402


# ============ 构造合成window_stats/raw_samples的小工具 ============
def _make_window_stats(n_windows, hop_seconds=0.5, window_seconds=2.0):
    windows = [(round(i * hop_seconds, 3), round(i * hop_seconds + window_seconds, 3)) for i in range(n_windows)]
    return windows


def _stats_from_deltas(deltas_per_window, raw_samples):
    """deltas_per_window: list[float] (窗口均值); raw_samples: dict[int, list[float]]
    (每个窗口9个原始样本, 已经和deltas_per_window的均值对应)。"""
    window_stats = []
    for i, mean_delta in enumerate(deltas_per_window):
        samples = np.array(raw_samples[i])
        std = float(samples.std())
        sign_mean = np.sign(mean_delta)
        consistency = 1.0 if sign_mean == 0 else float(np.mean(np.sign(samples) == sign_mean))
        window_stats.append(
            {
                "start_seconds": i * 0.5,
                "end_seconds": i * 0.5 + 2.0,
                "signed_delta": mean_delta,
                "std": std,
                "direction_consistency": consistency,
            }
        )
    return window_stats


# ============ 1) focal: 人工构造单一局部峰值 ============
def test_classify_focal_single_peak():
    n = 13
    windows = _make_window_stats(n)
    deltas = [0.005] * n
    deltas[6] = 0.15  # 中间一个窗口明显突出, 其余都很小
    raw_samples = {}
    for i, d in enumerate(deltas):
        # 9个样本(3 repeat x 3 init)全部同方向, 紧贴均值, 保证consistency高
        raw_samples[i] = [d + jitter for jitter in [-0.002, 0.0, 0.002] * 3]
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_FOCAL, pattern
    assert len(segments) >= 1
    assert segments[0]["peak_time_seconds"] == windows[6][0] + 1.0
    assert segments[0]["direction"] == "supports_fibrotic_prediction"
    print("test_classify_focal_single_peak: OK, segments=", segments)


# ============ 2) diffuse: 均匀同方向贡献 ============
def test_classify_diffuse_uniform():
    n = 13
    windows = _make_window_stats(n)
    rng = np.random.default_rng(0)
    deltas = [-0.08 + 0.02 * np.sin(i / 3.0) for i in range(n)]  # 缓慢起伏, 没有尖锐峰
    raw_samples = {i: [deltas[i] + rng.normal(0, 0.003) for _ in range(9)] for i in range(n)}
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_DIFFUSE, pattern
    assert segments == []
    print("test_classify_diffuse_uniform: OK")


# ============ 3) minimal: 全部影响很小 ============
def test_classify_minimal():
    n = 13
    windows = _make_window_stats(n)
    deltas = [0.005] * n
    raw_samples = {i: [deltas[i]] * 9 for i in range(n)}
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_MINIMAL, pattern
    assert segments == []
    print("test_classify_minimal: OK")


# ============ 4) unstable: 重复/初始化间方向不一致 ============
def test_classify_unstable_low_consistency():
    n = 13
    windows = _make_window_stats(n)
    deltas = [0.1] * n
    rng = np.random.default_rng(1)
    raw_samples = {}
    for i in range(n):
        # 9个样本里让接近一半反号, 方向一致率明显低于80%
        samples = [0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        raw_samples[i] = samples
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_UNSTABLE, pattern
    assert segments == []
    assert stability["overall_direction_consistency"] < cfg["minimum_direction_consistency"]
    print("test_classify_unstable_low_consistency: OK, stability=", stability)


# ============ 5) 相反方向峰值不合并 ============
def test_opposite_direction_peaks_not_merged():
    n = 13
    windows = _make_window_stats(n)
    deltas = [0.005] * n
    deltas[3] = 0.15   # 支持方向的峰(全局最强, 保证3个init都一致选中它, 不触发峰值位置不稳定判定)
    deltas[9] = -0.06  # 反对方向的次强峰, 时间上分开, 不应该被合并/也不应该被丢弃
    raw_samples = {}
    for i, d in enumerate(deltas):
        raw_samples[i] = [d + jitter for jitter in [-0.002, 0.0, 0.002] * 3]
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_FOCAL, pattern
    assert len(segments) == 2, segments
    directions = {s["direction"] for s in segments}
    assert directions == {"supports_fibrotic_prediction", "opposes_fibrotic_prediction"}
    # 两个区间不应该互相重叠/被合并成横跨两个峰的一个大区间
    a, b = sorted(segments, key=lambda s: s["start_seconds"])
    assert a["end_seconds"] <= b["start_seconds"] + 1e-6, segments
    print("test_opposite_direction_peaks_not_merged: OK, segments=", segments)


# ============ 6) 重叠窗口贡献取平均, 不求和 ============
def test_overlapping_windows_no_sum():
    window_stats = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "signed_delta": 0.1, "std": 0.0, "direction_consistency": 1.0},
        {"start_seconds": 0.5, "end_seconds": 2.5, "signed_delta": 0.1, "std": 0.0, "direction_consistency": 1.0},
    ]
    curve = _build_continuous_curve(window_stats, duration=2.5, time_resolution=0.1)
    signed = np.array(curve["signed_evidence"])
    time_s = np.array(curve["time_seconds"])
    overlap_region = signed[(time_s >= 0.5) & (time_s < 2.0)]
    # 重叠区间被两个delta=0.1的窗口同时覆盖, 取平均应该还是0.1, 不是0.2(求和)
    assert np.allclose(overlap_region, 0.1, atol=1e-9), overlap_region
    non_overlap = signed[(time_s >= 0.0) & (time_s < 0.5)]
    assert np.allclose(non_overlap, 0.1, atol=1e-9), non_overlap
    print("test_overlapping_windows_no_sum: OK")


# ============ 7) 不产生abnormality_score字段 ============
def test_no_abnormality_score_in_synthetic_result():
    n = 3
    windows = _make_window_stats(n)
    deltas = [0.1, 0.02, -0.01]
    raw_samples = {i: [deltas[i]] * 9 for i in range(n)}
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    blob = json.dumps({"pattern": pattern, "segments": segments, "stability": stability}, ensure_ascii=False)
    assert "abnormality_score" not in blob
    print("test_no_abnormality_score_in_synthetic_result: OK")


# ============ 8) 不出现禁用措辞 ============
# 注意: 强制免责声明(SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER)本身按用户给定的
# 原文必须包含"...尚不等同于经临床医生确认的病理异常声段"这句话, 这是在
# *否定*意义上引用"病理异常声段"这个词、澄清"可疑"不等于"确诊", 不在
# FORBIDDEN_TERMS的检查范围内——真正禁止的是在免责声明之外, 独立地把某个
# 声段直接断言成"confirmed abnormal segment"/"病理异常声段"/"确诊异常声段"。
def test_no_forbidden_wording():
    from resp_agent.localization import _PATTERN_NOTES, TARGET_CLASS

    blob = " ".join(_PATTERN_NOTES.values()) + TARGET_CLASS
    for term in FORBIDDEN_TERMS:
        assert term not in blob, f"发现禁用措辞: {term}"
    print("test_no_forbidden_wording: OK")


# ============ 9) 全空mask报错, 不产生全空mask下的"预测" ============
def test_full_zero_mask_raises():
    class _FakeModel:
        def __call__(self, feat, mask):
            raise AssertionError("不应该调用到模型前向, mask全零必须在此之前就报错")

    class _FakeDeployment:
        models = [_FakeModel()]
        device = "cpu"

    combined = np.zeros((6, 769), dtype=np.float32)
    mask = np.zeros(6, dtype=np.float32)
    try:
        _run_per_init_probs(_FakeDeployment(), combined, mask)
        raise AssertionError("应该抛出LocalizationInputError")
    except LocalizationInputError as e:
        print("test_full_zero_mask_raises: OK,", e)


# ============ 10) 配置从文件读取, 不硬编码 ============
def test_config_loaded_from_yaml():
    cfg = load_localization_config()
    for key in DEFAULT_CONFIG:
        assert key in cfg, f"缺少配置项 {key}"
    assert cfg["window_seconds"] == 2.0
    assert cfg["masking_repeats"] == 3
    print("test_config_loaded_from_yaml: OK,", cfg)


# ============ 11) 真实音频: patient 001 Site 4 不再被合并成完整8秒声段(回归) ============
def test_real_patient001_site4_not_whole_clip():
    from resp_agent.patient_model import load_deployment_model
    from resp_agent.localization import localize_suspected_abnormal_segments

    dep = load_deployment_model("site_self_attention")
    site_audio = {f"Site {i}": rf"D:\resp_data\audio\Fibrotic sounds\00{i} ({i}).wav" for i in range(1, 7)}
    result = localize_suspected_abnormal_segments(dep, site_audio, site="Site 4")

    assert result["site"] == "Site 4"
    assert result["target_site_selection"] == "user_specified"
    # 核心回归点: 不能再把整段8秒合并成一个"可疑异常声段"
    for seg in result["suspected_abnormal_segments"]:
        duration = seg["end_seconds"] - seg["start_seconds"]
        assert duration < 8.0, f"回归失败: 又把整段8秒合并成一个声段了 {seg}"
    result_without_disclaimer = {k: v for k, v in result.items() if k != "term_disclaimer"}
    blob = json.dumps(result_without_disclaimer, ensure_ascii=False)
    assert "abnormality_score" not in blob
    for term in FORBIDDEN_TERMS:
        assert term not in blob, f"发现禁用措辞: {term}"

    # 新增字段的结构性检查(存在/类型正确), 不对is_borderline/boundary_type
    # 的具体值做硬断言——这两个是从真实音频+真实模型跑出来的, 需要先看
    # 这次实际打印出来的数字, 再和"patient 001预期focal+borderline+
    # left-boundary-limited"这个期望对照, 不应该反过来为了凑这个期望去调
    # borderline_relative_margin/1.5阈值(见用户要求: 不得根据这个病例事后
    # 调整阈值)。
    for seg in result["suspected_abnormal_segments"]:
        for key in (
            "criteria", "minimum_relative_criterion_margin", "focality_strength", "is_borderline",
            "touches_recording_boundary", "boundary_type", "boundary_limited", "caveats",
        ):
            assert key in seg, f"缺少字段 {key}: {seg}"
        assert seg["boundary_type"] in ("none", "left", "right", "both")
        assert seg["focality_strength"] in ("borderline", "clear")
        assert seg["focality_strength"] == ("borderline" if seg["is_borderline"] else "clear"), seg
        print(
            "  segment:", seg["start_seconds"], "-", seg["end_seconds"],
            "direction=", seg["direction"],
            "is_borderline=", seg["is_borderline"],
            "focality_strength=", seg["focality_strength"],
            "minimum_relative_criterion_margin=", round(seg["minimum_relative_criterion_margin"], 4),
            "boundary_type=", seg["boundary_type"],
            "boundary_limited=", seg["boundary_limited"],
        )
        for caveat in seg["caveats"]:
            print("    caveat:", caveat)

    print(
        "test_real_patient001_site4_not_whole_clip: OK, evidence_pattern=",
        result["evidence_pattern"], "n_segments=", len(result["suspected_abnormal_segments"]),
    )
    return result


# ============ 12) 真实短合成音频: 固定输入重复运行结果完全一致 ============
def _make_short_synthetic_wav(path: Path, duration_sec: float = 3.0, sr: int = 16000):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    waveform = (0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.sin(2 * np.pi * 850 * t)).astype(np.float32)
    sf.write(str(path), waveform, sr, subtype="PCM_16")


def test_real_deterministic_repeat_short_audio():
    from resp_agent.patient_model import load_deployment_model
    from resp_agent.localization import localize_suspected_abnormal_segments

    dep = load_deployment_model("site_self_attention")
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "synthetic_site1.wav"
        _make_short_synthetic_wav(wav_path)
        site_audio = {"Site 1": str(wav_path)}

        r1 = localize_suspected_abnormal_segments(dep, site_audio)
        r2 = localize_suspected_abnormal_segments(dep, site_audio)

    assert r1["evidence_pattern"] == r2["evidence_pattern"]
    assert r1["strongest_window"] == r2["strongest_window"]
    assert r1["evidence_curve"]["signed_evidence"] == r2["evidence_curve"]["signed_evidence"]
    assert r1["raw_window_samples"] == r2["raw_window_samples"]
    print("test_real_deterministic_repeat_short_audio: OK")


# ============ 13) 缓存命中与未命中结果一致 ============
def test_real_cache_hit_matches_miss():
    from resp_agent.opera_encoder import CACHE_DIR, extract_opera_features

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "cache_test.wav"
        _make_short_synthetic_wav(wav_path, duration_sec=2.0)

        # 先确保没有缓存(miss路径)
        import hashlib
        # 直接调用两次, 第一次是miss, 第二次必然命中同一个缓存key(内容不变)
        feat_miss = extract_opera_features(str(wav_path))
        feat_hit = extract_opera_features(str(wav_path))
    assert np.allclose(feat_miss, feat_hit)
    print("test_real_cache_hit_matches_miss: OK")


# ============ 14) fix①: 连续曲线direction_consistency必须在[0,1]范围内 ============
def test_evidence_curve_consistency_bounded():
    window_stats = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "signed_delta": 0.1, "std": 0.01, "direction_consistency": 0.5},
        {"start_seconds": 0.5, "end_seconds": 2.5, "signed_delta": -0.05, "std": 0.02, "direction_consistency": 0.9},
        {"start_seconds": 1.0, "end_seconds": 3.0, "signed_delta": 0.02, "std": 0.0, "direction_consistency": 1.0},
    ]
    curve = _build_continuous_curve(window_stats, duration=3.0, time_resolution=0.1)
    consistency = np.array(curve["direction_consistency"])
    assert consistency.min() >= 0.0, consistency.min()
    assert consistency.max() <= 1.0, consistency.max()
    # 被单个window=0.5覆盖的时间点应该恰好是0.5, 不是旧bug里的1.5
    time_s = np.array(curve["time_seconds"])
    only_first_window = consistency[(time_s >= 0.0) & (time_s < 0.5)]
    assert np.allclose(only_first_window, 0.5), only_first_window
    print("test_evidence_curve_consistency_bounded: OK")


# ============ 15) fix②: minimal优先于unstable ============
def test_minimal_priority_over_unstable():
    n = 13
    windows = _make_window_stats(n)
    # 效应本身很小(<0.02下限), 但故意让样本方向也不一致——旧顺序会先判定
    # unstable, 新顺序必须先判minimal, 不下"不稳定"这种更强的结论。
    deltas = [0.005] * n
    raw_samples = {i: [0.006, 0.005, -0.004, 0.006, -0.004, 0.005, 0.006, -0.004, 0.005] for i in range(n)}
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_MINIMAL, pattern
    print("test_minimal_priority_over_unstable: OK")


# ============ 16) fix③: diffuse曲线不能被peak_location_spread误判成unstable ============
def test_diffuse_not_misclassified_as_unstable_from_peak_spread():
    n = 13
    windows = _make_window_stats(n)
    rng = np.random.default_rng(42)
    deltas = [-0.08 + 0.015 * np.sin(i / 2.3) for i in range(n)]  # 缓慢起伏, 没有真正的局灶峰
    raw_samples = {}
    for i in range(n):
        # 每个窗口内3个init的噪声幅度和窗口间差异同量级, 让"哪个init的
        # 全局argmax落在哪个窗口"天然容易不一致——这正是旧代码会误判成
        # unstable的场景。
        raw_samples[i] = [deltas[i] + rng.normal(0, 0.02) for _ in range(9)]
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_DIFFUSE, pattern
    # diffuse分支从不应该计算peak_location_spread(没有候选峰可评估稳定性)
    assert stability["peak_location_spread_seconds"] is None, stability
    print("test_diffuse_not_misclassified_as_unstable_from_peak_spread: OK, stability=", stability)


# ============ 17) fix④: 首尾边界峰不能被find_peaks的结构性盲区漏掉 ============
def test_boundary_peaks_detected():
    n = 13

    def make(peak_index):
        deltas = [0.005] * n
        deltas[peak_index] = 0.15
        raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
        return deltas, raw_samples

    windows = _make_window_stats(n)
    cfg = dict(DEFAULT_CONFIG)

    for peak_index, label in [(0, "first"), (n - 1, "last")]:
        deltas, raw_samples = make(peak_index)
        window_stats = _stats_from_deltas(deltas, raw_samples)
        pattern, segments, stability = _classify_evidence_pattern(
            window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
        )
        assert pattern == EVIDENCE_PATTERN_FOCAL, f"{label}-boundary peak missed, got {pattern}"
        assert len(segments) == 1 and segments[0]["peak_time_seconds"] == windows[peak_index][0] + 1.0, segments
    print("test_boundary_peaks_detected: OK")


def test_boundary_type_labels():
    """boundary_type字段: 首/末window峰应标"left"/"right", 内部峰应标"none",
    且touches_recording_boundary/boundary_limited必须和boundary_type一致
    (这三个字段目前定义上boundary_limited==touches_recording_boundary,
    见localization.py::_segment_boundary_info的调用处)。"""
    n = 13
    windows = _make_window_stats(n)
    cfg = dict(DEFAULT_CONFIG)

    for peak_index, expected_type in [(0, "left"), (n - 1, "right"), (6, "none")]:
        deltas = [0.005] * n
        deltas[peak_index] = 0.15
        raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
        window_stats = _stats_from_deltas(deltas, raw_samples)
        pattern, segments, stability = _classify_evidence_pattern(
            window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
        )
        assert pattern == EVIDENCE_PATTERN_FOCAL, pattern
        assert len(segments) == 1, segments
        seg = segments[0]
        assert seg["boundary_type"] == expected_type, (peak_index, seg["boundary_type"])
        expected_touches = expected_type != "none"
        assert seg["touches_recording_boundary"] == expected_touches, seg
        assert seg["boundary_limited"] == expected_touches, seg
    print("test_boundary_type_labels: OK")


def test_criteria_and_borderline_fields_present():
    """每个focal声段必须带4项预设工程判据(阈值未经临床时间段标注验证)的
    原始值/阈值/margin/relative_margin, 以及minimum_relative_criterion_
    margin(=4项relative_margin的最小值, 数值型)、focality_strength
    (="borderline"/"clear", 是minimum_relative_criterion_margin的文字
    等级)和is_borderline(=minimum_relative_criterion_margin <
    cfg["borderline_relative_margin"])。这里只检查字段存在、结构正确、和
    这几个字段之间的定义式内部自洽, 不检查具体病例的期望值(那部分是
    test_real_patient001_site4_not_whole_clip的责任)。"""
    n = 13
    windows = _make_window_stats(n)
    deltas = [0.005] * n
    deltas[6] = 0.15
    raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    assert pattern == EVIDENCE_PATTERN_FOCAL, pattern
    seg = segments[0]
    expected_keys = {"absolute_effect", "prominence", "peak_to_median_ratio", "direction_consistency"}
    assert set(seg["criteria"].keys()) == expected_keys, seg["criteria"].keys()
    margins = []
    for name, item in seg["criteria"].items():
        for sub in ("value", "threshold", "margin", "relative_margin"):
            assert sub in item, (name, sub)
        assert abs(item["margin"] - (item["value"] - item["threshold"])) < 1e-9, (name, item)
        margins.append(item["relative_margin"])
    assert abs(seg["minimum_relative_criterion_margin"] - min(margins)) < 1e-9, seg
    assert seg["is_borderline"] == (
        seg["minimum_relative_criterion_margin"] < cfg["borderline_relative_margin"]
    ), seg
    assert seg["focality_strength"] in ("borderline", "clear"), seg
    assert seg["focality_strength"] == ("borderline" if seg["is_borderline"] else "clear"), seg
    assert isinstance(seg["caveats"], list)
    # direction_consistency的relative_margin结构性上限是
    # (1.0 - minimum_direction_consistency) / minimum_direction_consistency
    # (consistency不可能超过1.0), 默认配置下是(1.0-0.8)/0.8=0.25——这解释了
    # 为什么很多本身很干净的focal声段的minimum_relative_criterion_margin
    # 也可能只比borderline_relative_margin(默认0.2)高一点点, 不代表峰值
    # 本身弱。
    consistency_ceiling = (1.0 - cfg["minimum_direction_consistency"]) / cfg["minimum_direction_consistency"]
    assert seg["criteria"]["direction_consistency"]["relative_margin"] <= consistency_ceiling + 1e-9, seg
    print(
        "test_criteria_and_borderline_fields_present: OK, minimum_relative_criterion_margin=",
        seg["minimum_relative_criterion_margin"], "focality_strength=", seg["focality_strength"],
        "is_borderline=", seg["is_borderline"], "direction_consistency_ceiling=", consistency_ceiling,
    )


def test_find_window_peaks_boundary_unit():
    values = np.array([0.15, 0.02, 0.03, 0.01, 0.005, 0.2])
    peaks, prominences, left_ips, right_ips = _find_window_peaks(values, height=0.02, rel_height=0.5)
    assert 0 in peaks, peaks  # 索引0(0.15, 首端且比唯一邻居更高)应该被找到
    assert 5 in peaks, peaks  # 索引5(0.2, 末端且比唯一邻居更高)应该被找到
    # 边界峰的prominence不能是scipy对"非标准边界索引"退化出来的0
    for p_i, idx in enumerate(peaks):
        if idx in (0, 5):
            assert prominences[p_i] > 0, (idx, prominences[p_i])
    print("test_find_window_peaks_boundary_unit: OK, peaks=", peaks, "prominences=", prominences)


# ============ 18) fix⑥: 正式调用配置缺失/不全必须报错, 不静默fallback ============
def test_config_missing_file_raises():
    try:
        load_localization_config(config_path=Path("this_config_does_not_exist.yaml"))
        raise AssertionError("应该抛出LocalizationConfigError")
    except LocalizationConfigError as e:
        print("test_config_missing_file_raises: OK,", e)


def test_config_missing_key_raises():
    with tempfile.TemporaryDirectory() as td:
        incomplete_path = Path(td) / "incomplete.yaml"
        # 故意只写一部分必需字段
        incomplete_path.write_text("window_seconds: 2.0\nhop_seconds: 0.5\n", encoding="utf-8")
        try:
            load_localization_config(config_path=incomplete_path)
            raise AssertionError("应该抛出LocalizationConfigError")
        except LocalizationConfigError as e:
            print("test_config_missing_key_raises: OK,", e)


def test_config_allow_fallback_opt_in_only():
    # allow_fallback=True是显式选择的开发/测试便利分支, 不是正式调用会走到的路径
    cfg = load_localization_config(config_path=Path("this_config_does_not_exist.yaml"), allow_fallback=True)
    assert cfg == DEFAULT_CONFIG
    print("test_config_allow_fallback_opt_in_only: OK")


ALL_TESTS = [
    test_classify_focal_single_peak,
    test_classify_diffuse_uniform,
    test_classify_minimal,
    test_classify_unstable_low_consistency,
    test_opposite_direction_peaks_not_merged,
    test_overlapping_windows_no_sum,
    test_no_abnormality_score_in_synthetic_result,
    test_no_forbidden_wording,
    test_full_zero_mask_raises,
    test_config_loaded_from_yaml,
    test_real_patient001_site4_not_whole_clip,
    test_real_deterministic_repeat_short_audio,
    test_real_cache_hit_matches_miss,
    test_evidence_curve_consistency_bounded,
    test_minimal_priority_over_unstable,
    test_diffuse_not_misclassified_as_unstable_from_peak_spread,
    test_boundary_peaks_detected,
    test_boundary_type_labels,
    test_criteria_and_borderline_fields_present,
    test_find_window_peaks_boundary_unit,
    test_config_missing_file_raises,
    test_config_missing_key_raises,
    test_config_allow_fallback_opt_in_only,
]

if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures.append((t.__name__, e))
            print(f"{t.__name__}: FAILED - {e}")
    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        sys.exit(1)
