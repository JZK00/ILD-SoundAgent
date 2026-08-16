# -*- coding: utf-8 -*-
"""阶段4静态可视化(resp_agent/visualization.py)测试。运行方式同
tests/test_localization.py:
    D:\\conda_envs\\resp\\python.exe tests\\test_visualization.py

真实音频/真实模型的测试(patient 001 Site 4)会真的调GPU跑OPERA-CT+patient
模型, 函数名标`_real_`前缀。生成的PNG存到
D:\\医学影像预测\\resp_agent_app\\visualization_dev_outputs\\, 这是开发期
输出目录, 不是阶段0锁定的部署产物, 可以覆盖重跑。

只验证paper主题的画图正确性用真实数据(用户要求"至少完整验证paper主题");
dark主题只做一个轻量合成数据冒烟测试(能正常出图、不报错), 不做深入校验。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resp_agent.localization import (  # noqa: E402
    DEFAULT_CONFIG,
    EVIDENCE_PATTERN_DIFFUSE,
    EVIDENCE_PATTERN_FOCAL,
    EVIDENCE_PATTERN_MINIMAL,
    EVIDENCE_PATTERN_UNSTABLE,
    SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
    TARGET_CLASS,
    _PATTERN_NOTES,
    _build_continuous_curve,
    _classify_evidence_pattern,
)
from resp_agent.visualization import (  # noqa: E402
    THEMES,
    VisualizationFontError,
    VisualizationInputError,
    _mel_spectrogram_db,
    _resolve_cjk_font,
    assert_disclaimer_present,
    assert_no_forbidden_wording,
    load_localization_waveform,
    plot_site_evidence_bars,
    plot_site_localization,
    save_figure,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "visualization_dev_outputs"


# ============ 复用test_localization.py的合成window_stats构造方式 ============
def _make_windows(n_windows, hop_seconds=0.5, window_seconds=2.0):
    return [(round(i * hop_seconds, 3), round(i * hop_seconds + window_seconds, 3)) for i in range(n_windows)]


def _stats_from_deltas(deltas_per_window, raw_samples):
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


def _make_synthetic_wav(path: Path, duration_sec: float = 8.0, sr: int = 16000):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    waveform = (0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.sin(2 * np.pi * 850 * t)).astype(np.float32)
    sf.write(str(path), waveform, sr, subtype="PCM_16")
    return waveform, sr


def _build_fake_result(site_label, deltas, raw_samples, duration=8.0):
    """构造和`localize_suspected_abnormal_segments`返回值结构一致(只含
    画图需要的字段)的合成结果, 分类逻辑直接调用真实的`_classify_evidence_
    pattern`/`_build_continuous_curve`, 不重新发明一套判断规则。"""
    n = len(deltas)
    windows = _make_windows(n)
    window_stats = _stats_from_deltas(deltas, raw_samples)
    cfg = dict(DEFAULT_CONFIG)
    pattern, segments, stability = _classify_evidence_pattern(
        window_stats, raw_samples, windows, cfg["masking_repeats"], cfg["window_seconds"], cfg
    )
    curve = _build_continuous_curve(window_stats, duration, cfg["time_resolution_seconds"])
    result = {
        "site": site_label,
        "evidence_pattern": pattern,
        "suspected_abnormal_segments": segments,
        "evidence_curve": curve,
        "term_disclaimer": SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER,
    }
    if pattern in _PATTERN_NOTES:
        result["note"] = _PATTERN_NOTES[pattern]
    return result


# ============ 1) 主题字典结构完整性 ============
def test_themes_have_required_keys():
    required = {
        "figure_facecolor", "axes_facecolor", "text_color", "muted_text_color", "grid_color",
        "spine_color", "supports_color", "opposes_color", "minimal_color", "na_color",
        "borderline_color", "std_band_alpha", "segment_fill_alpha", "cmap",
    }
    for name, theme in THEMES.items():
        missing = required - set(theme.keys())
        assert not missing, f"theme={name}缺少字段: {missing}"
    assert set(THEMES.keys()) == {"paper", "dark"}
    print("test_themes_have_required_keys: OK")


# ============ 1b) 渲染环境必须真的有能画中文的字体, 不只是检查字符串内容 ============
# FORBIDDEN_TERMS检查(见assert_no_forbidden_wording)只验证文字*内容*不含
# 禁用措辞, 不代表渲染环境真的有字体能把免责声明这些中文字符画出来而不是
# Regression test for incomplete segment overlays.
# 再次只在"跑测试全过"但"图片实际不可读"之间溜走。
def test_cjk_font_available():
    try:
        font_name = _resolve_cjk_font()
    except VisualizationFontError as e:
        raise AssertionError(
            f"渲染环境没有找到任何支持中文的字体, 免责声明会显示成方框: {e}"
        ) from e
    print("test_cjk_font_available: OK, 使用字体=", font_name)


# ============ 2) 合成数据: clear + supports方向, 应该画出1个实线声段框 ============
def test_synthetic_clear_supports_segment_drawn():
    n = 13
    deltas = [0.005] * n
    deltas[6] = 0.15
    raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
    result = _build_fake_result("Site 1", deltas, raw_samples)
    assert result["evidence_pattern"] == EVIDENCE_PATTERN_FOCAL
    assert len(result["suspected_abnormal_segments"]) == 1
    seg = result["suspected_abnormal_segments"][0]
    assert seg["direction"] == "supports_fibrotic_prediction"
    assert seg["is_borderline"] is False, "这个合成用例故意留了很大余量, 不应该是borderline"

    waveform, sr = _make_synthetic_wav(Path(tempfile.mkstemp(suffix=".wav")[1]))
    fig, meta = plot_site_localization(result, waveform, sr, theme="paper")
    assert meta["n_segments_drawn"] == 1
    assert_no_forbidden_wording(fig)
    assert_disclaimer_present(fig)
    save_figure(fig, OUTPUT_DIR / "synthetic_clear_supports.png")
    print("test_synthetic_clear_supports_segment_drawn: OK, meta=", meta)


# ============ 3) diffuse/minimal/unstable: 不应该画出任何声段框 ============
def test_synthetic_diffuse_minimal_unstable_no_forced_box():
    n = 13
    cases = {}

    rng = np.random.default_rng(0)
    deltas_diffuse = [-0.08 + 0.02 * np.sin(i / 3.0) for i in range(n)]
    raw_diffuse = {i: [deltas_diffuse[i] + rng.normal(0, 0.003) for _ in range(9)] for i in range(n)}
    cases[EVIDENCE_PATTERN_DIFFUSE] = _build_fake_result("Site 2", deltas_diffuse, raw_diffuse)

    deltas_minimal = [0.005] * n
    raw_minimal = {i: [deltas_minimal[i]] * 9 for i in range(n)}
    cases[EVIDENCE_PATTERN_MINIMAL] = _build_fake_result("Site 3", deltas_minimal, raw_minimal)

    deltas_unstable = [0.1] * n
    samples_unstable = [0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
    raw_unstable = {i: samples_unstable for i in range(n)}
    cases[EVIDENCE_PATTERN_UNSTABLE] = _build_fake_result("Site 5", deltas_unstable, raw_unstable)

    for expected_pattern, result in cases.items():
        assert result["evidence_pattern"] == expected_pattern, (expected_pattern, result["evidence_pattern"])
        assert result["suspected_abnormal_segments"] == []
        waveform, sr = _make_synthetic_wav(Path(tempfile.mkstemp(suffix=".wav")[1]))
        fig, meta = plot_site_localization(result, waveform, sr, theme="paper")
        assert meta["n_segments_drawn"] == 0, (expected_pattern, meta)
        assert_no_forbidden_wording(fig)
        assert_disclaimer_present(fig)
        save_figure(fig, OUTPUT_DIR / f"synthetic_{expected_pattern}.png")
    print("test_synthetic_diffuse_minimal_unstable_no_forced_box: OK,", list(cases.keys()))


# ============ 3b) 低采样率(如patient 001这批真实录音的原生4000Hz)不应该
# 触发空mel滤波器——fmax必须钳制到min(MEL_FMAX, sr/2), 不是简单调低n_mels
# ============
def test_low_sample_rate_fmax_clamped():
    rng = np.random.default_rng(3)
    sr = 4000
    waveform = (0.05 * rng.standard_normal(sr * 8)).astype(np.float32)
    mel_db, effective_fmax = _mel_spectrogram_db(waveform, sr)
    assert effective_fmax == sr / 2.0, effective_fmax
    assert np.isfinite(mel_db).all()
    print("test_low_sample_rate_fmax_clamped: OK, effective_fmax=", effective_fmax)


def test_synthetic_low_sample_rate_full_figure():
    """4000Hz合成波形跑完整的plot_site_localization, 不只测内部mel函数,
    确认标题/频谱轴在低采样率下也能正常出图不报错(patient 001 Site 4这批
    真实录音就是4000Hz原生采样率, 之前版本硬编码fmax=8000导致28个空mel
    滤波器直接报错, 见对应修复)。"""
    n = 13
    deltas = [0.005] * n
    deltas[6] = 0.15
    raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
    result = _build_fake_result("Site 4", deltas, raw_samples, duration=8.0)

    sr = 4000
    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    t = np.linspace(0, 8.0, int(8.0 * sr), endpoint=False)
    waveform = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(wav_path), waveform, sr, subtype="PCM_16")

    fig, meta = plot_site_localization(result, waveform, sr, theme="paper")
    assert_no_forbidden_wording(fig)
    assert_disclaimer_present(fig)
    save_figure(fig, OUTPUT_DIR / "synthetic_low_sample_rate.png")
    print("test_synthetic_low_sample_rate_full_figure: OK, meta=", meta)


# ============ 4) dark主题冒烟测试(合成数据, 不深入校验) ============
def test_dark_theme_smoke():
    n = 13
    deltas = [0.005] * n
    deltas[3] = -0.14
    raw_samples = {i: [deltas[i] + j for j in [-0.002, 0.0, 0.002] * 3] for i in range(n)}
    result = _build_fake_result("Site 6", deltas, raw_samples)
    waveform, sr = _make_synthetic_wav(Path(tempfile.mkstemp(suffix=".wav")[1]))
    fig, meta = plot_site_localization(result, waveform, sr, theme="dark")
    save_figure(fig, OUTPUT_DIR / "synthetic_dark_theme_smoke.png")
    print("test_dark_theme_smoke: OK, meta=", meta)


# ============ 5) theme参数校验 ============
def test_unknown_theme_raises():
    result = _build_fake_result("Site 1", [0.005] * 13, {i: [0.005] * 9 for i in range(13)})
    waveform, sr = _make_synthetic_wav(Path(tempfile.mkstemp(suffix=".wav")[1]))
    try:
        plot_site_localization(result, waveform, sr, theme="neon")
        raise AssertionError("应该抛出VisualizationInputError")
    except VisualizationInputError as e:
        print("test_unknown_theme_raises: OK,", e)


# ============ 6) waveform时长和evidence_curve对不上要报错, 不能静默画错位的图 ============
def test_mismatched_waveform_duration_raises():
    result = _build_fake_result("Site 1", [0.005] * 13, {i: [0.005] * 9 for i in range(13)}, duration=8.0)
    short_waveform, sr = _make_synthetic_wav(Path(tempfile.mkstemp(suffix=".wav")[1]), duration_sec=2.0)
    try:
        plot_site_localization(result, short_waveform, sr, theme="paper")
        raise AssertionError("应该抛出VisualizationInputError")
    except VisualizationInputError as e:
        print("test_mismatched_waveform_duration_raises: OK,", e)


# ============ 7) 真实数据: patient 001 Site 4, paper主题完整验证 ============
def test_real_patient001_site4_paper_theme():
    from resp_agent.inference import predict_patient
    from resp_agent.localization import localize_suspected_abnormal_segments
    from resp_agent.patient_model import load_deployment_model

    dep = load_deployment_model("site_self_attention")
    site_audio = {f"Site {i}": rf"D:\resp_data\audio\Fibrotic sounds\00{i} ({i}).wav" for i in range(1, 7)}

    result = localize_suspected_abnormal_segments(dep, site_audio, site="Site 4")
    assert result["evidence_pattern"] == EVIDENCE_PATTERN_FOCAL, result["evidence_pattern"]
    assert len(result["suspected_abnormal_segments"]) == 1, result["suspected_abnormal_segments"]
    seg = result["suspected_abnormal_segments"][0]

    # Regression values from the frozen patient 001 configuration.
    assert abs(seg["start_seconds"] - 0.0) < 1e-6, seg["start_seconds"]
    assert abs(seg["end_seconds"] - 3.5) < 1e-6, seg["end_seconds"]
    assert seg["direction"] == "opposes_fibrotic_prediction", seg["direction"]
    assert seg["is_borderline"] is True, seg
    assert seg["boundary_type"] == "left", seg["boundary_type"]
    assert seg["boundary_limited"] is True, seg

    waveform, sr = load_localization_waveform(site_audio["Site 4"])
    fig, meta = plot_site_localization(result, waveform, sr, theme="paper")
    assert meta["n_segments_drawn"] == 1, meta
    assert_no_forbidden_wording(fig)
    assert_disclaimer_present(fig)
    site4_png = save_figure(fig, OUTPUT_DIR / "patient001_site4_paper.png")

    prediction = predict_patient(dep, site_audio)
    bars_fig = plot_site_evidence_bars(prediction, theme="paper")
    assert_no_forbidden_wording(bars_fig)
    assert_disclaimer_present(bars_fig)
    bars_png = save_figure(bars_fig, OUTPUT_DIR / "patient001_site_bars_paper.png")

    assert site4_png.exists() and site4_png.stat().st_size > 0
    assert bars_png.exists() and bars_png.stat().st_size > 0
    print(
        "test_real_patient001_site4_paper_theme: OK, segment=", seg["start_seconds"], "-", seg["end_seconds"],
        "direction=", seg["direction"], "is_borderline=", seg["is_borderline"],
        "boundary_type=", seg["boundary_type"], "->", site4_png, bars_png,
    )


ALL_TESTS = [
    test_themes_have_required_keys,
    test_cjk_font_available,
    test_synthetic_clear_supports_segment_drawn,
    test_synthetic_diffuse_minimal_unstable_no_forced_box,
    test_low_sample_rate_fmax_clamped,
    test_synthetic_low_sample_rate_full_figure,
    test_dark_theme_smoke,
    test_unknown_theme_raises,
    test_mismatched_waveform_duration_raises,
    test_real_patient001_site4_paper_theme,
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
    print(f"PNG outputs: {OUTPUT_DIR}")
    if failures:
        sys.exit(1)
