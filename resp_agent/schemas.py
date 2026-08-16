# -*- coding: utf-8 -*-
"""共享输出结构定义。字段设计直接对应 VELCRO 呼吸音MVP实施计划(v3)里已经和
用户确认过的修订点, 不是随意起的名字:

- 位点/时间段贡献度必须保留符号方向, 不能只输出绝对值(计划修订点4)。
  `signed_probability_delta = original_probability - occluded_probability`:
  >0 表示遮掉该位点/时间段后概率下降, 即它是支持纤维化判断的证据;
  <0 表示遮掉后概率反而上升, 即它是反对纤维化判断的证据。
- 不产出`abnormality_score`(没有验证过的窗口级分类器), 时间段贡献度只用
  `importance_score`/`interpretation`, 文案统一"高贡献音频片段"而不是"异常"
  (计划修订点3)。
- 阈值分`default`(0.5)和`exploratory_sensitivity`两档, 后者不能叫
  "screening threshold"(未经外部验证), 且必须标注具体用的是哪一档
  (计划修订点5、v3修订)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

DIRECTION_SUPPORTS = "supports_fibrotic_prediction"
DIRECTION_OPPOSES = "opposes_fibrotic_prediction"
DIRECTION_MINIMAL = "minimal_influence"

THRESHOLD_TYPE_DEFAULT = "default"
THRESHOLD_TYPE_EXPLORATORY = "exploratory_sensitivity"


def classify_direction(signed_delta: float, minimal_threshold: float = 0.02) -> str:
    """signed_delta = original_probability - occluded_probability。
    绝对值小于minimal_threshold时不下方向性结论, 只报告"影响较小"。"""
    if abs(signed_delta) < minimal_threshold:
        return DIRECTION_MINIMAL
    return DIRECTION_SUPPORTS if signed_delta > 0 else DIRECTION_OPPOSES


@dataclass
class SiteContribution:
    site: str
    available: bool
    original_probability: Optional[float] = None
    masked_probability: Optional[float] = None
    signed_probability_delta: Optional[float] = None
    absolute_importance: Optional[float] = None
    direction: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentContribution:
    """时间段贡献度。不含abnormality_score, 见模块docstring。"""

    start_seconds: float
    end_seconds: float
    signed_probability_delta: float
    importance_score: float
    interpretation: str
    target_class: str = "fibrotic"
    masking_method: str = "deterministic_low_energy_noise"
    masking_repeats: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


SUSPECTED_ABNORMAL_SEGMENT_TERM_ZH = "可疑异常声段"
SUSPECTED_ABNORMAL_SEGMENT_TERM_EN = "Suspected abnormal respiratory sound segment"

SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER = (
    "可疑异常声段(Suspected abnormal respiratory sound segment)是指遮挡后会"
    "显著改变患者级预测概率的呼吸音时间段, 属于模型定位结果, 尚不等同于经"
    "临床医生确认的病理异常声段。"
)

# Forbidden diagnostic claims. Output validation is covered by the test suite.
FORBIDDEN_TERMS = (
    "confirmed abnormal segment",
    "病理异常声段",
    "确诊异常声段",
)


@dataclass
class SuspectedAbnormalSegment:
    """"可疑异常声段": 遮挡后会显著改变患者级预测概率的呼吸音时间段, 属于
    模型定位结果, 尚不等同于经临床医生确认的病理异常声段。见
    SUSPECTED_ABNORMAL_SEGMENT_DISCLAIMER。

    `criteria`/`minimum_relative_criterion_margin`/`focality_strength`/
    `is_borderline`/`touches_recording_boundary`/`boundary_type`/
    `boundary_limited`/`caveats`是在4项focal判据(预设工程判据, 阈值未经
    临床时间段标注验证)全部通过*之后*附加的说明性字段(见localization.py::
    _classify_evidence_pattern), 不改变focal/diffuse/minimal/unstable的
    判定结果本身, 只说明"这个已经判成focal的声段离阈值有多近、是否贴着
    录音边界"。`minimum_relative_criterion_margin`(float)是4项判据里
    relative_margin的最小值, `focality_strength`(str: "borderline"/
    "clear")是它的文字等级, 两者定义一致、只是数值/文字两种呈现形式。
    这个dataclass目前只作为输出结构的文档参考, localization.py里实际
    构造的是字段完全对应的plain dict, 没有实例化这个类。"""

    start_seconds: float
    end_seconds: float
    peak_time_seconds: float
    peak_signed_probability_delta: float
    importance_score: float
    direction: str
    direction_consistency: float
    evidence_std: float
    criteria: dict = field(default_factory=dict)
    minimum_relative_criterion_margin: Optional[float] = None
    focality_strength: Optional[str] = None
    is_borderline: Optional[bool] = None
    touches_recording_boundary: Optional[bool] = None
    boundary_type: Optional[str] = None
    boundary_limited: Optional[bool] = None
    caveats: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatientPrediction:
    prediction: str  # "fibrotic" / "non_fibrotic"
    fibrosis_probability: float
    threshold: float
    threshold_type: str  # THRESHOLD_TYPE_DEFAULT / THRESHOLD_TYPE_EXPLORATORY
    confidence: float
    available_sites: list
    missing_sites: list
    site_contributions: list  # list[dict], 每个来自 SiteContribution.as_dict()
    model_name: str
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)
