# -*- coding: utf-8 -*-
"""患者级推理接口。管线顺序(严格按此顺序, 不得调换):

    每个位点原始OPERA特征(768维)
    -> segment scaler(segment分类器自己的标准化, 不是患者级的)
    -> segment classifier 得到辅助概率(第769维, 原始概率, 不再标准化)
    -> 原始OPERA特征用患者级 feature_mean/feature_std 标准化(前768维)
    -> 拼接成769维: [标准化后的768维OPERA特征, 原始769维segment辅助概率]
    -> 按site_order固定顺序排列 + 缺失位点mask
    -> 3-init ensemble(softmax后对3个初始化取平均)

这个顺序和`train_patient_level_compare_v2.py::build_fold_features`训练时的
拼接方式完全一致: `combined = np.concatenate([reduced, aux[:, :, None]], -1)`,
其中`reduced`是标准化后的768维、`aux`是segment分类器的原始概率——第769维
从未被患者级mean/std标准化过, 这里也不能标准化。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .opera_encoder import extract_opera_features
from .patient_model import DeploymentModel, score_segment_classifier
from .schemas import PatientPrediction, SiteContribution, classify_direction

PREDICTION_FIBROTIC = "fibrotic"
PREDICTION_NON_FIBROTIC = "non_fibrotic"


class PredictionInputError(ValueError):
    """site_audio里给了模型不认识的位点名、或一个位点都没给, 抛这个。"""


def build_site_features(
    deployment: DeploymentModel, site_audio: Dict[str, str]
) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """返回 (combined_feats [n_sites,769], mask [n_sites], available_sites, missing_sites)。
    只对site_audio里实际给出的位点调用OPERA编码器和segment分类器; 缺失位点
    的行是全零占位, 靠mask让模型忽略, 不会被误当成"检测到阴性"的证据。"""
    site_order = deployment.site_order
    unknown = set(site_audio) - set(site_order)
    if unknown:
        raise PredictionInputError(
            f"site_audio里有模型不认识的位点名: {sorted(unknown)}; "
            f"合法位点名: {site_order}"
        )
    if not site_audio:
        raise PredictionInputError("site_audio不能为空, 至少需要1个位点的音频")

    feat_dim = deployment.inference_config["patient_feature_dim"]
    n_sites = len(site_order)
    combined = np.zeros((n_sites, feat_dim), dtype=np.float32)
    mask = np.zeros(n_sites, dtype=np.float32)
    available_sites, missing_sites = [], []

    for idx, site in enumerate(site_order):
        if site not in site_audio:
            missing_sites.append(site)
            continue
        raw_feature = extract_opera_features(site_audio[site])
        aux_prob = score_segment_classifier(deployment, raw_feature)
        standardized = (raw_feature - deployment.feature_mean) / deployment.feature_std
        combined[idx] = np.concatenate([standardized, [aux_prob]]).astype(np.float32)
        mask[idx] = 1.0
        available_sites.append(site)

    return combined, mask, available_sites, missing_sites


def run_ensemble(deployment: DeploymentModel, combined: np.ndarray, mask: np.ndarray) -> float:
    """combined: [n_sites,769], mask: [n_sites] -> 3-init ensemble平均后的纤维化概率。"""
    if not np.any(mask):
        raise PredictionInputError("模型推理至少需要一个有效位点(mask全为0, 没有任何可用输入)")
    feat_t = torch.from_numpy(combined).unsqueeze(0).to(deployment.device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(deployment.device)
    probs = []
    with torch.no_grad():
        for model in deployment.models:
            logits = model(feat_t, mask_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return float(np.mean(probs))


def predict_patient(
    deployment: DeploymentModel,
    site_audio: Dict[str, str],
    patient_info: Optional[dict] = None,
    threshold_type: str = "default",
) -> dict:
    """site_audio: {"Site 1": "path/to/site1.wav", ...}, 支持1-6个位点。
    threshold_type: "default"(0.5, 始终可用) 或 "exploratory_sensitivity"
    (来自55人重复划分OOF的探索性阈值, 未经外部验证, 见thresholds.json里的
    warning字段)。

    patient_info(年龄/性别/症状等)目前只是原样透传进返回值供报告模块使用,
    不作为模型输入——这个部署模型没有人口学分支, 喂进去也不会被用到,
    没必要假装它影响了预测。
    """
    if threshold_type not in ("default", "exploratory_sensitivity"):
        raise ValueError(f"threshold_type={threshold_type!r} 必须是 'default' 或 'exploratory_sensitivity'")

    combined, mask, available_sites, missing_sites = build_site_features(deployment, site_audio)

    fibrosis_probability = run_ensemble(deployment, combined, mask)

    warnings: List[str] = []
    if threshold_type == "exploratory_sensitivity":
        threshold = deployment.thresholds.get("exploratory_sensitivity_threshold")
        if threshold is None:
            warnings.append(
                f"{deployment.model_name}没有可用的exploratory_sensitivity_threshold"
                "(该模型的5-split平均OOF未能达到目标灵敏度), 已改用default_threshold=0.5"
            )
            threshold = deployment.thresholds["default_threshold"]
            threshold_type = "default"
        else:
            warnings.append(deployment.thresholds.get("warning", ""))
    else:
        threshold = deployment.thresholds["default_threshold"]

    prediction = PREDICTION_FIBROTIC if fibrosis_probability >= threshold else PREDICTION_NON_FIBROTIC
    confidence = fibrosis_probability if prediction == PREDICTION_FIBROTIC else 1.0 - fibrosis_probability

    if len(missing_sites) > 0:
        warnings.append(
            f"缺失位点 {missing_sites}, 当前结果只基于{len(available_sites)}/"
            f"{len(deployment.site_order)}个位点, 存在信息缺失"
        )

    # ---- 位点贡献度: 逐个已采集位点遮蔽(mask置0, 特征置零), 重新跑3-init
    # ensemble, 用有符号差值表达方向, 不重新调用OPERA编码器。 ----
    # 只有1个可用位点时, 遮掉它会让mask全为0("没有任何输入的患者概率"没有
    # 合理解释, 而且Transformer在全mask的输入上可能产生NaN), 这种情况下
    # 仍然正常输出患者预测(用那1个位点), 但不计算该位点的遮挡贡献。
    n_available = int(mask.sum())
    site_order = deployment.site_order
    site_contributions: List[dict] = []
    for idx, site in enumerate(site_order):
        if site not in available_sites:
            site_contributions.append(
                SiteContribution(site=site, available=False).as_dict()
            )
            continue
        if n_available <= 1:
            site_contributions.append(
                SiteContribution(
                    site=site,
                    available=True,
                    original_probability=fibrosis_probability,
                ).as_dict()
            )
            warnings.append(
                f"仅提供1个位点({site}), 遮挡该位点后模型将无有效输入, 因此不计算位点贡献。"
            )
            continue
        occluded = combined.copy()
        occluded_mask = mask.copy()
        occluded[idx] = 0.0
        occluded_mask[idx] = 0.0
        masked_probability = run_ensemble(deployment, occluded, occluded_mask)
        signed_delta = fibrosis_probability - masked_probability
        site_contributions.append(
            SiteContribution(
                site=site,
                available=True,
                original_probability=fibrosis_probability,
                masked_probability=masked_probability,
                signed_probability_delta=signed_delta,
                absolute_importance=abs(signed_delta),
                direction=classify_direction(signed_delta),
            ).as_dict()
        )

    result = PatientPrediction(
        prediction=prediction,
        fibrosis_probability=fibrosis_probability,
        threshold=threshold,
        threshold_type=threshold_type,
        confidence=confidence,
        available_sites=available_sites,
        missing_sites=missing_sites,
        site_contributions=site_contributions,
        model_name=deployment.model_name,
        warnings=[w for w in warnings if w],
    ).as_dict()
    if patient_info:
        result["patient_info"] = patient_info
    return result
