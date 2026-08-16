# -*- coding: utf-8 -*-
"""Load the deployment ensemble and auxiliary segment classifier.

Model classes are imported from the OPERA training code to keep the inference
architecture consistent with training.
"""

from __future__ import annotations

import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

OPERA_ROOT = Path(
    os.environ.get("OPERA_ROOT", r"D:\OPERA")
).expanduser().resolve()
if str(OPERA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERA_ROOT))

import train_patient_level_compare as base  # noqa: E402  (reuse validated model classes)

DEPLOYMENT_ROOT = (
    Path(__file__).resolve().parent.parent
    / "checkpoints"
    / "deployment"
)

MODEL_BUILDERS = {
    "site_self_attention": base.CrossAttentionTransformer,
    "patient_query_cross_attention": base.PatientQueryCrossAttention,
}


class CheckpointNotFoundError(RuntimeError):
    """部署checkpoint缺失时抛这个, 消息里给出确切期望路径和生成脚本, 方便
    Gradio界面直接把message展示给用户。"""


@dataclass
class DeploymentModel:
    model_name: str
    models: List[nn.Module]  # 3个初始化的ensemble, 均已.eval()
    segment_classifier: nn.Module  # 已.eval()
    segment_scaler_mean: np.ndarray
    segment_scaler_scale: np.ndarray
    feature_mean: np.ndarray  # 768维, 患者级OPERA特征标准化参数(全量55人拟合)
    feature_std: np.ndarray
    site_order: List[str]
    thresholds: dict  # {"default_threshold":..., "exploratory_sensitivity_threshold":..., ...}
    inference_config: dict
    device: torch.device


EXPECTED_INIT_COUNT = 3
EXPECTED_PATIENT_FEATURE_DIM = 769
EXPECTED_NUM_SITES = 6


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise CheckpointNotFoundError(
            f"缺少部署产物 {path}; 请先运行 D:\\OPERA\\train_final_deployment_model.py 生成。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_array(name: str, arr: np.ndarray, expected_shape: tuple) -> None:
    if arr.shape != expected_shape:
        raise CheckpointNotFoundError(f"{name} 形状为{arr.shape}, 期望{expected_shape}")
    if not np.isfinite(arr).all():
        raise CheckpointNotFoundError(f"{name} 含有NaN/Inf, 不能用于推理")


def _validate_checkpoint_set(raw_ckpts: List[dict], ckpt_paths: List[Path], model_name: str) -> dict:
    """校验3个初始化互相一致、结构参数符合预期。返回共享的model_config。
    只要有一项对不上就直接拒绝加载——少加载一个模型、或3个初始化配置其实不
    一致, 程序不应该"看起来还能跑"地悄悄退化成不是定义好的3-init ensemble。"""
    if len(raw_ckpts) != EXPECTED_INIT_COUNT:
        raise CheckpointNotFoundError(
            f"{model_name} 应该有{EXPECTED_INIT_COUNT}个初始化的checkpoint, "
            f"实际找到{len(raw_ckpts)}个: {[str(p) for p in ckpt_paths]}"
        )
    configs = []
    for ckpt, path in zip(raw_ckpts, ckpt_paths):
        if ckpt.get("model_name") != model_name:
            raise CheckpointNotFoundError(
                f"{path} 的model_name={ckpt.get('model_name')!r}, 期望{model_name!r}"
            )
        cfg = ckpt.get("model_config")
        if not cfg:
            raise CheckpointNotFoundError(f"{path} 缺少model_config")
        if cfg.get("input_dim") != EXPECTED_PATIENT_FEATURE_DIM:
            raise CheckpointNotFoundError(
                f"{path} 的input_dim={cfg.get('input_dim')}, 期望{EXPECTED_PATIENT_FEATURE_DIM}"
                "(768维OPERA特征+1维segment辅助概率)"
            )
        if cfg.get("num_sites") != EXPECTED_NUM_SITES:
            raise CheckpointNotFoundError(
                f"{path} 的num_sites={cfg.get('num_sites')}, 期望{EXPECTED_NUM_SITES}"
            )
        configs.append(cfg)
    first_cfg = configs[0]
    for cfg, path in zip(configs[1:], ckpt_paths[1:]):
        if cfg != first_cfg:
            raise CheckpointNotFoundError(
                f"{path} 的model_config与第一个初始化不一致(3个初始化必须共用完全"
                f"相同的结构/超参, 只应在initialization_seed和权重上不同): "
                f"{cfg} vs {first_cfg}"
            )
    return first_cfg


def load_deployment_model(
    model_name: str, device: Optional[torch.device] = None
) -> DeploymentModel:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"未知model_name={model_name!r}, 可选: {list(MODEL_BUILDERS)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = DEPLOYMENT_ROOT / model_name
    ckpt_paths = sorted(model_dir.glob("model_init_*.pt"))
    if not ckpt_paths:
        raise CheckpointNotFoundError(
            f"缺少部署checkpoint: {model_dir}\\model_init_*.pt; "
            "请先运行 D:\\OPERA\\train_final_deployment_model.py 生成。"
        )

    inference_config = _load_json(DEPLOYMENT_ROOT / "inference_config.json")
    thresholds_all = _load_json(DEPLOYMENT_ROOT / "thresholds.json")
    if model_name not in thresholds_all:
        raise CheckpointNotFoundError(f"thresholds.json里没有model_name={model_name!r}的阈值条目")
    thresholds = thresholds_all[model_name]

    preproc_path = DEPLOYMENT_ROOT / "preprocessing.npz"
    if not preproc_path.exists():
        raise CheckpointNotFoundError(f"缺少预处理参数: {preproc_path}")
    preproc = np.load(preproc_path, allow_pickle=True)
    feature_mean = preproc["feature_mean"].astype(np.float32)
    feature_std = preproc["feature_std"].astype(np.float32)
    site_order = [str(s) for s in preproc["site_order"]]

    _validate_array("preprocessing.npz的feature_mean", feature_mean, (768,))
    _validate_array("preprocessing.npz的feature_std", feature_std, (768,))
    if site_order != list(inference_config.get("site_order", [])):
        raise CheckpointNotFoundError(
            f"preprocessing.npz的site_order={site_order}与inference_config.json的"
            f"{inference_config.get('site_order')}不一致"
        )

    # 先把7个checkpoint文件全部load成dict做完整性校验, 再决定要不要真的
    # 用它们构建模型——避免"前两个校验通过、第三个才发现结构对不上"时已经
    # 有部分模型对象被构造出来的半成品状态。
    raw_ckpts = [torch.load(p, map_location=device, weights_only=False) for p in ckpt_paths]
    shared_cfg = _validate_checkpoint_set(raw_ckpts, ckpt_paths, model_name)

    models: List[nn.Module] = []
    builder = MODEL_BUILDERS[model_name]
    for ckpt in raw_ckpts:
        cfg = ckpt["model_config"]
        model = builder(
            feat_dim=cfg["input_dim"],
            model_dim=cfg["projection_dim"],
            n_sites=cfg["num_sites"],
            n_layers=cfg["num_layers"],
            n_heads=cfg["num_heads"],
            ffn_dim=cfg["projection_dim"] * 2,
            dropout=cfg["dropout"],
            n_classes=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    seg_ckpt_path = DEPLOYMENT_ROOT / "segment_classifier" / "model.pt"
    seg_config = _load_json(DEPLOYMENT_ROOT / "segment_classifier" / "config.json")
    if not seg_ckpt_path.exists():
        raise CheckpointNotFoundError(f"缺少segment分类器checkpoint: {seg_ckpt_path}")
    seg_ckpt = torch.load(seg_ckpt_path, map_location=device, weights_only=False)
    segment_classifier = nn.Linear(seg_config["input_dim"], seg_config["n_classes"]).to(device)
    segment_classifier.load_state_dict(seg_ckpt["model_state_dict"])
    segment_classifier.eval()

    segment_scaler_mean = np.asarray(seg_ckpt["scaler_mean"], dtype=np.float32)
    segment_scaler_scale = np.asarray(seg_ckpt["scaler_scale"], dtype=np.float32)
    _validate_array("segment分类器的scaler_mean", segment_scaler_mean, (seg_config["input_dim"],))
    _validate_array("segment分类器的scaler_scale", segment_scaler_scale, (seg_config["input_dim"],))
    if np.any(segment_scaler_scale == 0):
        raise CheckpointNotFoundError(
            "segment分类器的scaler_scale含有0, 标准化时会除零, 不能用于推理"
        )

    return DeploymentModel(
        model_name=model_name,
        models=models,
        segment_classifier=segment_classifier,
        segment_scaler_mean=segment_scaler_mean,
        segment_scaler_scale=segment_scaler_scale,
        feature_mean=feature_mean,
        feature_std=feature_std,
        site_order=site_order,
        thresholds=thresholds,
        inference_config=inference_config,
        device=device,
    )


def score_segment_classifier(deployment: DeploymentModel, opera_feature_768: np.ndarray) -> float:
    """OPERA-CT 768维原始特征(不是标准化后的) -> segment辅助分类器给出的
    纤维化概率(即patient级输入的第769维)。"""
    x = (opera_feature_768 - deployment.segment_scaler_mean) / deployment.segment_scaler_scale
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(deployment.device)
    with torch.no_grad():
        logits = deployment.segment_classifier(x_t)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)
