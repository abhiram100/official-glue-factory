"""Training and inference for RaCo's detector, covariance estimator, and ranker."""

from pathlib import Path

import torch
from torch import nn

from ...geometry.gt_generation import gt_matches_from_homography
from ..base_model import BaseModel
from .raco_model import RaCoNet

WEIGHTS_URL = "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"


def checkpoint_state_dict(checkpoint):
    """Unwrap a public state dictionary or a Glue Factory training checkpoint."""
    state = checkpoint.get("model", checkpoint)
    result = {}
    for key, value in state.items():
        parts = key.split(".")
        while parts[0] in {"module", "_orig_mod", "extractor", "model"}:
            parts.pop(0)
        name = ".".join(parts)
        if name == "_training_step":
            continue
        if name in result:
            raise ValueError(f"Duplicate RaCo parameter: {name}")
        result[name] = value
    return result


class RaCo(BaseModel):
    default_conf = {
        "weights": None,
        "heads_to_train": ["detector"],
        "max_num_keypoints": 2048,
        "max_sampled_keypoints": 512,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "ranker": True,
        "covariance_estimator": True,
        "sort_by_ranker": False,
        "d_max": 1.2,
        "negative_reward_slope": 1e-6,
        "negative_reward_limit": 0.01,
        "ranking_loss_weight": 1.0,
        "match_threshold": 3.0,
    }
    required_data_keys = ["image"]
    strict_conf = True

    def _init(self, conf):
        heads = set(conf.heads_to_train)
        if conf.trainable and not heads:
            raise ValueError("Select at least one training head")
        if conf.max_sampled_keypoints <= 0:
            raise ValueError("max_sampled_keypoints must be positive")
        if conf.subpixel_temp <= 0:
            raise ValueError("subpixel_temp must be positive")
        if not heads <= {"detector", "covariance_estimator", "ranking"}:
            raise ValueError(f"Unknown training heads: {heads}")
        if "ranking" in heads and not conf.ranker:
            raise ValueError("Ranking training requires the ranker head")
        if "covariance_estimator" in heads and not conf.covariance_estimator:
            raise ValueError("Covariance training requires the covariance head")
        if "detector" in heads and len(heads) > 1:
            raise ValueError("Train the detector before training the other heads")
        self.model = RaCoNet(
            max_num_keypoints=conf.max_num_keypoints,
            nms_radius=conf.nms_radius,
            subpixel_sampling=conf.subpixel_sampling,
            subpixel_temp=conf.subpixel_temp,
            ranker=conf.ranker,
            covariance_estimator=conf.covariance_estimator,
            sort_by_ranker=conf.sort_by_ranker,
        )
        self.model.detector_training = conf.trainable and "detector" in heads
        self.register_buffer("_training_step", torch.zeros((), dtype=torch.long))
        self._trainable_modules = []
        for name, module in self.model.named_children():
            head = (
                "ranking"
                if name == "ranker_head"
                else (
                    "covariance_estimator"
                    if name == "covariance_estimator_head"
                    else "detector"
                )
            )
            active = conf.trainable and head in heads
            module.requires_grad_(active)
            if active:
                self._trainable_modules.append(module)
        if conf.weights is not None:
            if str(conf.weights).startswith("https://"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    conf.weights, map_location="cpu", weights_only=True
                )
            else:
                checkpoint = torch.load(
                    Path(conf.weights), map_location="cpu", weights_only=False
                )
            state = checkpoint_state_dict(checkpoint)
            # Disabled heads may still be present in the official full-model weights.
            for name, enabled in (
                ("ranker_head", conf.ranker),
                ("covariance_estimator_head", conf.covariance_estimator),
            ):
                if not enabled:
                    state = {
                        k: v for k, v in state.items() if not k.startswith(name + ".")
                    }
            result = self.model.load_state_dict(state, strict=False)
            new_heads = []
            if conf.trainable:
                if "ranking" in heads:
                    new_heads.append("ranker_head.")
                if "covariance_estimator" in heads:
                    new_heads.append("covariance_estimator_head.")
            new_heads = [
                h for h in new_heads if not any(k.startswith(h) for k in state)
            ]
            missing = [
                k for k in result.missing_keys if not k.startswith(tuple(new_heads))
            ]
            if missing or result.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible RaCo weights: missing={missing}, "
                    f"unexpected={result.unexpected_keys}"
                )
            self.set_initialized()
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.model.training = mode
        if mode:
            for module in self._trainable_modules:
                module.train()
            if self.conf.freeze_batch_normalization:
                for module in self.model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def _forward(self, data):
        self.model.num_keypoints = (
            self.conf.max_sampled_keypoints
            if self.training
            else self.conf.max_num_keypoints
        )
        return self.model(
            data,
            return_logits=self.conf.trainable
            and "detector" in self.conf.heads_to_train,
        )

    def loss(self, pred, data):
        from .raco_losses import (
            covariance_loss,
            detector_loss,
            ranking_loss,
        )

        losses, metrics = {}, {}
        with torch.autocast(device_type=pred["keypoints0"].device.type, enabled=False):
            matches = gt_matches_from_homography(
                pred["keypoints0"].float(),
                pred["keypoints1"].float(),
                data["H_0to1"].float(),
                pos_th=self.conf.match_threshold,
                neg_th=self.conf.match_threshold,
            )
            pred = {
                **pred,
                "gt_matches0": matches["matches0"],
                "gt_matches1": matches["matches1"],
            }
            for head in self.conf.heads_to_train:
                if head == "detector":
                    loss, head_metrics = detector_loss(
                        pred, data, self.conf, int(self._training_step.item())
                    )
                elif head == "covariance_estimator":
                    loss, head_metrics = covariance_loss(pred, data)
                else:
                    loss, head_metrics = ranking_loss(
                        pred, self.conf.ranking_loss_weight
                    )
                losses[head] = loss
                metrics.update(head_metrics)
        if self.training and "detector" in self.conf.heads_to_train:
            self._training_step.add_(1)
        losses["total"] = sum(losses.values())
        return losses, metrics
