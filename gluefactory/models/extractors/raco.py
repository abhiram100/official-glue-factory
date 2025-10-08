from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from ..base_model import BaseModel
from .raco_model import RacoModel
import logging

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TanhTimesN(nn.Module):
    """
    Custom activation function that applies N * tanh(x) to the input.
    This is used to ensure the output is in the range [-N, N].
    """

    def __init__(self, N=1.0):
        super().__init__()
        self.N = N

    def forward(self, x):
        return self.N * torch.tanh(x)

    def extra_repr(self):
        return f"N={self.N}"


def get_grid(B, H, W, device):
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


class RaCo(BaseModel):
    default_conf = {
        "name": "raco",
        "weights": None,
        "trainable": False,
        "max_num_keypoints": 512,  # Inference
        "nms_radius": 3,  # px
        "inference_sampling": "balanced",
        "subpixel_sampling": True,  # TODO(Abhiram): handle training and inference separately
        "reward_function": "strek",  # strek, mnn
        "binary_reward": True,
        "detection_threshold": -1,  # Threshold for keypoint detection
        "ranker": True,
        "covariance_estimator": True,
    }

    def _init(self, conf):  # type: ignore

        self.conf = SimpleNamespace(**conf)
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.detection_threshold = self.conf.detection_threshold

        # Set ranker and covariance_estimator flags for model initialization
        self.ranker = self.conf.ranker
        self.covariance_estimator = self.conf.covariance_estimator

        self._model_init()

        # Load the weights if provided
        if conf.weights is not None:
            state_dict = torch.load(conf.weights, map_location="cpu")["model"]
            self.load_state_dict(state_dict, strict=True)
            logging.info(f"[RaCo] Loaded weights from {conf.weights}")

    def _model_init(self):
        self.model = RacoModel(
            {
                "ranker": self.ranker,
                "covariance_estimator": self.covariance_estimator,
            }
        )

    def _balanced_sampling(
        self,
        keypoint_probs: torch.Tensor,
        nms_radius,
        num_kpts=None,
        raw_logits: Optional[torch.Tensor] = None,
        subpixel: bool = False,
        subpixel_temp: float = 0.5,
    ):
        if num_kpts is None:
            num_kpts = (
                self.conf.max_sampled_keypoints
                if self.training
                else self.conf.max_num_keypoints
            )

        def to_pixel_coords(normalized_coords, h, w) -> torch.Tensor:
            if normalized_coords.shape[-1] != 2:
                raise ValueError(
                    f"Expected shape (..., 2), but got {normalized_coords.shape}"
                )
            pixel_coords = torch.stack(
                (
                    w * (normalized_coords[..., 0] + 1) / 2,
                    h * (normalized_coords[..., 1] + 1) / 2,
                ),
                dim=-1,
            )
            return pixel_coords

        B, C, H, W = keypoint_probs.size()
        device = keypoint_probs.device

        increase_coverage = self.training
        if increase_coverage:
            coverage_pow = 1 / 2
            coverage_size = 51
            weights = (
                -(torch.linspace(-2, 2, steps=coverage_size, device=device) ** 2)
            ).exp()[None, None]
            # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
            local_density_x = F.conv2d(
                (keypoint_probs + 1e-6) * 10000,
                weights[..., None, :],
                padding=(0, coverage_size // 2),
            )
            local_density = F.conv2d(
                local_density_x, weights[..., None], padding=(coverage_size // 2, 0)
            )[:, 0]
            keypoint_probs = keypoint_probs * (local_density.unsqueeze(1) + 1e-8) ** (
                -coverage_pow
            )
        grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)

        assert nms_radius % 2 == 1, "nms_radius should be odd"
        keypoint_probs = keypoint_probs * (
            keypoint_probs
            == F.max_pool2d(
                keypoint_probs, nms_radius, stride=1, padding=nms_radius // 2
            )
        )

        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_kpts).indices
        kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_kpts, 2))
        # TODO(Abhiram): handle training and inference separately
        if subpixel:
            offsets = get_grid(B, nms_radius, nms_radius, device=device).reshape(
                B, nms_radius**2, 2
            )  # B x K_H x K_W x 2
            offsets[..., 0] = offsets[..., 0] * nms_radius / W
            offsets[..., 1] = offsets[..., 1] * nms_radius / H
            keypoint_patch_scores = extract_patches_from_inds(
                raw_logits.squeeze(1), inds, nms_radius
            )
            keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(
                dim=1
            )  # B x K_H * K_W x N
            keypoint_offsets = torch.einsum(
                "bkn, bkd ->bnd", keypoint_patch_probs, offsets
            )
            kps_subpixel = kps + keypoint_offsets
            kps_subpixel = to_pixel_coords(kps_subpixel, H, W) - 0.5

        # Convert to pixel coordinates
        kps = to_pixel_coords(kps, H, W) - 0.5
        # To indices
        idxs = kps[..., 1] * W + kps[..., 0]
        idxs = idxs.long()
        idxs = torch.clamp(idxs, min=0, max=W * H - 1)

        if subpixel:
            kps = kps_subpixel

        return idxs, kps

    def _forward(self, data):

        if "total_n_samples" in data:  # TODO(Abhiram): Any other way to do this?
            # Print only once that the constants are being updated
            if not hasattr(self, "_printed_constants_update"):
                print("Updating constants...")
                self._printed_constants_update = True
            self._update_constants(data)

        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # Convert to 3-channel greyscale
        image = self.normalizer(image)  # imagenet normalization the image

        raw_score_map, ranker_map, cov_maps = self.model(image)

        # Batchwise global softmax normalization
        logx = nn.functional.log_softmax(raw_score_map.flatten(1), dim=1).reshape(
            raw_score_map.size()
        )
        x = torch.exp(logx)

        kps = None
        idxs, kps = self._balanced_sampling(
            keypoint_probs=x,
            nms_radius=self.conf.nms_radius,
            raw_logits=raw_score_map,
            subpixel=self.conf.subpixel_sampling,
        )

        B, _, H, W = x.size()
        probs = x.view(B, -1).gather(1, idxs.view(B, -1))

        if self.conf.detection_threshold > 0:
            mask = probs > self.conf.detection_threshold
            idxs = idxs[mask].view(B, -1)
            probs = probs[mask].view(B, -1)

        xs = idxs % W
        ys = idxs // W
        keypoints = torch.stack([xs, ys], dim=-1).float() if kps is None else kps

        # Gather the probabilities and log probabilities
        log_probs = logx.view(B, -1).gather(1, idxs.view(B, -1))
        raw_keypoint_scores = raw_score_map.reshape(B, -1).gather(1, idxs.view(B, -1))

        ranker_scores = None
        if self.conf.ranker:
            ranker_scores = ranker_map.reshape(B, -1).gather(1, idxs.view(B, -1))
            ranker_scores = ranker_scores.view(B, -1)

        cholesky_scores, means = None, None
        if self.conf.covariance_estimator and cov_maps is not None:
            # Apply activations to ensure L11 and L22 are positive
            # cov_maps has shape (B, 5, H, W) with [L11_prime, L21, L22_prime, mean_x, mean_y]
            var_activation = nn.Softplus()

            cov_maps = torch.stack(
                [
                    var_activation(cov_maps[:, 0, ...]),  # L11
                    cov_maps[:, 1, ...],  # L21 (no constraint)
                    var_activation(cov_maps[:, 2, ...]),  # L22
                ],
                dim=1,
            )
            # Sample the cholesky elements at the keypoints
            cholesky_scores = (
                cov_maps.view(B, 3, -1)
                .gather(2, idxs.unsqueeze(1).expand(B, 3, -1))
                .permute(0, 2, 1)
            )  # (B, N, 3)

        out_dict = {
            # Heatmaps
            "heatmap": x,
            "raw_logits": raw_score_map,
            "ranker_map": ranker_map,
            "cholesky_elements_heatmap": (
                cov_maps[:, :3, ...] if self.conf.covariance_estimator else None
            ),
            "means_heatmap": cov_maps[:, 3:, ...],
            # Points
            "keypoints": keypoints + 0.5,  # Add 0.5 to center the keypoints
            "keypoint_scores": probs,
            "log_keypoint_scores": log_probs,
            "raw_keypoint_scores": raw_keypoint_scores,
            "ranker_scores": ranker_scores,
            "cholesky_scores": cholesky_scores,
            "means": means,
        }

        return out_dict

    def loss(self):
        return NotImplementedError


if __name__ == "__main__":
    model = RaCo(RaCo.default_conf)
    print(model)
