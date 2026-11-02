"""RaCo network and sampling, shared by training and inference.

Architecture and inference follow https://github.com/cvg/RaCo.
"""

# Detector architecture adapted from ALIKED (https://github.com/Shiaoming/ALIKED,
# BSD-3-Clause, Copyright (c) 2022 Zhao Xiaoming)
# Sampling adapted from DaD (https://github.com/Parskatt/dad, MIT,
# Copyright (c) 2025 Johan Edstedt)

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class InputPadder:

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = ((self.ht // divis_by + 1) * divis_by - self.ht) % divis_by
        pad_wd = ((self.wd // divis_by + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))
        x = self.gate(self.bn2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate = nn.SELU(inplace=True)
        self.match_dims = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.match_dims(identity)
        out += identity
        out = self.gate(out)
        return out


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


def _extract_patches_from_indices(
    x: torch.Tensor, inds: torch.Tensor, patch_size: int
) -> torch.Tensor:
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    B, H, W = x.shape
    N = inds.shape[1]
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])
    patches = torch.gather(
        unfolded_x, dim=2, index=inds[:, None, :].expand(B, patch_size**2, N)
    )
    return patches


def _covariance_matrix_from_cholesky_elements(
    cholesky_elements_vec: torch.Tensor,
) -> torch.Tensor:
    L11, L21, L22 = torch.unbind(cholesky_elements_vec, dim=-1)
    zeros = torch.zeros_like(L11)
    L = torch.stack(
        [torch.stack([L11, zeros], dim=-1), torch.stack([L21, L22], dim=-1)], dim=-2
    )
    if cholesky_elements_vec.dim() > 1:
        original_shape = cholesky_elements_vec.shape[:-1]
        L_flat = L.view(-1, 2, 2)
        cov_matrix_flat = torch.bmm(L_flat, L_flat.transpose(-1, -2))
        return cov_matrix_flat.view(original_shape + (2, 2))
    else:
        return torch.matmul(L, L.transpose(-1, -2))


def _compute_subpixel_offsets(
    raw_logits: torch.Tensor,
    inds: torch.Tensor,
    nms_radius: int,
    subpixel_temp: float = 0.5,
) -> torch.Tensor:
    B = raw_logits.shape[0]
    device = raw_logits.device
    offset_range = torch.linspace(
        -(nms_radius - 1) / 2, (nms_radius - 1) / 2, nms_radius, device=device
    )
    offset_grid = torch.meshgrid(offset_range, offset_range, indexing="ij")
    offsets = torch.stack((offset_grid[1], offset_grid[0]), dim=-1).reshape(
        nms_radius**2, 2
    )
    offsets = offsets.unsqueeze(0).expand(B, -1, -1)
    keypoint_patch_scores = _extract_patches_from_indices(
        raw_logits.squeeze(1), inds, nms_radius
    )
    keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(dim=1)
    keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
    return keypoint_offsets


def _sample_at_keypoints(
    feature_map: torch.Tensor,
    keypoints: torch.Tensor,
    H: int,
    W: int,
    use_subpixel: bool,
) -> torch.Tensor:
    B, C = feature_map.shape[:2]
    if use_subpixel:
        grid_coords = torch.stack(
            [
                2.0 * keypoints[..., 0] / (W - 1) - 1.0,
                2.0 * keypoints[..., 1] / (H - 1) - 1.0,
            ],
            dim=-1,
        ).unsqueeze(2)
        sampled = F.grid_sample(
            feature_map,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1)
    else:
        idxs = (
            torch.round(keypoints[..., 1]).long() * W
            + torch.round(keypoints[..., 0]).long()
        )
        idxs = torch.clamp(idxs, min=0, max=W * H - 1)
        sampled = feature_map.reshape(B, C, -1).gather(
            2, idxs.unsqueeze(1).expand(B, C, -1)
        )
    sampled = sampled.permute(0, 2, 1)
    return sampled.squeeze(-1) if C == 1 else sampled


class RaCoNet(nn.Module):
    default_conf = {
        "name": "raco",
        "max_num_keypoints": 2048,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "ranker": True,
        "covariance_estimator": True,
        "sort_by_ranker": False,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        if self.conf.nms_radius <= 0 or self.conf.nms_radius % 2 == 0:
            raise ValueError(f"nms_radius must be odd, got {self.conf.nms_radius}")
        if self.conf.max_num_keypoints <= 0:
            raise ValueError(
                f"max_num_keypoints must be positive, got {self.conf.max_num_keypoints}"
            )
        if self.conf.sort_by_ranker and (not self.conf.ranker):
            raise ValueError("Cannot sort by ranker if ranker head is disabled")
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.gate = nn.SELU(inplace=True)
        c1, c2, c3, c4 = (16, 32, 64, 128)
        self.block1 = ConvBlock(3, c1)
        self.block2 = ResBlock(c1, c2)
        self.block3 = ResBlock(c2, c3)
        self.block4 = ResBlock(c3, c4)
        dim = c4
        self.conv1 = conv1x1(c1, dim // 4)
        self.conv2 = conv3x3(c2, dim // 4)
        self.conv3 = conv3x3(c3, dim // 4)
        self.conv4 = conv3x3(c4, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.score_head = nn.Sequential(
            conv1x1(dim, 8),
            nn.SELU(inplace=True),
            conv3x3(8, 4),
            nn.SELU(inplace=True),
            conv3x3(4, 4),
            nn.SELU(inplace=True),
            conv3x3(4, 1),
        )
        if self.conf.ranker:
            ranker_dim = 12
            ranker_layers = [ResBlock(3, ranker_dim)]
            ranker_layers += [ResBlock(ranker_dim, ranker_dim) for _ in range(8)]
            ranker_layers += [
                nn.Conv2d(
                    ranker_dim,
                    1,
                    kernel_size=5,
                    padding=2,
                    bias=True,
                    padding_mode="reflect",
                )
            ]
            self.ranker_head = nn.Sequential(*ranker_layers)
        if self.conf.covariance_estimator:
            modules = []
            in_channels = dim
            for out_channels in [64, 32, 32]:
                modules.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        padding_mode="reflect",
                    )
                )
                modules.append(nn.LeakyReLU(inplace=True))
                in_channels = out_channels
            modules.append(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=1,
                    bias=True,
                    padding_mode="reflect",
                )
            )
            self.covariance_estimator_head = nn.Sequential(*modules)
            self.var_activation = nn.Softplus()
        self.detector_training = False
        self.num_keypoints = self.conf.max_num_keypoints

    def _sampling(
        self,
        keypoint_probs: torch.Tensor,
        nms_radius: int,
        raw_logits: Optional[torch.Tensor] = None,
        subpixel: bool = False,
        subpixel_temp: Optional[float] = None,
    ) -> torch.Tensor:
        if subpixel_temp is None:
            subpixel_temp = self.conf.subpixel_temp
        B, C, H, W = keypoint_probs.size()
        num_kpts = self.num_keypoints
        if self.training and self.detector_training:
            weights = (
                -(torch.linspace(-2, 2, 51, device=keypoint_probs.device) ** 2)
            ).exp()[None, None]
            density = F.conv2d(
                (keypoint_probs + 1e-06) * 10000, weights[..., None, :], padding=(0, 25)
            )
            density = F.conv2d(density, weights[..., None], padding=(25, 0))
            keypoint_probs = keypoint_probs * (density + 1e-08).pow(-0.5)
        max_pooled = F.max_pool2d(
            keypoint_probs, nms_radius, stride=1, padding=nms_radius // 2
        )
        keypoint_probs = keypoint_probs * (keypoint_probs == max_pooled)
        topk = torch.topk(keypoint_probs.reshape(B, H * W), k=num_kpts)
        hw_inds = topk.indices
        h_inds = hw_inds // W
        w_inds = hw_inds % W
        kpts = torch.stack([w_inds.float(), h_inds.float()], dim=-1)
        if subpixel and raw_logits is not None:
            kpts = kpts + _compute_subpixel_offsets(
                raw_logits, hw_inds, nms_radius, subpixel_temp
            )
        return kpts

    def forward(self, data: dict, return_logits=False) -> dict:
        use_subpixel = self.conf.subpixel_sampling and (
            not (self.training and self.detector_training)
        )
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = self.normalizer(image)
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        x = padder.pad(image)
        x1 = self.block1(x)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)
        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))
        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        raw_score_map = self.score_head(x1234)
        raw_score_map = padder.unpad(raw_score_map)
        ranker_map = None
        if self.conf.ranker:
            ranker_map = self.ranker_head(x)
            ranker_map = padder.unpad(ranker_map)
        cholesky_maps = None
        if self.conf.covariance_estimator:
            cholesky_maps = self.covariance_estimator_head(x1234)
            cholesky_maps = padder.unpad(cholesky_maps)
            cholesky_maps = torch.stack(
                [
                    self.var_activation(cholesky_maps[:, 0]),
                    cholesky_maps[:, 1],
                    self.var_activation(cholesky_maps[:, 2]),
                ],
                dim=1,
            )
        keypoint_probs = F.softmax(raw_score_map.flatten(1), dim=1).reshape(
            raw_score_map.size()
        )
        keypoints = self._sampling(
            keypoint_probs=keypoint_probs,
            nms_radius=self.conf.nms_radius,
            raw_logits=raw_score_map,
            subpixel=use_subpixel,
        )
        B, _, H, W = keypoint_probs.size()
        probs = _sample_at_keypoints(keypoint_probs, keypoints, H, W, use_subpixel)
        out_dict = {"keypoints": keypoints + 0.5, "keypoint_scores": probs}
        if self.conf.ranker and ranker_map is not None:
            out_dict["ranker_scores"] = _sample_at_keypoints(
                ranker_map, keypoints, H, W, use_subpixel
            )
        if self.conf.covariance_estimator and cholesky_maps is not None:
            cholesky_scores = _sample_at_keypoints(
                cholesky_maps, keypoints, H, W, use_subpixel
            )
            out_dict["covariances"] = _covariance_matrix_from_cholesky_elements(
                cholesky_scores
            )
        if return_logits:
            out_dict["raw_logits"] = raw_score_map
        if self.conf.sort_by_ranker and "ranker_scores" in out_dict:
            sort_indices = torch.argsort(
                out_dict["ranker_scores"], dim=1, descending=True
            )
            out_dict["keypoints"] = torch.gather(
                out_dict["keypoints"],
                dim=1,
                index=sort_indices[..., None].expand(-1, -1, 2),
            )
            out_dict["keypoint_scores"] = torch.gather(
                out_dict["keypoint_scores"], dim=1, index=sort_indices
            )
            out_dict["ranker_scores"] = torch.gather(
                out_dict["ranker_scores"], dim=1, index=sort_indices
            )
            if "covariances" in out_dict:
                out_dict["covariances"] = torch.gather(
                    out_dict["covariances"],
                    dim=1,
                    index=sort_indices[..., None, None].expand(-1, -1, 2, 2),
                )
        return out_dict
