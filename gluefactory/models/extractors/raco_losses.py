"""Homography supervision for the three RaCo training stages."""

# Reward and loss adapted from DaD (https://github.com/Parskatt/dad, MIT,
# Copyright (c) 2025 Johan Edstedt)

import math

import torch
import torch.nn.functional as F

from ...geometry.homography import is_inside_img, warp_points_torch


def detector_loss(pred, data, conf, step):
    losses, repeatabilities = [], []
    negative_reward = -min(
        conf.negative_reward_limit, step * conf.negative_reward_slope
    )
    for i, j in ((0, 1), (1, 0)):
        keypoints = pred[f"keypoints{i}"]
        homography = data["H_0to1"].float()
        if i:
            homography = torch.linalg.inv(homography)
        projected = warp_points_torch(keypoints.float(), homography, inverse=False)
        visible = is_inside_img(projected, data[f"view{j}"]["image"].shape[-2:])
        distances = torch.cdist(projected.float(), pred[f"keypoints{j}"].float())
        good = (distances.amin(-1) < conf.d_max).detach()
        positive_reward = good.float() / (good.float().mean(-1, keepdim=True) + 1e-2)
        reward = torch.where(good, positive_reward, negative_reward)
        h, w = data[f"view{i}"]["image"].shape[-2:]
        grid = keypoints.detach() * keypoints.new_tensor([2 / w, 2 / h]) - 1
        logits = (
            F.grid_sample(
                pred[f"raw_logits{i}"].float(), grid[:, None], align_corners=False
            )
            .squeeze(1)
            .squeeze(1)
        )
        # All-invisible pairs contribute a differentiable zero instead of NaNs.
        log_probs = logits.masked_fill(~visible, -torch.inf)
        log_probs = log_probs.masked_fill(~visible.any(-1, keepdim=True), 0)
        log_probs = log_probs.log_softmax(-1).masked_fill(~visible, 0)
        count = visible.sum(-1).clamp_min(1)
        losses.append(-(reward * log_probs).sum(-1) / count)
        matched = pred[f"gt_matches{i}"] >= 0
        repeatabilities.append((matched & visible).sum(-1).float() / count)
    repeatability = torch.stack(repeatabilities).mean(0)
    return sum(losses), {
        "repeatability": repeatability,
        "negative_repeatability": -repeatability,
    }


def homography_jacobian(homography, keypoints):
    """Jacobian of the perspective division at each keypoint."""
    homogeneous = F.pad(keypoints, (0, 1), value=1)
    projected = homogeneous @ homography.transpose(-1, -2)
    denominator = projected[..., 2:3]
    return (
        homography[..., None, :2, :2] * denominator[..., None]
        - projected[..., :2, None] * homography[..., None, 2:3, :2]
    ) / denominator[..., None].square()


def gaussian_nll(delta, covariance):
    covariance = (
        covariance + torch.eye(2, device=delta.device, dtype=delta.dtype) * 1e-6
    )
    chol = torch.linalg.cholesky(covariance)
    residual = torch.linalg.solve_triangular(chol, delta[..., None], upper=False)
    return (
        chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        + residual.square().sum((-1, -2)) / 2
        + math.log(2 * math.pi)
    )


def covariance_loss(pred, data):
    losses = []
    for b, matches in enumerate(pred["gt_matches0"]):
        valid = matches >= 0
        cov0 = pred["covariances0"][b]
        cov1 = pred["covariances1"][b]
        if not valid.any():
            losses.append((cov0.sum() + cov1.sum()) * 0)
            continue
        kp0 = pred["keypoints0"][b, valid].detach().float()
        kp1 = pred["keypoints1"][b, matches[valid]].detach().float()
        cov0, cov1 = cov0[valid].float(), cov1[matches[valid]].float()
        h01 = data["H_0to1"][b].float()
        h10 = torch.linalg.inv(h01)
        j01 = homography_jacobian(h01, kp0)
        j10 = homography_jacobian(h10, kp1)
        delta0 = warp_points_torch(kp1, h10, inverse=False) - kp0
        delta1 = warp_points_torch(kp0, h01, inverse=False) - kp1
        nll0 = gaussian_nll(delta0, cov0 + j10 @ cov1 @ j10.transpose(-1, -2))
        nll1 = gaussian_nll(delta1, cov1 + j01 @ cov0 @ j01.transpose(-1, -2))
        losses.append((nll0.mean() + nll1.mean()) / 2)
    return torch.stack(losses), {}


def ranking_loss(pred, weight):
    from fast_soft_sort.pytorch_ops import soft_rank

    def rank(scores):
        return soft_rank(
            scores.float()[None].cpu(),
            direction="DESCENDING",
            regularization_strength=1.0,
        )[0].to(scores.device)

    def hard_rank(scores):
        return scores.detach().neg().argsort().argsort().float() + 1

    def pullup(ranks, valid):
        partitions = ranks.new_tensor([1, ranks.numel()])[:, None]
        return torch.where(
            valid[None],
            F.relu(ranks[None] - partitions),
            F.relu(partitions - ranks[None]),
        ).mean()

    losses = []
    for b, matches in enumerate(pred["gt_matches0"]):
        valid0 = matches >= 0
        valid1 = pred["gt_matches1"][b] >= 0
        scores0, scores1 = pred["ranker_scores0"][b], pred["ranker_scores1"][b]
        if not valid0.any():
            losses.append((scores0.sum() + scores1.sum()) * 0)
            continue
        matched0, matched1 = scores0[valid0], scores1[matches[valid0]]
        consistency = (
            F.mse_loss(rank(matched0), hard_rank(matched1))
            + F.mse_loss(hard_rank(matched0), rank(matched1))
        ) / matched0.numel()
        penalty = pullup(rank(scores0), valid0) + pullup(rank(scores1), valid1)
        losses.append(consistency + weight * penalty)
    return torch.stack(losses), {}
