import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt

from ..geometry.depth import symmetric_reprojection_error
from ..geometry.epipolar import generalized_epi_dist, relative_pose_error
from ..geometry.gt_generation import IGNORE_FEATURE, gt_matches_from_pose_depth
from ..geometry.homography import (
    homography_corner_error,
    is_inside_img,
    sym_homography_error,
)
from ..robust_estimators import load_estimator
from ..utils.tensor import batch_to_device, index_batch
from ..utils.tools import AUCMetric


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        {check_keys_recursive(d[k], v) for k, v in pattern.items()}
    else:
        for k in pattern:
            assert k in d.keys(), "Key {} not found in dict".format(k)


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def eval_per_batch_item(data: dict, pred: dict, eval_f, *args, **kwargs):
    # Batched data
    results = [
        eval_f(data_i, pred_i, *args, **kwargs)
        for data_i, pred_i in zip(index_batch(data), index_batch(pred))
    ]
    # Return a dictionary of lists with the evaluation of each item
    return {k: [r[k] for r in results] for k in results[0].keys()}


def eval_matches_epipolar(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    # match metrics
    n_epi_err = generalized_epi_dist(
        pts0[None],
        pts1[None],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        False,
        essential=True,
    )[0]
    results["epi_prec@1e-4"] = (n_epi_err < 1e-4).float().mean().nan_to_num()
    results["epi_prec@5e-4"] = (n_epi_err < 5e-4).float().mean().nan_to_num()
    results["epi_prec@1e-3"] = (n_epi_err < 1e-3).float().mean().nan_to_num()

    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0

    return results


def eval_matches_depth(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(data["view0"], ["depth", "camera"])
    check_keys_recursive(data["view1"], ["depth", "camera"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)

    camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]
    T_0to1 = data["T_0to1"]

    depth0 = data["view0"]["depth"]
    depth1 = data["view1"]["depth"]

    reproj_error, valid = symmetric_reprojection_error(
        pts0[None],
        pts1[None],
        camera0,
        camera1,
        T_0to1,
        depth0,
        depth1,
    )
    reproj_error, valid = reproj_error[0], valid[0]

    results = {}
    reproj_error = reproj_error[valid].nan_to_num(nan=float("inf"))
    results["reproj_prec@1px"] = (reproj_error < 1).float().mean().nan_to_num().item()
    results["reproj_prec@3px"] = (reproj_error < 3).float().mean().nan_to_num().item()
    results["reproj_prec@5px"] = (reproj_error < 5).float().mean().nan_to_num().item()
    results["covisible"] = valid.float().sum().item()
    results["covisible_percent"] = valid.float().mean().item() * 100.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_pred = gt_matches_from_pose_depth(
        kp0[None].to(device),
        kp1[None].to(device),
        batch_to_device(data, device),
        pos_th=3.0,
        neg_th=5.0,
    )

    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    results["gt_match_recall@3px"] = recall(
        pred["matches0"][None], gt_pred["matches0"].cpu()
    )[0].item()

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    results["gt_match_precision@3px"] = precision(
        pred["matches0"][None], gt_pred["matches0"].cpu()
    )[0].item()
    return results


def eval_matches_homography(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["H_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_matches_homography)

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    err = sym_homography_error(pts0, pts1, H_gt)
    results = {}
    results["prec@1px"] = (err < 1).float().mean().nan_to_num().item()
    results["prec@3px"] = (err < 3).float().mean().nan_to_num().item()
    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0
    return results


def eval_relative_pose_robust(data, pred, conf):
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    T_gt = data["T_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    estimator = load_estimator("relative_pose", conf["estimator"])(conf)
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": data["view0"]["camera"][0],
        "camera1": data["view1"]["camera"][0],
    }
    est = estimator(data_)

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].numpy()
        t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
        results["rel_pose_error"] = max(r_error, t_error)
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results


def eval_homography_robust(data, pred, conf):
    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_relative_pose_robust, conf)

    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {}
    if "keypoints0" in pred:
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, scores0 = pred["matches0"], pred["matching_scores0"]
        pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)
        data_["m_kpts0"] = pts0
        data_["m_kpts1"] = pts1
    if "lines0" in pred:
        if "orig_lines0" in pred:
            lines0 = pred["orig_lines0"]
            lines1 = pred["orig_lines1"]
        else:
            lines0 = pred["lines0"]
            lines1 = pred["lines1"]
        m_lines0, m_lines1, _ = get_matches_scores(
            lines0, lines1, pred["line_matches0"], pred["line_matching_scores0"]
        )
        data_["m_lines0"] = m_lines0
        data_["m_lines1"] = m_lines1

    est = estimator(data_)
    if est["success"]:
        M = est["M_0to1"]
        error_r = homography_corner_error(M, H_gt, data["view0"]["image_size"]).item()
    else:
        error_r = float("inf")

    results = {}
    results["H_error_ransac"] = error_r
    if "inliers" in est:
        inl = est["inliers"]
        results["ransac_inl"] = inl.float().sum().item()
        results["ransac_inl%"] = inl.float().sum().item() / max(len(inl), 1)

    return results


def eval_homography_dlt(data, pred):
    H_gt = data["H_0to1"]
    H_inf = torch.ones_like(H_gt) * float("inf")

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    scores = scores.to(pts0)
    results = {}
    try:
        if H_gt.ndim == 2:
            pts0, pts1, scores = pts0[None], pts1[None], scores[None]
        h_dlt = find_homography_dlt(pts0, pts1, scores)
        if H_gt.ndim == 2:
            h_dlt = h_dlt[0]
    except AssertionError:
        h_dlt = H_inf

    error_dlt = homography_corner_error(h_dlt, H_gt, data["view0"]["image_size"])
    results["H_error_dlt"] = error_dlt.item()
    return results


def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th


def get_tp_fp_pts(pred_matches, gt_matches, pred_scores):
    """
    Computes the True Positives (TP), False positives (FP), the score associated
    to each match and the number of positives for a set of matches.
    """
    assert pred_matches.shape == pred_scores.shape
    ignore_mask = gt_matches != IGNORE_FEATURE
    pred_matches, gt_matches, pred_scores = (
        pred_matches[ignore_mask],
        gt_matches[ignore_mask],
        pred_scores[ignore_mask],
    )
    num_pos = np.sum(gt_matches != -1)
    pred_positives = pred_matches != -1
    tp = pred_matches[pred_positives] == gt_matches[pred_positives]
    fp = pred_matches[pred_positives] != gt_matches[pred_positives]
    scores = pred_scores[pred_positives]
    return tp, fp, scores, num_pos


def AP(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


def aggregate_pr_results(results, suffix=""):
    tp_list = np.concatenate(results["tp" + suffix], axis=0)
    fp_list = np.concatenate(results["fp" + suffix], axis=0)
    scores_list = np.concatenate(results["scores" + suffix], axis=0)
    n_gt = max(results["num_pos" + suffix], 1)

    out = {}
    idx = np.argsort(scores_list)[::-1]
    tp_vals = np.cumsum(tp_list[idx]) / n_gt
    fp_vals = np.cumsum(fp_list[idx]) / n_gt
    out["curve_recall" + suffix] = tp_vals
    out["curve_precision" + suffix] = tp_vals / np.maximum(tp_vals + fp_vals, 1e-9)
    out["AP" + suffix] = AP(tp_vals, fp_vals) * 100
    return out


def get_valid_kpts(proj_kpts, img_size):
    """
    img_size = data["view0"]["image"].shape[-2:] = (h, w)
    """
    valid_mask = is_inside_img(proj_kpts, img_size)
    return proj_kpts[valid_mask]


def eval_repeatability_loc_error(
    data: dict, pred: dict, match_thresholds: list = [1, 2, 3]
) -> dict:
    """
    Evaluate repeatability and localization error for keypoint detection.

    Args:
        data: Dictionary containing view0 and view1 data
        pred: Dictionary containing keypoints, matches, and projections
        match_thresholds: List of pixel thresholds for evaluation

    Returns:
        Dictionary with repeatability and localization error metrics
    """
    # Validate required keys
    check_keys_recursive(data, ["view0", "view1"])
    check_keys_recursive(
        pred,
        [
            "keypoints0",
            "keypoints1",
            "proj_0to1",
            "proj_1to0",
            "matches0",
            "matches1",
        ],
    )

    # Handle batched inputs
    if pred["keypoints0"].ndim > 2:
        return eval_per_batch_item(
            data, pred, eval_repeatability_loc_error, match_thresholds=match_thresholds
        )

    # Extract data
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, m1 = pred["matches0"], pred["matches1"]
    proj0to1, proj1to0 = pred["proj_0to1"], pred["proj_1to0"]
    valid_matches = m0.detach().cpu() > -1

    # Get valid keypoints within image boundaries
    img_shape0 = data["view0"]["image"].shape[-2:]
    img_shape1 = data["view1"]["image"].shape[-2:]

    valid_kpts0 = get_valid_kpts(proj0to1, img_shape1)
    valid_kpts1 = get_valid_kpts(proj1to0, img_shape0)

    # Calculate repeatability
    num_matches0 = (m0 > -1).sum().item()
    num_matches1 = (m1 > -1).sum().item()

    # Avoid division by zero
    num_valid_kpts0 = len(valid_kpts0)
    num_valid_kpts1 = len(valid_kpts1)

    rep0 = num_matches0 / (num_valid_kpts0 + 1e-6)
    rep1 = num_matches1 / (num_valid_kpts1 + 1e-6)
    repeatability = (rep0 + rep1) / 2

    # Calculate localization error
    matched_kp0 = kp0[valid_matches]
    matched_kp1 = kp1[m0[valid_matches]]
    proj_matched_kp0 = proj0to1[valid_matches]
    proj_matched_kp1 = proj1to0[m0[valid_matches]]

    # Compute localization errors
    loc_error0 = np.linalg.norm(
        matched_kp0.cpu().numpy() - proj_matched_kp1.cpu().numpy(), axis=1
    )
    loc_error1 = np.linalg.norm(
        matched_kp1.cpu().numpy() - proj_matched_kp0.cpu().numpy(), axis=1
    )

    # Filter outliers
    filtered_error0 = loc_error0
    filtered_error1 = loc_error1

    # Compute average localization error
    avg_loc_error = (np.mean(filtered_error0) + np.mean(filtered_error1)) / 2

    # Build output dictionary
    results = {"repeatability": repeatability, "localization_error": avg_loc_error}

    # Add threshold-specific metrics
    if match_thresholds is not None:
        for threshold in match_thresholds:
            # Count valid matches within threshold
            valid_error0 = filtered_error0 < threshold
            valid_error1 = filtered_error1 < threshold
            num_valid0 = valid_error0.sum()
            num_valid1 = valid_error1.sum()

            # Calculate threshold-specific repeatability
            threshold_rep0 = num_valid0 / (num_valid_kpts0 + 1e-6)
            threshold_rep1 = num_valid1 / (num_valid_kpts1 + 1e-6)
            results[f"repeatability@{threshold}px"] = (
                threshold_rep0 + threshold_rep1
            ) / 2

            # Calculate threshold-specific localization error
            if num_valid0 > 0 and num_valid1 > 0:
                threshold_loc_error = (
                    filtered_error0[valid_error0].mean()
                    + filtered_error1[valid_error1].mean()
                ) / 2
                results[f"localization_error@{threshold}px"] = threshold_loc_error

    # Sanity check
    if repeatability > 1 or repeatability < 0:
        print(f"Warning: Repeatability {repeatability:.3f} should be in [0, 1]")

    return results
