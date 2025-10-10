import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from ..utils.tools import AUCMetric
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
    eval_poses,
    eval_repeatability_loc_error,
)


def _evaluate_single_item_worker(item_data_tuple):
    """
    Worker function to evaluate a single data item in a separate process.
    """
    data_name, data_original, pred_file_path, conf_full = item_data_tuple

    data = map_tensor(data_original, lambda t: torch.squeeze(t.cpu().detach(), dim=0))
    cache_loader = CacheLoader({"path": str(pred_file_path), "collate": None}).eval()
    pred_batched = cache_loader(data_original)
    pred = map_tensor(pred_batched, lambda t: torch.squeeze(t.cpu().detach(), dim=0))

    results_i = {}

    if "keypoints0" in pred:
        results_i = eval_matches_homography(data, pred)
        results_i = {**results_i, **eval_homography_dlt(data, pred)}

    pose_results_per_item_per_threshold = {}
    test_thresholds = (
        (
            [conf_full.eval.ransac_th]
            if conf_full.eval.ransac_th > 0
            else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        )
        if not isinstance(conf_full.eval.ransac_th, Iterable)
        else conf_full.eval.ransac_th
    )
    for th in test_thresholds:
        current_pose_results = eval_homography_robust(
            data,
            pred,
            {"estimator": conf_full.eval.estimator, "ransac_th": th},
        )
        pose_results_per_item_per_threshold[th] = current_pose_results

    # we also store the names for later reference
    results_i["names"] = data["name"][0]
    results_i["scenes"] = data["scene"][0]
    results_i["wall_clock_diff"] = data["wall_clock_diff"]

    # Repeatability and localization error
    rep_loc_err = eval_repeatability_loc_error(
        data,
        pred,
        match_thresholds=[1, 2, 3],
    )
    results_i = {**results_i, **rep_loc_err}

    return results_i, pose_results_per_item_per_threshold


class DNIMPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "dnim",
            "num_workers": 2,
            "homography": {
                "difficulty": 0.0,
                "translation": 0.3,
                "max_angle": 25,
                "n_angles": 10,
                "patch_shape": [768, 555],
                "min_convexity": 0.05,
            },
        },
        "model": {
            "matcher": {
                "name": "homography_matcher",
                "th_positive": 3,
                "th_negative": 3,
            },
            "ground_truth": {
                "name": None,
            },
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": -1,
            "rank_criterion": "ranker_scores",
            "run_ranking": False,
        },
    }
    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]

    optional_export_keys = [
        "lines0",
        "lines1",
        "orig_lines0",
        "orig_lines1",
        "proj_0to1",
        "proj_1to0",
        "covariances0",
        "covariances1",
        "precision_scores0",
        "precision_scores1",
        "ranker_scores0",
        "ranker_scores1",
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
    ]

    def _init(self, conf):
        self.conf = conf

    @classmethod
    def get_dataloader(cls, data_conf=None):
        """
        Returns a data loader for the DNIM dataset.
        """
        from gluefactory.datasets.dnim import DNIMDataset

        data_conf = data_conf if data_conf else cls.default_conf["data"]
        dataset = DNIMDataset(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)

            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        assert pred_file.exists(), f"Prediction file not found: {pred_file}"

        results = defaultdict(list)
        pose_results = defaultdict(lambda: defaultdict(list))

        num_processes = self.conf.data.num_workers
        if num_processes is None or num_processes <= 0:
            num_processes = os.cpu_count()

        print(f"Running evaluation with {num_processes} processes (using joblib)...")

        processed_results = Parallel(n_jobs=num_processes, verbose=0, backend="loky")(
            delayed(_evaluate_single_item_worker)(
                (data_item["name"][0], data_item, pred_file, self.conf)
            )
            for data_item in tqdm(
                loader, total=len(loader), desc="Evaluating DNIM in parallel"
            )
        )

        for results_i, pose_results_per_item_per_threshold in processed_results:
            for k, v in results_i.items():
                results[k].append(v)

            for th, th_results in pose_results_per_item_per_threshold.items():
                for k, v in th_results.items():
                    pose_results[th][k].append(v)

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.nanmedian(arr), 3)

        plot_dict = {}
        auc_ths = [1, 3, 5]

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
        )

        plot_dict[self.conf.eval.estimator] = pose_results[best_th]["H_error_ransac"]

        if "H_error_dlt" in results.keys():
            dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
            for i, ath in enumerate(auc_ths):
                summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]
            plot_dict["DLT"] = results["H_error_dlt"]

        for k, v_list in pose_results[best_th].items():
            results[k].extend(v_list)

        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "homography_recall": plot_cumulative(
                plot_dict,
                [0, 10],
                unit="px",
                title="Homography ",
            )
        }

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(DNIMPipeline.default_conf)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    output_dir = Path(
        EVAL_PATH, "ranking_dnim" if conf.eval.run_ranking else dataset_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DNIMPipeline(conf)

    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    if not conf.eval.run_ranking:
        pprint(s)
