import logging
import os
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from joblib import Parallel, delayed  # Import Parallel and delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.utils.tensor import batch_to_device

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_matches_epipolar,
    eval_poses,
    eval_relative_pose_robust,
    eval_repeatability_loc_error,
)

logger = logging.getLogger(__name__)


# Define the worker function at the top level for multiprocessing
def _evaluate_single_item_worker_megadepth(item_data_tuple):
    """
    Worker function to evaluate a single data item for MegaDepth in a separate process.
    """
    data_name, data_original, pred_file_path, conf_full, num_kp_list = item_data_tuple

    # Instantiate CacheLoader within the worker process.
    data = batch_to_device(data_original, "cpu")
    cache_loader = CacheLoader({"path": str(pred_file_path), "collate": None}).eval()
    pred_batched = cache_loader(data_original)
    pred = batch_to_device(pred_batched, "cpu")

    results_i = {}

    # add custom evaluations here
    results_i = eval_matches_epipolar(data, pred)

    pose_results_per_item_per_threshold = defaultdict(dict)
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
        pose_results_i = eval_relative_pose_robust(
            data,
            pred,
            {"estimator": conf_full.eval.estimator, "ransac_th": th},
        )
        # Store results for this specific item and threshold
        for k, v in pose_results_i.items():
            pose_results_per_item_per_threshold[th][
                k
            ] = v  # Assign directly, not append

    # we also store the names for later reference
    results_i["names"] = data["name"][0]
    if "scene" in data.keys():
        results_i["scenes"] = data["scene"][0]

    # Repeatability and localization error
    rep_loc_err = eval_repeatability_loc_error(
        data,
        pred,
        match_thresholds=[1, 2, 3],
    )
    results_i = {**results_i, **rep_loc_err}

    return results_i, pose_results_per_item_per_threshold


class ETH3DPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "ETH3D_undistorted_resizedx2",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/ground_truth_depth_dense_9_SAM",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/covisibility/100pairs_5-0.1-0.9.txt",
            "depth_format": "h5",
            "scene_list": None,
            "preprocessing": {
                "resize": 1024,
                "side": "long",
                "interpolation": "area",
                "antialias": False,
            },
            "num_workers": os.cpu_count(),
        },
        "model": {
            "matcher": {
                "name": "depth_matcher",
                "th_positive": 5,
                "th_negative": 5,
            },
            "ground_truth": {
                "name": None,  # no ground truth
            },
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": -1,  # -1 runs a bunch of thresholds and selects the best
            "rank_criterion": "keypoint_scores",  # ranker_scores or keypoint_scores
            "run_ranking": False,  # whether to run ranking eval
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
        "proj_0to1",
        "proj_1to0",
        "ranker_scores0",
        "ranker_scores1",
        "depth_keypoints0",
        "depth_keypoints1",
        "valid_depth_keypoints0",
        "valid_depth_keypoints1",
    ]

    def _init(self, conf):
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

    @classmethod
    def get_dataloader(cls, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else cls.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
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
        """Run the eval on cached predictions"""
        results = defaultdict(list)
        num_kp_list = get_ranking_num_kpts_list(
            self.conf.model.extractor.max_num_keypoints
        )

        pose_results = defaultdict(
            lambda: defaultdict(list)
        )  # This will store aggregated pose results

        num_processes = self.conf.data.num_workers
        if num_processes is None or num_processes <= 0:
            num_processes = os.cpu_count()

        print(f"Running evaluation with {num_processes} processes (using joblib)...")

        processed_results = Parallel(n_jobs=num_processes, verbose=0, backend="loky")(
            delayed(_evaluate_single_item_worker_megadepth)(
                (data_item["name"][0], data_item, pred_file, self.conf, num_kp_list)
            )
            for data_item in tqdm(
                loader, total=len(loader), desc="Evaluating MegaDepth in parallel"
            )
        )

        # Aggregate results from all parallel processes
        for results_i, pose_results_per_item_per_threshold in processed_results:
            for k, v in results_i.items():
                results[k].append(v)

            # Aggregate pose results
            for th, th_results_for_item in pose_results_per_item_per_threshold.items():
                for k, v in th_results_for_item.items():
                    pose_results[th][k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.nanmean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ETH3DPipeline.default_conf)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    # mingle paths
    output_dir = Path(
        EVAL_PATH, "ranking_eth3d" if conf.eval.run_ranking else dataset_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ETH3DPipeline(conf)

    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    if not conf.eval.run_ranking:
        pprint(s)
