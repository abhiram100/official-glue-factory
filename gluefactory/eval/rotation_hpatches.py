"""
Eval script for the rotation equivariance test on the rotation HPatches dataset
"""

import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets.rotation_image_folder import RotationImageFolder

from .. import logger
from ..datasets import get_dataset
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


class RotationHPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "rotation_image_folder",
            "subsample_angle_every": 10,  # Generate images every 90 degrees
            "number_of_images": 20,  # Number of images to sample
            "preprocessing": {
                "resize": 512,
                "side": "long",  # Square input anyways
                "interpolation": "bilinear",
                "antialias": True,
            },
            "add_antialias_noise": True,  # Add noise to the images
            "batch_size": 1,  # Batch size of 1 for rotation pairs
            "num_workers": os.cpu_count() //2,
        },
        "model": {
            "extractor": {
                "max_num_keypoints": 200,
            },
            "matcher": {
                "name": "matchers.homography_matcher",
                "th_positive": 3,
                "th_negative": 3,
            },
            "ground_truth": {
                "name": "matchers.homography_matcher",
                "th_positive": 3,
                "th_negative": 3,
            },
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": -1,  # -1 runs a bunch of thresholds and selects the best
        },
    }
    export_keys = [
        "keypoints0",
        "keypoints1",
        "proj_0to1",
        "proj_1to0",
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
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
    ]

    def _init(self, conf):
        self.conf = conf

    @classmethod
    def get_dataloader(cls, data_conf=None):
        data_conf = RotationImageFolder.default_conf
        data_conf = OmegaConf.merge(data_conf, cls.default_conf["data"])
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):

        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                if "max_num_keypoints" in self.conf.model.extractor:
                    num_keypoints = default_conf.model.extractor.max_num_keypoints
                    if self.conf.model.extractor.max_num_keypoints != num_keypoints:
                        logger.warning(
                            f"Overriding max_num_keypoints in model config: "
                            f"{self.conf.model.extractor.max_num_keypoints} -> {num_keypoints}"
                        )
                        self.conf.model.extractor.max_num_keypoints = num_keypoints
                        self.conf.model.extractor.max_sampled_keypoints = num_keypoints
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
        assert pred_file.exists()
        results = defaultdict(list)

        conf = self.conf.eval

        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        rep_angle_dict_1 = defaultdict(list)
        rep_angle_dict_2 = defaultdict(list)
        rep_angle_dict_3 = defaultdict(list)

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)

            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here

            # TODO(Abhiram): Consider removing the homography dlt evaluation
            if "keypoints0" in pred:
                results_i = eval_matches_homography(data, pred)
                results_i = {**results_i, **eval_homography_dlt(data, pred)}
            else:
                results_i = {}
            for th in test_thresholds:
                pose_results_i = eval_homography_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # Custom evaluations
            # TODO(Abhiram): Use match radius from config
            rep_loc_err = eval_repeatability_loc_error(
                data,
                pred,
            )
            results_i = {**results_i, **rep_loc_err}

            angle = int(data["angle"].item())
            rep_angle_dict_1[angle].append(rep_loc_err["repeatability@1px"])
            rep_angle_dict_2[angle].append(rep_loc_err["repeatability@2px"])
            rep_angle_dict_3[angle].append(rep_loc_err["repeatability@3px"])

            # we also store the names for later reference
            results_i["angle"] = angle
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["name"][0].split("_")[0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.median(arr), 3)
            summaries[f"mean{k}"] = np.mean(arr)

        auc_ths = [1, 3, 5]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
        )
        if "H_error_dlt" in results.keys():
            dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
            for i, ath in enumerate(auc_ths):
                summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]

        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "homography_recall": plot_cumulative(
                {
                    "DLT": results["H_error_dlt"],
                    self.conf.eval.estimator: results["H_error_ransac"],
                },
                [0, 10],
                unit="px",
                title="Homography ",
            )
        }

        # TODO(Abhiram): Do not save things out here/Save things in a better way
        np.save(experiment_dir / "rep_angle_dict_1.npy", rep_angle_dict_1)
        np.save(experiment_dir / "rep_angle_dict_2.npy", rep_angle_dict_2)
        np.save(experiment_dir / "rep_angle_dict_3.npy", rep_angle_dict_3)

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(RotationHPatchesPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = RotationHPatchesPipeline(conf)

    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    # print results
    pprint(s)
