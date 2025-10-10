import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..settings import EVAL_PATH
from . import get_benchmark
from .eval_pipeline import load_eval

# Moving average window size
MOV_AVG_WINDOW_SIZE = 5  # Samples, not degrees
MATCH_THRESHOLD_PX = "2"  # px

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("dotlist", nargs="*")
    parser.add_argument(
        "--match-threshold-px",
        type=str,
        default=MATCH_THRESHOLD_PX,
        help="Match threshold in pixels for repeatability evaluation.",
    )
    args = parser.parse_intermixed_args()

    if args.match_threshold_px not in ["1", "2", "3"]:
        raise ValueError(
            f"Invalid match threshold: {args.match_threshold_px}. Must be one of ['1', '2', '3']."
        )
    MATCH_THRESHOLD_PX = args.match_threshold_px

    output_dir = Path(EVAL_PATH, args.benchmark)

    results = {}
    summaries = defaultdict(dict)
    rep_angle_dict = {}

    predictions = {}

    bm = get_benchmark(args.benchmark)
    loader = bm.get_dataloader()

    for name in args.dotlist:
        experiment_dir = output_dir / name
        pred_file = experiment_dir / "predictions.h5"
        s, results[name] = load_eval(experiment_dir)
        predictions[name] = pred_file
        for k, v in s.items():
            summaries[k][name] = v
        rep_angle_dict[name] = np.load(
            experiment_dir / f"rep_angle_dict_{MATCH_THRESHOLD_PX}.npy",
            allow_pickle=True,
        ).item()

    for name, angle_dict in rep_angle_dict.items():
        for angles, rep in angle_dict.items():
            rep = np.array(rep)
            rep_angle_dict[name][angles] = rep.mean()

    aucs = {}
    plt.figure(constrained_layout=True, figsize=(8, 5))
    for name, rep_angle in rep_angle_dict.items():
        # Apply moving average filter
        angles = np.array(list(rep_angle.keys()))
        values = np.array(list(rep_angle.values()))

        # Pad the array to make it circular
        angles = np.concatenate([angles - 360, angles, angles + 360])
        values = np.concatenate([values, values, values])

        # Pad the array so that the moving average doesn't cut off the ends
        angles = np.concatenate(
            [angles[-MOV_AVG_WINDOW_SIZE:], angles, angles[:MOV_AVG_WINDOW_SIZE]]
        )
        values = np.concatenate(
            [values[-MOV_AVG_WINDOW_SIZE:], values, values[:MOV_AVG_WINDOW_SIZE]]
        )

        # Apply moving average filter
        smoothed_values = np.convolve(
            values, np.ones(MOV_AVG_WINDOW_SIZE) / MOV_AVG_WINDOW_SIZE, mode="valid"
        )
        smoothed_angles = angles[MOV_AVG_WINDOW_SIZE // 2 : -(MOV_AVG_WINDOW_SIZE // 2)]

        # Remove padded values
        smoothed_angles = smoothed_angles[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        smoothed_values = smoothed_values[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        angles = angles[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        values = values[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]

        # Pad the array so that the moving average doesn't cut off the ends
        angles = np.concatenate(
            [angles[-MOV_AVG_WINDOW_SIZE:], angles, angles[:MOV_AVG_WINDOW_SIZE]]
        )
        values = np.concatenate(
            [values[-MOV_AVG_WINDOW_SIZE:], values, values[:MOV_AVG_WINDOW_SIZE]]
        )

        # Apply moving average filter
        smoothed_values = np.convolve(
            values, np.ones(MOV_AVG_WINDOW_SIZE) / MOV_AVG_WINDOW_SIZE, mode="valid"
        )
        smoothed_angles = angles[MOV_AVG_WINDOW_SIZE // 2 : -(MOV_AVG_WINDOW_SIZE // 2)]

        # Remove padded values
        smoothed_angles = smoothed_angles[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        smoothed_values = smoothed_values[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        angles = angles[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]
        values = values[MOV_AVG_WINDOW_SIZE:-MOV_AVG_WINDOW_SIZE]

        # Plot raw values (grey line) first
        plt.plot(angles, values, alpha=0.4, color="gray", zorder=1)

        # Plot smoothed values (colored line) on top
        plt.plot(
            smoothed_angles,
            smoothed_values,
            label=name,
            linewidth=2.5,
            zorder=2,
        )

        # Calculate the AUC but normalized to the range [0, 1] for the x axis
        # Do it only for angles in the range [0, 360]
        mask = (smoothed_angles >= 0) & (smoothed_angles <= 360)
        final_smoothed_angles = smoothed_angles[mask]
        final_smoothed_values = smoothed_values[mask]
        auc = np.trapz(final_smoothed_values, final_smoothed_angles / 360)
        aucs[name] = auc

    print("\nAUC for each method (at threshold px = {})".format(MATCH_THRESHOLD_PX))
    for name, auc in aucs.items():
        print(f"{name}: {auc:.4f}")

    # Add dotted lines at 0, 90, 180, 270, 360
    plt.xlabel("Rotation Angle (degrees)", fontsize=16)
    plt.ylabel("Average Repeatability", fontsize=16)
    plt.title(f"Repeatability vs Rotation Angle @ {MATCH_THRESHOLD_PX} px", fontsize=20)
    plt.xticks(np.arange(-360, 719, 45), fontsize=12)
    plt.xlim(0, 360)
    plt.ylim(0, 1.1)
    plt.grid(axis="x")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
