"""
DNIM Dataset with Synthetic Homographies

Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)
from ..models.cache_loader import CacheLoader, pad_local_features
from ..settings import DATA_PATH
from ..utils.image import read_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def sample_homography(img, conf: dict, size: list):
    """Sample a homography and warp the image accordingly."""
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)

    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def random_rotations(data, p=0.5):
    """Apply random 90-degree rotations to image and update coordinates."""
    if np.random.rand() > p:
        return data

    # Choose random rotation: 90, 180, or 270 degrees
    angle = np.random.choice([90, 180, 270])
    k = angle // 90

    # Rotate image
    if isinstance(data["image"], torch.Tensor):
        data["image"] = torch.rot90(data["image"], k=k, dims=[1, 2])
    else:
        data["image"] = np.rot90(data["image"], k=k, axes=(0, 1))

    # Update coordinates and homography based on rotation
    h, w = data["image_size"]

    if k == 1:  # 90 degrees
        data["coords"] = np.column_stack(
            [data["coords"][:, 1], w - 1 - data["coords"][:, 0]]
        )
        data["image_size"] = np.array([h, w], dtype=np.float32)
        rot_mat = np.array([[0, -1, w - 1], [1, 0, 0], [0, 0, 1]])
    elif k == 2:  # 180 degrees
        data["coords"] = np.column_stack(
            [w - 1 - data["coords"][:, 0], h - 1 - data["coords"][:, 1]]
        )
        rot_mat = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]])
    elif k == 3:  # 270 degrees
        data["coords"] = np.column_stack(
            [h - 1 - data["coords"][:, 1], data["coords"][:, 0]]
        )
        data["image_size"] = np.array([h, w], dtype=np.float32)
        rot_mat = np.array([[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]])

    data["H_"] = np.matmul(rot_mat, data["H_"])
    return data


class DNIMDataset(BaseDataset):
    default_conf = {
        # Data paths
        "data_dir": "DNIM/",
        "image_dir": "Image/",
        "time_stamp_dir": "time_stamp/",
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # Image processing
        "grayscale": False,
        "right_only": True,
        "reseed": False,
        "p_random_rotations": 0,
        # Homography configuration
        "homography": {
            "difficulty": 0.8,
            "translation": 0.5,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [768, 555],
            "min_convexity": 0.05,
        },
        # Photometric augmentation
        "photometric": {
            "name": "identity",
        },
        # Feature loading
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
            "thresh": 0.0,
            "max_num_keypoints": -1,
            "force_num_keypoints": False,
        },
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)

        self.image_dir = data_dir / conf.image_dir
        seed = getattr(conf, "pair_seed", 42)
        self.pairs = self.create_pairs(seed=seed)
        print(f"Created {len(self.pairs)} pairs of images.")

    def get_dataset(self, split):
        return _Dataset(self.conf, self.pairs, self.image_dir)

    def create_pairs(
        self,
        N_same=3,
        N_other=4,
        K=4,
        S=30,
        min_time_diff=1800,
        max_attempts=5,
        seed=0,
    ):
        """Create pairs of images based on timestamps."""
        if seed is not None:
            np.random.seed(seed)

        folder_image_timestamps = self._load_timestamps()
        pairs = []

        for folder_name, image_timestamps in folder_image_timestamps.items():
            image_timestamps.sort(key=lambda x: x[1])

            if len(image_timestamps) < K:
                print(
                    f"Skipping folder {folder_name}: only {len(image_timestamps)} "
                    f"images, need at least {K}"
                )
                continue

            # Generate same-cluster pairs
            pairs.extend(
                self._generate_same_cluster_pairs(
                    folder_name,
                    image_timestamps,
                    N_same,
                    K,
                    S,
                    min_time_diff,
                    max_attempts,
                )
            )

            # Generate different-cluster pairs
            pairs.extend(
                self._generate_different_cluster_pairs(
                    folder_name,
                    image_timestamps,
                    N_other,
                    K,
                    min_time_diff,
                    max_attempts,
                )
            )

        return pairs

    def _load_timestamps(self):
        """Load timestamp data from files."""
        folder_image_timestamps = {}
        timestamp_dir = DATA_PATH / self.conf.data_dir / self.conf.time_stamp_dir

        for file in os.listdir(timestamp_dir):
            if not file.endswith(".txt"):
                continue

            folder_name = file.split(".")[0]
            folder_image_timestamps[folder_name] = []

            with open(timestamp_dir / file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    image_file = parts[0]
                    timestamp = (
                        int(parts[1]) * 3600 + int(parts[2]) * 60 + int(parts[3])
                    )
                    folder_image_timestamps[folder_name].append((image_file, timestamp))

        return folder_image_timestamps

    def _generate_same_cluster_pairs(
        self, folder_name, image_timestamps, N_same, K, S, min_time_diff, max_attempts
    ):
        """Generate pairs within the same timestamp cluster."""
        pairs = []

        for _ in range(N_same):
            for i in range(0, len(image_timestamps) - S + 1, S):
                if i + S > len(image_timestamps):
                    continue

                sampled_indices = np.random.choice(
                    range(i, min(i + S, len(image_timestamps))),
                    min(K, len(image_timestamps) - i),
                    replace=False,
                )

                for idx1, idx2 in zip(sampled_indices[:-1], sampled_indices[1:]):
                    for _ in range(max_attempts):
                        t1, t2 = image_timestamps[idx1][1], image_timestamps[idx2][1]
                        if abs(t1 - t2) >= min_time_diff:
                            pairs.append(
                                (
                                    f"{folder_name}/{image_timestamps[idx1][0]}",
                                    f"{folder_name}/{image_timestamps[idx2][0]}",
                                    "same",
                                    t1,
                                    t2,
                                )
                            )
                            break
        return pairs

    def _generate_different_cluster_pairs(
        self, folder_name, image_timestamps, N_other, K, min_time_diff, max_attempts
    ):
        """Generate pairs across different timestamp clusters."""
        pairs = []

        for _ in range(N_other):
            for _ in range(K):
                for _ in range(max_attempts):
                    idx1 = np.random.randint(0, len(image_timestamps))
                    candidates = [
                        i
                        for i in range(len(image_timestamps))
                        if abs(image_timestamps[i][1] - image_timestamps[idx1][1])
                        >= min_time_diff
                    ]

                    if candidates:
                        idx2 = np.random.choice(candidates)
                        t1, t2 = image_timestamps[idx1][1], image_timestamps[idx2][1]
                        if abs(t1 - t2) >= min_time_diff:
                            pairs.append(
                                (
                                    f"{folder_name}/{image_timestamps[idx1][0]}",
                                    f"{folder_name}/{image_timestamps[idx2][0]}",
                                    "different",
                                    t1,
                                    t2,
                                )
                            )
                            break
        return pairs


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, pairs, image_dir):
        self.conf = conf
        self.pairs = pairs
        self.image_dir = image_dir

        self._setup_augmentations()

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

    def _setup_augmentations(self):
        """Initialize augmentation functions."""
        aug_conf = self.conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'

        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if self.conf.right_only else self.photo_augment
        )

    def _transform_keypoints(self, features, data):
        """Transform keypoints by homography and filter them."""
        # Warp points
        features["keypoints"] = warp_points(
            features["keypoints"], data["H_"], inverse=False
        )

        # Filter valid points
        h, w = data["image"].shape[1:3]
        valid = (
            (features["keypoints"][:, 0] >= 0)
            & (features["keypoints"][:, 0] <= w - 1)
            & (features["keypoints"][:, 1] >= 0)
            & (features["keypoints"][:, 1] <= h - 1)
        )
        features["keypoints"] = features["keypoints"][valid]

        # Apply score threshold
        if self.conf.load_features.thresh > 0:
            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
            features = {k: v[valid] for k, v in features.items()}

        # Keep top keypoints
        n = self.conf.load_features.max_num_keypoints
        if n > -1:
            inds = np.argsort(-features["keypoint_scores"])
            features = {k: v[inds[:n]] for k, v in features.items()}

            if self.conf.load_features.force_num_keypoints:
                features = pad_local_features(
                    features, self.conf.load_features.max_num_keypoints
                )

        return features

    def _read_view(self, img, H_conf, ps, left=False):
        """Process a single view with homography and augmentations."""
        data = sample_homography(img, H_conf, ps)

        # Apply photometric augmentation
        if left:
            data["image"] = self.left_augment(data["image"], return_tensor=True)
        else:
            data["image"] = self.photo_augment(data["image"], return_tensor=True)

        # Apply random rotations
        if self.conf.p_random_rotations > 1e-3 and not (left and self.conf.right_only):
            data = random_rotations(data, p=self.conf.p_random_rotations)

        # Convert to grayscale if needed
        if self.conf.grayscale:
            gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)

        # Load and transform features if needed
        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            features = self._transform_keypoints(features, data)
            data["cache"] = features

        return data

    def _load_image(self, name):
        """Load and preprocess an image."""
        img = read_image(self.image_dir / name, False)
        if img is None:
            logging.warning("Image %s could not be read.", name)
            img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        return img.astype(np.float32) / 255.0

    def getitem(self, idx):
        """Get a single data item."""
        name0, name1, scene, ts0, ts1 = self.pairs[idx]

        # Load images
        img0 = self._load_image(name0)
        img1 = self._load_image(name1)

        # Calculate time difference
        diff = abs(ts0 - ts1)
        wall_clock_diff = min(diff, abs(24 * 3600 - diff)) / 3600.0

        # Setup homography configurations
        size = img0.shape[:2][::-1]
        ps = self.conf.homography.patch_shape

        left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
        if self.conf.right_only:
            left_conf["difficulty"] = 0.0

        # Process both views
        data0 = self._read_view(img0, left_conf, ps, left=True)
        data1 = self._read_view(img1, self.conf.homography, ps, left=False)

        # Add scales
        data0["scales"] = torch.tensor(1.0)
        data1["scales"] = torch.tensor(1.0)

        # Compute homography between views
        H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        name = (
            f"{name0.split('/')[-1].split('.')[0]}_{name1.split('/')[-1].split('.')[0]}"
        )

        # Create final data structure
        return {
            "name": name,
            "original_image_size": np.array(size),
            "H_0to1": H.astype(np.float32),
            "idx": idx,
            "scene": scene,
            "view0": data0,
            "view1": data1,
            "wall_clock_diff": wall_clock_diff,
        }

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def __len__(self):
        return len(self.pairs)


def visualize(args):
    """Visualize dataset samples."""
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "homography": {
            "difficulty": 0.5,
            "translation": 0.5,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [768, 555],
            "min_convexity": 0.05,
        },
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = DNIMDataset(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    for run_idx in range(10):
        with fork_rng(seed=dataset.conf.seed + run_idx):
            images = []
            for i, data in zip(range(args.num_items * 10), loader):
                images.append(
                    [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
                )
                if i % args.num_items == args.num_items - 1:
                    plt.close()
                    plot_image_grid(images, dpi=args.dpi)
                    plt.tight_layout()
                    plt.show()
                    images = []


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
