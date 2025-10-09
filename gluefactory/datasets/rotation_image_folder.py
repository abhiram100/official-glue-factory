"""
ImagePairs like dataset for the rotation HPatches dataset.

Read the file paths of the images
Create image pairs, subsample it based on the config
Create the homography according to the angle in the filename
Camera calibration is the identity matrix
Return view0, view1, H_0to1, name and angle
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


class RotationImageFolder(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "root": "hpatches_images",
        "subsample_angle_every": 10,  # Generate images every 10 degrees
        "number_of_images": 10,  # Number of images to sample
        "preprocessing": ImagePreprocessor.default_conf,
        "add_antialias_noise": True,  # Add noise to the images
    }

    def _init(self, conf: dict):
        # List all dirs in the root
        self.images = list((DATA_PATH / conf.root).glob("*.png"))
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.dataset_creation_matrix = self.get_dataset_creation_matrix(
            self.images, conf.subsample_angle_every
        )

    def get_dataset_creation_matrix(self, images, subsample_angle_every):
        """
        Create a list where the first entry is index of the image path and the
        second entry is the angle in degrees.
        """
        dataset_creation_matrix = []
        # Only consider the first `number_of_images` images
        images = images[: self.conf.number_of_images]
        for image_path in images:
            for angle in range(0, 360, subsample_angle_every):
                # Add the image path and angle to the dataset creation matrix
                dataset_creation_matrix.append((image_path, angle))

        self.keys = [image_path.stem for image_path, _ in dataset_creation_matrix]
        self.image_sets = defaultdict(list)
        for image_path, angle in dataset_creation_matrix:
            self.image_sets[image_path.stem].append(image_path)
        return dataset_creation_matrix

    def get_dataset(self, split: str):
        return self

    def _read_view(self, name: Path) -> dict:
        path = DATA_PATH / self.conf.root / name
        img = load_image(path)
        return img

    def __getitem__(self, idx: int) -> dict:
        """
        Get the item at the given index using the dataset creation matrix.
        The first view is the image at the index and the second view is the
        same image rotated by the angle from the dataset creation matrix.
        """
        image_path, angle_deg = self.dataset_creation_matrix[idx]
        data0 = self._read_view(image_path)

        # Convert to numpy for OpenCV operations if needed
        if isinstance(data0, torch.Tensor):
            image = data0.permute(1, 2, 0).numpy()
        else:
            image = data0.copy()

        height, width = image.shape[:2]  # Get image dimensions
        center = (width / 2, height / 2)  # Calculate rotation center

        # Calculate crop dimensions
        d = min(height, width)
        crop = int(d / (2 * 2**0.5))
        crop_x = int(center[0] - crop)
        crop_y = int(center[1] - crop)

        # --- First approach: crop the original image ---
        cropped_orig = image[
            crop_y : crop_y + 2 * crop,
            crop_x : crop_x + 2 * crop,
            :,
        ]

        # --- Second approach: rotate then crop ---
        # For rotation in OpenCV, negative angle rotates clockwise
        rotation_matrix_2x3 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_image = cv2.warpAffine(
            image, rotation_matrix_2x3, (width, height), flags=cv2.INTER_CUBIC
        )

        cropped_rotated = rotated_image[
            crop_y : crop_y + 2 * crop,
            crop_x : crop_x + 2 * crop,
            :,
        ]

        # --- Calculate the correct homography matrix ---
        # Convert angle to radians and negate for clockwise rotation
        angle_rad = -np.deg2rad(angle_deg)
        orig_image_size = [
            cropped_orig.shape[0],
            cropped_orig.shape[1],
        ]
        R_mat = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1],
            ]
        )
        T_mat_0 = np.array(  # Rotate about the center of the original image
            [
                [1, 0, -orig_image_size[0] / 2],
                [0, 1, -orig_image_size[1] / 2],
                [0, 0, 1],
            ]
        )
        T_mat_1 = np.array(  # Rotate about the center of the original image
            [
                [1, 0, orig_image_size[0] / 2],
                [0, 1, orig_image_size[1] / 2],
                [0, 0, 1],
            ]
        )
        H = T_mat_1 @ R_mat @ T_mat_0

        # Convert back to tensor if needed
        if isinstance(data0, torch.Tensor):
            cropped_orig = torch.from_numpy(cropped_orig).permute(2, 0, 1)
            cropped_rotated = torch.from_numpy(cropped_rotated).permute(2, 0, 1)

        data0 = (
            self.preprocessor(cropped_orig)
            if hasattr(self, "preprocessor")
            else cropped_orig
        )
        data1 = (
            self.preprocessor(cropped_rotated)
            if hasattr(self, "preprocessor")
            else cropped_rotated
        )

        H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])

        # Add noise if specified
        if self.conf.add_antialias_noise:
            # Add noise to tensor
            noise0 = torch.randn_like(data0["image"]) * 0.035
            noise1 = torch.randn_like(data1["image"]) * 0.035
            data0["image"] = data0["image"] + noise0
            data1["image"] = data1["image"] + noise1

        # Clip image values to the range [0, 1]
        data0["image"] = torch.clamp(data0["image"], 0.0, 1.0)
        data1["image"] = torch.clamp(data1["image"], 0.0, 1.0)

        # Return result dictionary
        return {
            "view0": {**data0},
            "view1": {**data1},
            "H_0to1": torch.tensor(H).float(),
            "name": str(image_path.stem) + f"_{angle_deg}",
            "angle": angle_deg,
        }

    def __len__(self):
        return len(self.dataset_creation_matrix)


if __name__ == "__main__":
    # Conf from rotation_hpatches.py
    data_conf = {
        "name": "rotation_image_folder",
        "subsample_angle_every": 90,  # Generate images every 10 degrees
    }

    dataset = RotationImageFolder(data_conf)

    import matplotlib.pyplot as plt

    # Get the first image and its rotations through the dataloader
    first_image_path, _ = dataset.dataset_creation_matrix[0]
    first_image_name = first_image_path.stem

    # Calculate the indices for all rotations of the first image
    # Since rotations are grouped by subsample_angle_every, we know the pattern
    rotation_indices = list(range(0, 20))

    # Collect all rotations of the first image
    rotations = []
    angles = []

    for idx in rotation_indices:
        data = dataset[idx]  # Access through dataloader
        rotations.append(data["view1"]["image"].squeeze().permute(1, 2, 0).numpy())
        angles.append(data["angle"])

    # Calculate grid dimensions
    n_images = len(rotations)
    grid_size = int(np.ceil(np.sqrt(n_images)))

    # Plot all rotations
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f"Rotations of image {first_image_name}", fontsize=16)

    for i, (img, angle) in enumerate(zip(rotations, angles)):
        row, col = i // grid_size, i % grid_size
        if grid_size == 1:
            ax = axes
        else:
            ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"Angle: {angle}°")
        ax.axis("off")

    # Turn off any unused subplots
    for i in range(n_images, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        if grid_size > 1:
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
