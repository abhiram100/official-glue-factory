"""
Standalone script to extract the first image from each HPatches sequence and save as PNG files.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets.hpatches import HPatches
from gluefactory.settings import DATA_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_first_images(
    output_dir=None,
    subset=None,
):
    """
    Extract the first image from each HPatches sequence and save as PNG files.

    Args:
        output_dir (str, optional): Output directory for PNG files. Defaults to DATA_PATH/hpatches_first_images
        subset (str, optional): Only process sequences starting with this character ('i' or 'v')
        ignore_large_images (bool): Whether to ignore large images that were excluded in papers
        grayscale (bool): Whether to process images in grayscale
    """
    # Setup configuration
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "subset": subset,
    }
    conf = OmegaConf.create(conf)

    # Initialize dataset
    dataset = HPatches(conf)

    # Create output directory
    if output_dir is None:
        output_dir = DATA_PATH / "hpatches_images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(
        f"Extracting first images from {len(dataset.sequences)} sequences to {output_dir}"
    )

    successful_extractions = 0
    failed_extractions = 0

    for seq in dataset.sequences:
        # Apply same filtering logic as dataset
        if dataset.conf.ignore_large_images and seq in dataset.ignored_scenes:
            logger.debug(f"Skipping large image sequence: {seq}")
            continue
        if dataset.conf.subset is not None and dataset.conf.subset != seq[0]:
            logger.debug(f"Skipping sequence {seq} (subset filter)")
            continue

        # Read the first image (index 1) from the sequence
        try:
            data = dataset._read_image(seq, 1)
            img = data["image"]

            # Convert from tensor to numpy array and handle channels
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)  # CHW to HWC

            # Convert to 0-255 range if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            # Handle color channel conversion
            if img.ndim == 2:
                # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                # RGB to BGR for cv2.imwrite
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Save as PNG
            output_path = output_dir / f"{seq}.png"
            success = cv2.imwrite(str(output_path), img)

            if success:
                logger.info(f"Saved {seq} -> {output_path}")
                successful_extractions += 1
            else:
                logger.error(f"Failed to write image for sequence {seq}")
                failed_extractions += 1

        except Exception as e:
            logger.error(f"Failed to process sequence {seq}: {e}")
            failed_extractions += 1

    logger.info(
        f"Extraction complete: {successful_extractions} successful, {failed_extractions} failed"
    )
    logger.info(f"Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract first images from HPatches sequences and save as PNG files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_PATH}/hpatches_images",
        help="Output directory for PNG files (default: DATA_PATH/hpatches_images)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["i", "v"],
        help="Only process sequences starting with this character ('i' for illumination, 'v' for viewpoint)",
    )

    args = parser.parse_args()

    extract_first_images(
        output_dir=args.output_dir,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
