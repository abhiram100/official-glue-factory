"""ALIKED descriptors at externally supplied pixel-center keypoints."""

import torch
from lightglue import ALIKED

from ..base_model import BaseModel


class ALIKEDDescriptor(BaseModel):
    default_conf = {"model_name": "aliked-n16", "trainable": False}
    required_data_keys = ["image", "keypoints"]

    def _init(self, conf):
        # Glue Factory keypoints place the first pixel center at (0.5, 0.5).
        self.net = ALIKED(model_name=conf.model_name, origin="corner")
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        descriptors = [
            self.net.describe(keypoints[None], view, resize=None)
            for keypoints, view in zip(data["keypoints"], image)
        ]
        return {"descriptors": torch.cat(descriptors)}

    def loss(self, pred, data):
        raise NotImplementedError
