import unittest

import numpy as np
import torch

from gluefactory.datasets.homographies import rotate_quarter_turns
from gluefactory.geometry.homography import warp_points


class TestHomographies(unittest.TestCase):
    def test_rotate_quarter_turns(self):
        size = 5
        image = torch.arange(3 * size * size, dtype=torch.float32)
        image = image.reshape(3, size, size)
        H = np.array([[1, 0.1, 2], [0, 1, -1], [0, 0, 1]], dtype=np.float32)
        coords = np.array([[0, 0], [0, 4], [4, 4], [4, 0]], dtype=np.float32)
        xy = np.stack(np.meshgrid(range(size), range(size)), -1).reshape(-1, 2)

        for k in range(4):
            data = rotate_quarter_turns({"image": image, "H_": H, "coords": coords}, k)
            # The update applied to H_ must map each pixel to its rotated position.
            R = data["H_"] @ np.linalg.inv(H)
            xy_rot = np.rint(warp_points(xy.astype(np.float32), R, inverse=False))
            xy_rot = xy_rot.astype(int)
            torch.testing.assert_close(
                data["image"][:, xy_rot[:, 1], xy_rot[:, 0]],
                image[:, xy[:, 1], xy[:, 0]],
            )
            np.testing.assert_allclose(
                data["coords"], warp_points(coords, R, inverse=False), atol=1e-5
            )
            np.testing.assert_array_equal(data["image_size"], [size, size])

        with self.assertRaises(AssertionError):
            rotate_quarter_turns({"image": torch.zeros(3, 4, 6)}, 1)


if __name__ == "__main__":
    unittest.main()
