import unittest

import numpy as np
import torch
from PIL import Image

from glyph_sr.images import degrade_image, pil_to_tensor, tensor_to_pil


class ImageTests(unittest.TestCase):
    def test_degradation_matches_two_bicubic_resizes(self):
        array = np.arange(30 * 24 * 3, dtype=np.uint8).reshape(24, 30, 3)
        image = Image.fromarray(array)
        expected = image.resize((10, 8), Image.Resampling.BICUBIC).resize(
            image.size, Image.Resampling.BICUBIC
        )
        np.testing.assert_array_equal(degrade_image(image, 3), expected)

    def test_input_rounding_and_requested_output_size_are_distinct(self):
        tensor, height, width = pil_to_tensor(
            Image.new("RGB", (80, 40), (0, 128, 255)), upscale=2, min_size=64
        )
        self.assertEqual(tensor.shape, (3, 64, 128))
        self.assertEqual((height, width), (80, 160))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([-1, 128 / 255 * 2 - 1, 1]))
        self.assertEqual(tensor_to_pil(tensor, height, width).size, (160, 80))

    def test_minimum_size_and_fixed_resize_preserve_aspect_ratio(self):
        image = Image.new("RGB", (96, 48))
        minimum, height, width = pil_to_tensor(image, min_size=128)
        self.assertEqual(minimum.shape, (3, 128, 256))
        self.assertEqual((height, width), (48, 96))
        fixed, height, width = pil_to_tensor(image, min_size=128, fix_resize=64)
        self.assertEqual(fixed.shape, (3, 64, 128))
        self.assertEqual((height, width), (64, 128))

    def test_impossible_degradation_fails_with_an_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "positive image dimensions"):
            degrade_image(Image.new("RGB", (4, 4)), 10)


if __name__ == "__main__":
    unittest.main()
