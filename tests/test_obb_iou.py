from __future__ import annotations

import unittest

import torch

from orsgg.ops.rotated_iou import obb5_to_poly, obb_iou, obb_iou_python


class TestOBBIoU(unittest.TestCase):
    def test_identical_axis_aligned(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        iou = obb_iou_python(boxes, boxes)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=6)

    def test_disjoint(self) -> None:
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        b = torch.tensor([[5.0, 0.0, 2.0, 2.0, 0.0]])
        iou = obb_iou_python(a, b)
        self.assertAlmostEqual(float(iou[0, 0]), 0.0, places=6)

    def test_partial_axis_aligned(self) -> None:
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 2.0, 2.0, 0.0]])
        iou = obb_iou_python(a, b)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 3.0, places=5)

    def test_poly_and_obb5_agree(self) -> None:
        a5 = torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.3]])
        b5 = torch.tensor([[11.0, 9.0, 7.0, 3.0, -0.2]])
        a8 = obb5_to_poly(a5)
        b8 = obb5_to_poly(b5)
        iou5 = obb_iou_python(a5, b5)
        iou8 = obb_iou_python(a8, b8)
        self.assertTrue(torch.allclose(iou5, iou8, atol=1e-6))

    def test_extension_matches_reference_when_available(self) -> None:
        a = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 10.0, 8.0, 4.0, 0.3],
            ],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [
                [1.0, 0.0, 2.0, 2.0, 0.0],
                [11.0, 9.0, 7.0, 3.0, -0.2],
            ],
            dtype=torch.float32,
        )
        ref = obb_iou_python(a, b)
        fast = obb_iou(a, b)
        self.assertTrue(torch.allclose(ref, fast.cpu(), atol=1e-5, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_extension_matches_reference(self) -> None:
        a = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 10.0, 8.0, 4.0, 0.3],
                [2.0, -1.0, 5.0, 20.0, 1.2],
            ],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [
                [1.0, 0.0, 2.0, 2.0, 0.0],
                [11.0, 9.0, 7.0, 3.0, -0.2],
                [-2.0, 3.0, 3.0, 17.0, 1.1],
            ],
            dtype=torch.float32,
        )
        ref = obb_iou_python(a, b)
        fast = obb_iou(a.cuda(), b.cuda()).cpu()
        self.assertTrue(torch.allclose(ref, fast, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
