#include <torch/extension.h>

torch::Tensor obb_iou_cpu(torch::Tensor boxes1, torch::Tensor boxes2);

#ifdef WITH_CUDA
torch::Tensor obb_iou_cuda(torch::Tensor boxes1, torch::Tensor boxes2);
#endif

torch::Tensor obb_iou(torch::Tensor boxes1, torch::Tensor boxes2) {
  TORCH_CHECK(boxes1.dim() == 2 && boxes1.size(1) == 8, "boxes1 must be [N,8]");
  TORCH_CHECK(boxes2.dim() == 2 && boxes2.size(1) == 8, "boxes2 must be [M,8]");
  TORCH_CHECK(boxes1.device() == boxes2.device(), "boxes must be on the same device");
  TORCH_CHECK(boxes1.scalar_type() == torch::kFloat32, "boxes1 must be float32");
  TORCH_CHECK(boxes2.scalar_type() == torch::kFloat32, "boxes2 must be float32");
  if (boxes1.is_cuda()) {
#ifdef WITH_CUDA
    return obb_iou_cuda(boxes1.contiguous(), boxes2.contiguous());
#else
    TORCH_CHECK(false, "CUDA rotated IoU extension was not built");
#endif
  }
  return obb_iou_cpu(boxes1.contiguous(), boxes2.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("obb_iou", &obb_iou, "Pairwise OBB IoU");
}
