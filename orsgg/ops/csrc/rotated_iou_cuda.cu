#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>

namespace {

struct Point {
  float x;
  float y;
};

__device__ void sort_quad(const float* box, Point* pts) {
  float cx = 0.0f;
  float cy = 0.0f;
  for (int i = 0; i < 4; ++i) {
    pts[i].x = box[2 * i];
    pts[i].y = box[2 * i + 1];
    cx += pts[i].x;
    cy += pts[i].y;
  }
  cx *= 0.25f;
  cy *= 0.25f;
  float angles[4];
  for (int i = 0; i < 4; ++i) {
    angles[i] = atan2f(pts[i].y - cy, pts[i].x - cx);
  }
  for (int i = 1; i < 4; ++i) {
    Point key = pts[i];
    float key_angle = angles[i];
    int j = i - 1;
    while (j >= 0 && angles[j] > key_angle) {
      pts[j + 1] = pts[j];
      angles[j + 1] = angles[j];
      --j;
    }
    pts[j + 1] = key;
    angles[j + 1] = key_angle;
  }
}

__device__ float polygon_area(const Point* poly, int n) {
  if (n < 3) {
    return 0.0f;
  }
  float area = 0.0f;
  for (int i = 0; i < n; ++i) {
    const Point p = poly[i];
    const Point q = poly[(i + 1) % n];
    area += p.x * q.y - q.x * p.y;
  }
  return fabsf(area) * 0.5f;
}

__device__ bool inside(const Point& p, const Point& a, const Point& b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= -1e-6f;
}

__device__ Point line_intersection(
    const Point& p1,
    const Point& p2,
    const Point& q1,
    const Point& q2) {
  const float x1 = p1.x;
  const float y1 = p1.y;
  const float x2 = p2.x;
  const float y2 = p2.y;
  const float x3 = q1.x;
  const float y3 = q1.y;
  const float x4 = q2.x;
  const float y4 = q2.y;
  const float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (fabsf(den) < 1e-12f) {
    return p2;
  }
  const float cross1 = x1 * y2 - y1 * x2;
  const float cross2 = x3 * y4 - y3 * x4;
  return Point{
      (cross1 * (x3 - x4) - (x1 - x2) * cross2) / den,
      (cross1 * (y3 - y4) - (y1 - y2) * cross2) / den};
}

__device__ int clip_polygon(const Point* subject, int subject_n, const Point* clip, Point* output) {
  Point input_a[8];
  Point input_b[8];
  int input_n = subject_n;
  for (int i = 0; i < subject_n; ++i) {
    input_a[i] = subject[i];
  }
  Point* input = input_a;
  Point* temp = input_b;
  for (int edge = 0; edge < 4; ++edge) {
    const Point a = clip[edge];
    const Point b = clip[(edge + 1) & 3];
    int out_n = 0;
    if (input_n <= 0) {
      return 0;
    }
    Point s = input[input_n - 1];
    for (int i = 0; i < input_n; ++i) {
      const Point e = input[i];
      const bool e_inside = inside(e, a, b);
      const bool s_inside = inside(s, a, b);
      if (e_inside) {
        if (!s_inside && out_n < 8) {
          temp[out_n++] = line_intersection(s, e, a, b);
        }
        if (out_n < 8) {
          temp[out_n++] = e;
        }
      } else if (s_inside && out_n < 8) {
        temp[out_n++] = line_intersection(s, e, a, b);
      }
      s = e;
    }
    Point* swap_ptr = input;
    input = temp;
    temp = swap_ptr;
    input_n = out_n;
  }
  for (int i = 0; i < input_n; ++i) {
    output[i] = input[i];
  }
  return input_n;
}

__device__ float quad_iou(const float* box1, const float* box2) {
  Point poly1[4];
  Point poly2[4];
  sort_quad(box1, poly1);
  sort_quad(box2, poly2);
  const float area1 = polygon_area(poly1, 4);
  const float area2 = polygon_area(poly2, 4);
  if (area1 <= 0.0f || area2 <= 0.0f) {
    return 0.0f;
  }
  Point inter_poly[8];
  const int inter_n = clip_polygon(poly1, 4, poly2, inter_poly);
  const float inter = polygon_area(inter_poly, inter_n);
  const float uni = area1 + area2 - inter;
  if (uni <= 0.0f) {
    return 0.0f;
  }
  return fminf(1.0f, fmaxf(0.0f, inter / uni));
}

__global__ void obb_iou_kernel(const float* boxes1, const float* boxes2, float* out, int n, int m) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * m;
  if (idx >= total) {
    return;
  }
  const int i = idx / m;
  const int j = idx - i * m;
  out[idx] = quad_iou(boxes1 + i * 8, boxes2 + j * 8);
}

}  // namespace

torch::Tensor obb_iou_cuda(torch::Tensor boxes1, torch::Tensor boxes2) {
  const int n = static_cast<int>(boxes1.size(0));
  const int m = static_cast<int>(boxes2.size(0));
  auto out = torch::empty({n, m}, boxes1.options());
  const int total = n * m;
  if (total == 0) {
    return out;
  }
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  obb_iou_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      boxes1.data_ptr<float>(), boxes2.data_ptr<float>(), out.data_ptr<float>(), n, m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
