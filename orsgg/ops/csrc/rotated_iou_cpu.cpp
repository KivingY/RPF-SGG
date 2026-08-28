#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct Point {
  double x;
  double y;
};

std::vector<Point> sort_quad(const float* box) {
  std::vector<Point> pts(4);
  double cx = 0.0;
  double cy = 0.0;
  for (int i = 0; i < 4; ++i) {
    pts[i] = Point{static_cast<double>(box[2 * i]), static_cast<double>(box[2 * i + 1])};
    cx += pts[i].x;
    cy += pts[i].y;
  }
  cx /= 4.0;
  cy /= 4.0;
  std::sort(pts.begin(), pts.end(), [cx, cy](const Point& a, const Point& b) {
    return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
  });
  return pts;
}

double polygon_area(const std::vector<Point>& poly) {
  const int n = static_cast<int>(poly.size());
  if (n < 3) {
    return 0.0;
  }
  double area = 0.0;
  for (int i = 0; i < n; ++i) {
    const Point& p = poly[i];
    const Point& q = poly[(i + 1) % n];
    area += p.x * q.y - q.x * p.y;
  }
  return std::abs(area) * 0.5;
}

inline bool inside(const Point& p, const Point& a, const Point& b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= -1e-9;
}

Point line_intersection(const Point& p1, const Point& p2, const Point& q1, const Point& q2) {
  const double x1 = p1.x;
  const double y1 = p1.y;
  const double x2 = p2.x;
  const double y2 = p2.y;
  const double x3 = q1.x;
  const double y3 = q1.y;
  const double x4 = q2.x;
  const double y4 = q2.y;
  const double den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (std::abs(den) < 1e-12) {
    return p2;
  }
  const double cross1 = x1 * y2 - y1 * x2;
  const double cross2 = x3 * y4 - y3 * x4;
  return Point{
      (cross1 * (x3 - x4) - (x1 - x2) * cross2) / den,
      (cross1 * (y3 - y4) - (y1 - y2) * cross2) / den};
}

std::vector<Point> clip_polygon(std::vector<Point> subject, const std::vector<Point>& clip) {
  std::vector<Point> output = std::move(subject);
  for (int i = 0; i < static_cast<int>(clip.size()); ++i) {
    const Point a = clip[i];
    const Point b = clip[(i + 1) % clip.size()];
    const std::vector<Point> input = output;
    output.clear();
    if (input.empty()) {
      break;
    }
    Point s = input.back();
    for (const Point& e : input) {
      if (inside(e, a, b)) {
        if (!inside(s, a, b)) {
          output.push_back(line_intersection(s, e, a, b));
        }
        output.push_back(e);
      } else if (inside(s, a, b)) {
        output.push_back(line_intersection(s, e, a, b));
      }
      s = e;
    }
  }
  return output;
}

double quad_iou(const float* box1, const float* box2) {
  const std::vector<Point> poly1 = sort_quad(box1);
  const std::vector<Point> poly2 = sort_quad(box2);
  const double area1 = polygon_area(poly1);
  const double area2 = polygon_area(poly2);
  if (area1 <= 0.0 || area2 <= 0.0) {
    return 0.0;
  }
  const std::vector<Point> inter_poly = clip_polygon(poly1, poly2);
  const double inter = polygon_area(inter_poly);
  const double uni = area1 + area2 - inter;
  if (uni <= 0.0) {
    return 0.0;
  }
  return std::max(0.0, std::min(1.0, inter / uni));
}

}  // namespace

torch::Tensor obb_iou_cpu(torch::Tensor boxes1, torch::Tensor boxes2) {
  const int64_t n = boxes1.size(0);
  const int64_t m = boxes2.size(0);
  auto out = torch::empty({n, m}, boxes1.options());
  const float* b1 = boxes1.data_ptr<float>();
  const float* b2 = boxes2.data_ptr<float>();
  float* o = out.data_ptr<float>();
  at::parallel_for(0, n * m, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t i = idx / m;
      const int64_t j = idx % m;
      o[idx] = static_cast<float>(quad_iou(b1 + i * 8, b2 + j * 8));
    }
  });
  return out;
}
