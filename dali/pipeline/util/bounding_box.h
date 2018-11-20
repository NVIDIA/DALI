//
// Created by pribalta on 19.11.18.
//

#ifndef DALI_BOUNDING_BOX_H
#define DALI_BOUNDING_BOX_H

#include "dali/error_handling.h"

namespace dali {

class BoundingBox {
 public:
  const static size_t kBBoxSize = 4;

  BoundingBox(BoundingBox& other)
      : left_{other.left_},
        top_{other.top_},
        right_{other.right_},
        bottom_{other.bottom_},
        area_{other.area_} {}

  BoundingBox(BoundingBox&& other) noexcept
      : left_{other.left_},
        top_{other.top_},
        right_{other.right_},
        bottom_{other.bottom_},
        area_{other.area_} {}

  static BoundingBox FromLtrb(float l, float t, float r, float b) {
    DALI_ENFORCE(l >= 0 && l < 1.f,
                 "Expected 0 <= left <= 1. Received: " + to_string(l));
    DALI_ENFORCE(t >= 0 && t < 1.f,
                 "Expected 0 <= top <= 1. Received: " + to_string(t));
    DALI_ENFORCE(r >= 0 && r < 1.f,
                 "Expected 0 <= right <= 1. Received: " + to_string(r));
    DALI_ENFORCE(b >= 0 && b < 1.f,
                 "Expected 0 <= bottom <= 1. Received: " + to_string(b));

    DALI_ENFORCE(l <= r, "Expected left <= right. Received: " + to_string(l) +
                             " <= " + to_string(r));
    DALI_ENFORCE(t <= b, "Expected top <= bottom. Received: " + to_string(t) +
                             " <= " + to_string(b));

    return {l, t, r, b};
  }
  static BoundingBox FromXywh(float x, float y, float w, float h) {
    DALI_ENFORCE(x >= 0 && x <= 1.f,
                 "Expected 0 <= x <= 1. Received: " + to_string(x));
    DALI_ENFORCE(y >= 0 && y <= 1.f,
                 "Expected 0 <= y <= 1. Received: " + to_string(y));
    DALI_ENFORCE(w >= 0 && w <= 1.f,
                 "Expected 0 <= width <= 1. Received: " + to_string(w));
    DALI_ENFORCE(h >= 0 && h <= 1.f,
                 "Expected 0 <= height <= 1. Received: " + to_string(h));

    DALI_ENFORCE(x + w <= 1, "Expected x + width <= 1. Received: " +
                                 to_string(x) + " + " + to_string(w));
    DALI_ENFORCE(y + h <= 1, "Expected y + height <= 1. Received: " +
                                 to_string(y) + " + " + to_string(h));

    return {x, y, x + w, y + h};
  }

  BoundingBox ClampTo(const BoundingBox& other) const {
    return {std::max(other.left_, left_), std::max(other.top_, top_),
            std::min(other.right_, right_), std::min(other.bottom_, bottom_)};
  }

  float Area() const { return area_; }

  bool Contains(float x, float y) const {
    return x >= left_ && x <= right_ && y >= top_ && y <= bottom_;
  }

  bool Overlaps(const BoundingBox& other) const {
    return left_ < other.right_ && right_ > other.left_ &&
           top_ < other.bottom_ && bottom_ > other.top_;
  }

  float IntersectionOverUnion(const BoundingBox& other) const {
    if (this->Overlaps(other)) {
      const float intersection_area = this->ClampTo(other).area_;

      return intersection_area / (area_ + other.area_ - intersection_area);
    }
    return 0.0f;
  }

  BoundingBox HorizontalFlip() const {
    return {1 - right_, top_, right_, bottom_};
  }

  BoundingBox VerticalFlip() const {
    return {left_, 1 - bottom_, right_, bottom_};
  }

  std::array<float, kBBoxSize> AsLtrb() const {
    return {left_, top_, right_, bottom_};
  }

  std::array<float, kBBoxSize> AsXywh() const {
    return {left_, top_, right_ - left_, bottom_ - top_};
  }

 private:
  BoundingBox(float l, float t, float r, float b)
      : left_{l}, top_{t}, right_{r}, bottom_{b}, area_{(r - l) * (b - t)} {}

  const float left_, top_, right_, bottom_, area_;
};

}  // namespace dali

#endif  // DALI_BOUNDING_BOX_H
