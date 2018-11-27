// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.c++ copyswa

#ifndef DALI_PIPELINE_UTIL_BOUNDING_BOX_H_
#define DALI_PIPELINE_UTIL_BOUNDING_BOX_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "dali/error_handling.h"

namespace dali {

class BoundingBox {
 public:
  static const size_t kSize = 4;

  BoundingBox(const BoundingBox& other)
      : left_{other.left_},
        top_{other.top_},
        right_{other.right_},
        bottom_{other.bottom_},
        area_{other.area_} {}

  BoundingBox(BoundingBox&& other) noexcept : BoundingBox() { swap(*this, other); }

  BoundingBox& operator=(BoundingBox other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(BoundingBox& lhs, BoundingBox& rhs) {
    using std::swap;
    swap(lhs.left_, rhs.left_);
    swap(lhs.right_, rhs.right_);
    swap(lhs.top_, rhs.top_);
    swap(lhs.bottom_, rhs.bottom_);
    swap(lhs.area_, rhs.area_);
  }

  static BoundingBox FromLtrb(const float* data, bool check_input = true) {
    return FromLtrb(data[0], data[1], data[2], data[3], check_input);
  }

  static BoundingBox FromLtrb(float l, float t, float r, float b, bool check_input = true) {
    if (check_input) {
      DALI_ENFORCE(l >= 0 && l <= 1.f,
                  "Expected 0 <= left <= 1. Received: " + to_string(l));
      DALI_ENFORCE(t >= 0 && t <= 1.f,
                  "Expected 0 <= top <= 1. Received: " + to_string(t));
      DALI_ENFORCE(r >= 0 && r <= 1.f,
                  "Expected 0 <= right <= 1. Received: " + to_string(r));
      DALI_ENFORCE(b >= 0 && b <= 1.f,
                  "Expected 0 <= bottom <= 1. Received: " + to_string(b));

      DALI_ENFORCE(l <= r, "Expected left <= right. Received: " + to_string(l) +
                              " <= " + to_string(r));
      DALI_ENFORCE(t <= b, "Expected top <= bottom. Received: " + to_string(t) +
                              " <= " + to_string(b));
    }

    return {l, t, r, b};
  }

  static BoundingBox FromXywh(const float* data) {
    return FromXywh(data[0], data[1], data[2], data[3]);
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

  BoundingBox RemapTo(const BoundingBox& other) const {
    const float crop_width = other.right_ - other.left_;
    const float crop_height = other.bottom_ - other.top_;

    const float new_left =
        (std::max(other.left_, left_) - other.left_) / crop_width;
    const float new_top =
        (std::max(other.top_, top_) - other.top_) / crop_height;
    const float new_right =
        (std::min(other.right_, right_) - other.left_) / crop_width;
    const float new_bottom =
        (std::min(other.bottom_, bottom_) - other.top_) / crop_height;

    return {std::max(0.0f, std::min(new_left, 1.0f)),
            std::max(0.0f, std::min(new_top, 1.0f)),
            std::max(0.0f, std::min(new_right, 1.0f)),
            std::max(0.0f, std::min(new_bottom, 1.0f))};
  }

  BoundingBox HorizontalFlip() const {
    return {1 - right_, top_, 1 - left_, bottom_};
  }

  BoundingBox VerticalFlip() const {
    return {left_, 1 - bottom_, right_, 1 - top_};
  }

  std::array<float, kSize> AsLtrb() const {
    return {left_, top_, right_, bottom_};
  }

  std::array<float, kSize> AsXywh() const {
    return {left_, top_, right_ - left_, bottom_ - top_};
  }

  void CopyAsCenterWh(float *destination) const {
    destination[0] = (left_ + right_) * 0.5f;
    destination[1] = (top_ + bottom_) * 0.5f;
    destination[2] = right_ - left_;
    destination[3] = bottom_ - top_;
  }

  static vector<BoundingBox> FromLtrbArray(const float *source, int num, bool check_input = true) {
    vector<BoundingBox> boxes;
    auto elem = source;

    for (int idx = 0; idx < num; ++idx) {
      boxes.push_back(FromLtrb(elem, check_input));
      elem += kSize;
    }

    return boxes;
  }

 private:
  BoundingBox() = default;

  BoundingBox(float l, float t, float r, float b)
      : left_{l}, top_{t}, right_{r}, bottom_{b}, area_{(r - l) * (b - t)} {}

  float left_, top_, right_, bottom_, area_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BOUNDING_BOX_H_
