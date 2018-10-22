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
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_H_

#include <vector>
#include <utility>
#include <algorithm>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

class BBoxCrop : public Operator<CPUBackend> {
  static const unsigned int kAttempts = 100;
  static const unsigned int kBboxSize = 4;

 protected:
  struct Bounds {
    explicit Bounds(std::vector<float> &&bounds)
        : min(bounds[0]), max(bounds[1]) {
      DALI_ENFORCE(min >= 0, "Min should be at least 0.0. Received: " +
                                 std::to_string(min));
      DALI_ENFORCE(bounds.size() == 2, "Bounds should be provided as 2 values");
      DALI_ENFORCE(min <= max, "Bounds should be provided as: [min, max]");
    }

    bool Contains(float k) const { return k >= min && k <= max; }

    const float min, max;
  };

  struct Rectangle {
    explicit Rectangle(float left, float top, float right, float bottom)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          area((right- left) * (bottom - top)) {
            // Enforce ltrb
            DALI_ENFORCE(left >= 0 && left <= 1);
            DALI_ENFORCE(top >= 0 && top <= 1);
            DALI_ENFORCE(right >= 0 && right <= 1);
            DALI_ENFORCE(bottom >= 0 && bottom <= 1);
            DALI_ENFORCE(left <= right);
            DALI_ENFORCE(top <= bottom);
          }

    bool Contains(float x, float y) const {
      return x >= left && (x - left) <= right && y >= top &&
             (y - top) <= bottom;
    }

    Rectangle ClampTo(const Rectangle &other) const {
      const float new_left = std::max(other.left, left);
      const float new_top = std::max(other.top, top);
      const float new_right = std::min(other.right, right);
      const float new_bottom = std::min(other.bottom, bottom);

      return Rectangle(std::max(other.left, left), std::max(other.top, top),
                       std::min(other.right, right),
                       std::min(other.bottom, bottom));
    }

    Rectangle RemapTo(const Rectangle &other, unsigned int height, unsigned int width) const {
      const float crop_width = other.right - other.left;
      const float crop_height = other.bottom - other.top;

      const float new_left = (std::max(other.left, left) - other.left) / crop_width;
      const float new_top = (std::max(other.top, top) - other.top) / crop_height;
      const float new_right = (std::min(other.right, right) - other.left) / crop_width;
      const float new_bottom = (std::min(other.bottom, bottom) - other.top) / crop_height;

      return Rectangle(
          std::max(0.0f, std::min(new_left, 1.0f)), std::max(0.0f, std::min(new_top, 1.0f)),
          std::max(0.0f, std::min(new_right, 1.0f)), std::max(0.0f, std::min(new_bottom, 1.0f)));
    }

    float IntersectionOverUnion(const Rectangle &other) const {
      if (this->Overlaps(other)) {
        const float intersection_area = this->ClampTo(other).area;

        return intersection_area /
               static_cast<float>(area + other.area - intersection_area);
      }
      return 0.0f;
    }

    bool Overlaps(const Rectangle &other) const {
      return left < other.right && right > other.left && top < other.bottom &&
             bottom > other.top;
    }

    const float left, top, right, bottom, area;
  };

  using Crop = Rectangle;
  using BoundingBox = Rectangle;
  using BoundingBoxes = std::vector<Rectangle>;

 public:
  explicit inline BBoxCrop(const OpSpec &spec)
      : Operator<CPUBackend>(spec),
        thresholds_{spec.GetRepeatedArgument<float>("thresholds")},
        scaling_bounds_{Bounds(spec.GetRepeatedArgument<float>("scaling"))},
        aspect_ratio_bounds_{
            Bounds(spec.GetRepeatedArgument<float>("aspect_ratio"))},
        ltrb_{spec.GetArgument<bool>("ltrb")}

  {
    DALI_ENFORCE(!thresholds_.empty(),
                 "At least one threshold value must be provided");

    for (const auto &threshold : thresholds_) {
      DALI_ENFORCE(0.0 <= threshold,
                   "Threshold value must be >= 0.0. Received: " +
                       std::to_string(threshold));
      DALI_ENFORCE(threshold <= 1.0,
                   "Threshold value must be <= 1.0. Received: " +
                       std::to_string(threshold));
    }
  }

  virtual ~BBoxCrop() = default;

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) {
    const auto &image = ws->Input<CPUBackend>(0);
    const auto &bounding_boxes = ws->Input<CPUBackend>(1);

    const auto minimum_overlap = SelectMinimumOverlap();

    const auto prospective_crop =
        FindProspectiveCrop(image, bounding_boxes, minimum_overlap);

    WriteCropToOutput(ws, prospective_crop.first, image.dim(0), image.dim(1));
    WriteBoxesToOutput(ws, prospective_crop.second);
  }

  void WriteCropToOutput(SampleWorkspace *ws, const Crop &crop,
                         unsigned int height, unsigned int width) {
    // Copy the anchor to output 0
    auto *anchor_out = ws->Output<CPUBackend>(0);
    anchor_out->Resize({2});

    auto *anchor_out_data = anchor_out->mutable_data<float>();
    anchor_out_data[0] = crop.left * width;
    anchor_out_data[1] = crop.top * height;

    // Copy the offsets to output 1
    auto *offsets_out = ws->Output<CPUBackend>(1);
    offsets_out->Resize({2});

    auto *offsets_out_data = offsets_out->mutable_data<float>();
    offsets_out_data[0] = (crop.right - crop.left) * width;
    offsets_out_data[1] = (crop.bottom - crop.top) * height;
  }

  void WriteBoxesToOutput(SampleWorkspace *ws,
                          const BoundingBoxes &bounding_boxes) {
    auto *bbox_out = ws->Output<CPUBackend>(2);
    bbox_out->Resize({static_cast<Index>(bounding_boxes.size()), kBboxSize});

    auto *bbox_out_data = bbox_out->mutable_data<float>();
    for (size_t i = 0; i < bounding_boxes.size(); ++i) {
      auto *output = bbox_out_data + i * kBboxSize;
      output[0] = bounding_boxes[i].left;
      output[1] = bounding_boxes[i].top;
      output[2] = ltrb_ ? bounding_boxes[i].right
                        : bounding_boxes[i].right - bounding_boxes[i].left;
      output[3] = ltrb_ ? bounding_boxes[i].bottom
                        : bounding_boxes[i].bottom - bounding_boxes[i].top;
    }
  }

  float SelectMinimumOverlap() {
    static std::uniform_int_distribution<> sampler(0, thresholds_.size()-1);
    return thresholds_[sampler(rd_)];
  }

  float Rescale(unsigned int k) {
    std::uniform_real_distribution<> sampler(scaling_bounds_.min,
                                                    scaling_bounds_.max);
    return sampler(rd_) * k;
  }

  bool ValidAspectRatio(float width, float height) const {
    return aspect_ratio_bounds_.Contains(width / height);
  }

  bool ValidOverlap(const Crop& crop, const BoundingBoxes& boxes, float threshold) {
    return std::all_of(boxes.begin(), boxes.end(), [&crop, threshold](const BoundingBox& box){
      return crop.IntersectionOverUnion(box) >= threshold;
    });
  }

  BoundingBoxes RemapBoxes(const Crop& crop, const BoundingBoxes& boxes,
                           float height, float width) const {
    BoundingBoxes remapped_boxes;
    remapped_boxes.reserve(boxes.size());

    for (const auto& box : boxes) {
        remapped_boxes.emplace_back(
          box.RemapTo(crop, height, width));
    }

    return remapped_boxes;
  }

  Rectangle SamplePatch(float scaled_height, float scaled_width, float height,
                        float width) {
    std::uniform_real_distribution<float> width_sampler(0.,
                                                        width - scaled_width);
    std::uniform_real_distribution<float> height_sampler(
        0., height - scaled_height);

    const auto left_offset = width_sampler(rd_);
    const auto height_offset = height_sampler(rd_);

    // Crop is ltrb
    return Crop(left_offset / width,
                height_offset / height,
                (left_offset + scaled_width) / width,
                (height_offset + scaled_height) / height);
  }

  std::vector<Rectangle> DiscardBoundingBoxesByCentroid(
      const Crop &crop, const Tensor<CPUBackend> &bounding_boxes) {
    BoundingBoxes result;
    result.reserve(bounding_boxes.dim(0));

    // Discard bboxes whose centroid is not in the cropped area
    for (int i = 0; i < bounding_boxes.dim(0); ++i) {
      const auto *box = bounding_boxes.data<float>() + (i * kBboxSize);

      const float x_center = 0.5 * (box[2] - box[0]) + box[0];
      const float y_center = 0.5 * (box[3] - box[1]) + box[1];

      if (crop.Contains(x_center, y_center)) {
        result.emplace_back(box[0], box[1], box[2], box[3]);
      }
    }

    return result;
  }

  std::pair<Crop, BoundingBoxes> FindProspectiveCrop(
      const Tensor<CPUBackend> &image, const Tensor<CPUBackend> &bounding_boxes,
      float minimum_overlap) {
    if (minimum_overlap > 0) {
      for (size_t i = 0; i < kAttempts; ++i) {
        // Image is HWC
        const auto rescaled_height = Rescale(image.dim(0));
        const auto rescaled_width = Rescale(image.dim(1));

        if (ValidAspectRatio(rescaled_height, rescaled_width)) {
          const auto candidate_crop = SamplePatch(
              rescaled_height, rescaled_width, image.dim(0), image.dim(1));

          auto candidate_boxes =
              DiscardBoundingBoxesByCentroid(candidate_crop, bounding_boxes);

          if (ValidOverlap(candidate_crop, candidate_boxes, minimum_overlap)) {
            const auto remapped_boxes = RemapBoxes(candidate_crop, candidate_boxes,
                                                   rescaled_height, rescaled_width);
            return std::make_pair(candidate_crop, remapped_boxes);
          }
        }
      }
    }

    // If overlap is 0.0 or fallback if we run out of attempts
    BoundingBoxes result;
    result.reserve(bounding_boxes.dim(0));

    for (int i = 0; i < bounding_boxes.dim(0); ++i) {
      const auto *box = bounding_boxes.data<float>() + (i * kBboxSize);

      result.emplace_back(box[0], box[1], box[2], box[3]);
    }

    return std::make_pair(Crop(0, 0, 1, 1), result);
  }

  const std::vector<float> thresholds_;
  const Bounds scaling_bounds_;
  const Bounds aspect_ratio_bounds_;
  const bool ltrb_;

 private:
  std::random_device rd_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_BBOX_CROP_H_
