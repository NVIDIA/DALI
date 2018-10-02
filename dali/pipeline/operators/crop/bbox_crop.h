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
        : min_(bounds[0]), max_(bounds[1]) {
      DALI_ENFORCE(
          min_ >= 0 && min_ <= 1.0,
          "Min should be in [0.0-1.0]. Received: " + std::to_string(min_));
      DALI_ENFORCE(
          max_ >= 0 && max_ <= 1.0,
          "Max should be in [0.0-1.0]. Received: " + std::to_string(max_));
      DALI_ENFORCE(bounds.size() == 2, "Bounds should be provided as 2 values");
      DALI_ENFORCE(min_ <= max_, "Bounds should be provided as: [min, max]");
    }

    bool Contains(float k) const { return k >= min_ && k <= max_; }

    const float min_, max_;
  };

  struct Rectangle {
    explicit Rectangle(float left, float top, float right, float bottom)
        : left_(left),
          top_(top),
          right_(right),
          bottom_(bottom),
          area_(right * bottom) {}

    bool Contains(float x, float y) const {
      return x >= left_ && (x - left_) <= right_ && y >= top_ &&
             (y - top_) <= bottom_;
    }

    float IntersectionOverUnion(const Rectangle &other) const {
      const float x_left = std::max(left_, other.left_);
      const float x_right = std::min(right_, other.right_);
      const float y_top = std::max(top_, other.top_);
      const float y_bottom = std::min(bottom_, other.bottom_);

      const float intersection_area =
          std::max(0.f, (x_right - x_left) * (y_top - y_bottom));

      return intersection_area /
             static_cast<float>(area_ + other.area_ - intersection_area);
    }

    const float left_, top_, right_, bottom_, area_;
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
            Bounds(spec.GetRepeatedArgument<float>("aspect_ratio"))}

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

    WriteCropToOutput(ws, prospective_crop.first);
    WriteBoxesToOutput(ws, prospective_crop.second);
  }

  void WriteCropToOutput(SampleWorkspace *ws, const Crop &crop) {
    // Copy the anchor to output 0
    auto *anchor_out = ws->Output<CPUBackend>(0);
    anchor_out->Resize({2});

    auto *anchor_out_data = anchor_out->mutable_data<float>();
    anchor_out_data[0] = crop.left_;
    anchor_out_data[1] = crop.top_;

    // Copy the offsets to output 1
    auto *offsets_out = ws->Output<CPUBackend>(1);
    offsets_out->Resize({2});

    auto *offsets_out_data = offsets_out->mutable_data<float>();
    offsets_out_data[0] = crop.right_ - crop.left_;
    offsets_out_data[1] = crop.bottom_ - crop.top_;
  }

  void WriteBoxesToOutput(SampleWorkspace *ws,
                          const BoundingBoxes &bounding_boxes) {
    auto *bbox_out = ws->Output<CPUBackend>(2);
    bbox_out->Resize({static_cast<int>(bounding_boxes.size()), kBboxSize});

    auto *bbox_out_data = bbox_out->mutable_data<float>();
    for (size_t i = 0; i < bounding_boxes.size(); ++i) {
      auto *output = bbox_out_data + i * kBboxSize;
      output[0] = bounding_boxes[i].left_;
      output[1] = bounding_boxes[i].top_;
      output[2] = bounding_boxes[i].right_;
      output[3] = bounding_boxes[i].bottom_;
    }
  }

  float SelectMinimumOverlap() {
    static std::uniform_int_distribution<> sampler(0, thresholds_.size());
    return thresholds_[sampler(rd_)];
  }

  float Rescale(unsigned int k) {
    static std::uniform_real_distribution<> sampler(scaling_bounds_.min_,
                                                    scaling_bounds_.max_);
    return sampler(rd_) * k;
  }

  float ValidAspectRatio(float width, float height) const {
    return aspect_ratio_bounds_.Contains(width / height);
  }

  Rectangle SamplePatch(float scaled_height, float scaled_width, float height,
                        float width) {
    std::uniform_real_distribution<float> width_sampler(0.,
                                                        width - scaled_width);
    std::uniform_real_distribution<float> height_sampler(
        0., height - scaled_height);

    const auto left_offset = width_sampler(rd_);
    const auto height_offset = height_sampler(rd_);

    // Crop is LEFT/TOP/RIGHT/BOTTOM
    return Crop(left_offset, height_offset, scaled_width, scaled_height);
  }

  std::vector<Rectangle> DiscardBoundingBoxesByCentroid(
      const Crop &crop, const Tensor<CPUBackend> &bounding_boxes) {
    BoundingBoxes result;
    result.reserve(bounding_boxes.dim(0));

    // Discard bboxes whose centroid is not in the cropped area
    for (int i = 0; i < bounding_boxes.dim(0); ++i) {
      const auto *box = bounding_boxes.data<float>() + (i * kBboxSize);

      const float x_center = 0.5 * box[2] + box[0];
      const float y_center = 0.5 * box[3] + box[1];

      if (crop.Contains(x_center, y_center)) {
        result.emplace_back(BoundingBox(box[0], box[1], box[2], box[3]));
      }
    }

    return result;
  }

  BoundingBoxes ClampBoundingBoxes(const Crop &crop,
                                   const BoundingBoxes &boxes) {
    BoundingBoxes clamped_boxes;
    clamped_boxes.reserve(boxes.size());

    const auto crop_x_right = crop.left_ + crop.right_;
    const auto crop_x_bottom = crop.top_ + crop.bottom_;

    for (const auto &box : boxes) {
      const auto left = std::max(crop.left_, box.left_);
      const auto top = std::max(crop.top_, box.top_);

      const auto box_x_right = box.left_ + box.right_;
      const auto box_x_bottom = box.top_ + box.bottom_;

      const auto right = std::min(crop_x_right, box_x_right) - left;
      const auto bottom = std::min(crop_x_bottom, box_x_bottom) - top;

      clamped_boxes.emplace_back(BoundingBox(left, top, right, bottom));
    }

    return clamped_boxes;
  }

  std::pair<Crop, BoundingBoxes> FindProspectiveCrop(
      const Tensor<CPUBackend> &image, const Tensor<CPUBackend> &bounding_boxes,
      float minimum_overlap) {
    if (minimum_overlap > 0.0) {
      for (size_t i = 0; i < kAttempts; ++i) {
        // Image is HWC
        const auto rescaled_height = Rescale(image.dim(0));
        const auto rescaled_width = Rescale(image.dim(1));

        if (ValidAspectRatio(rescaled_height, rescaled_width)) {
          const auto candidate_crop = SamplePatch(
              rescaled_height, rescaled_width, image.dim(0), image.dim(1));

          const auto candidate_boxes =
              DiscardBoundingBoxesByCentroid(candidate_crop, bounding_boxes);

          if (std::all_of(
                  candidate_boxes.begin(), candidate_boxes.end(),
                  [&minimum_overlap, &candidate_crop](const BoundingBox &box) {
                    return candidate_crop.IntersectionOverUnion(box) >=
                           minimum_overlap;
                  })) {
            return std::make_pair(
                candidate_crop,
                ClampBoundingBoxes(candidate_crop, candidate_boxes));
          }
        }
      }
    }

    // If overlap is 0.0 or fallback if we run out of attempts
    BoundingBoxes result;
    result.reserve(bounding_boxes.dim(0));

    for (int i = 0; i < bounding_boxes.dim(0); ++i) {
      const auto *box = bounding_boxes.data<float>() + (i * kBboxSize);

      result.emplace_back(BoundingBox(box[0], box[1], box[2], box[3]));
    }

    return std::make_pair(Crop(0, 0, image.dim(1), image.dim(0)), result);
  }

  const std::vector<float> thresholds_;
  const Bounds scaling_bounds_;
  const Bounds aspect_ratio_bounds_;

 private:
  std::random_device rd_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
