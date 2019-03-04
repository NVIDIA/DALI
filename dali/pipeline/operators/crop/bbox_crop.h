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

#ifndef DALI_PIPELINE_OPERATORS_CROP_BBOX_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_BBOX_CROP_H_

#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <tuple>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/util/bounding_box.h"

namespace dali {

template <typename Backend>
class RandomBBoxCrop : public Operator<Backend> {
 protected:
  struct Bounds {
    explicit Bounds(const std::vector<float> &bounds)
        : min(!bounds.empty() ? bounds[0] : -1),
          max(bounds.size() > 1 ? bounds[1] : -1) {
      DALI_ENFORCE(bounds.size() == 2, "Bounds should be provided as 2 values");
      DALI_ENFORCE(min >= 0, "Min should be at least 0.0. Received: " +
                                 std::to_string(min));
      DALI_ENFORCE(min <= max, "Bounds should be provided as: [min, max]");
    }

    bool Contains(float k) const { return k >= min && k <= max; }

    const float min, max;
  };

  using Crop = BoundingBox;
  using BoundingBoxes = std::vector<BoundingBox>;

  struct ProspectiveCrop {
    bool success = false;
    Crop crop = Crop::FromLtrb(0, 0, 1, 1);
    BoundingBoxes boxes;
    std::vector<int> labels;

    ProspectiveCrop(
      bool success, const Crop &crop, const BoundingBoxes &boxes, const std::vector<int> &labels) :
      success(success), crop(crop), boxes(boxes), labels(labels) {}
    ProspectiveCrop() = default;
  };

 public:
  explicit inline RandomBBoxCrop(const OpSpec &spec)
      : Operator<Backend>(spec),
        scaling_bounds_{Bounds(spec.GetRepeatedArgument<float>("scaling"))},
        aspect_ratio_bounds_{
            Bounds(spec.GetRepeatedArgument<float>("aspect_ratio"))},
        ltrb_{spec.GetArgument<bool>("ltrb")},
        num_attempts_{spec.GetArgument<int>("num_attempts")},
        gen_(spec.GetArgument<int64_t>("seed")) {
    auto thresholds = spec.GetRepeatedArgument<float>("thresholds");

    DALI_ENFORCE(!thresholds.empty(),
                 "At least one threshold value must be provided");

    for (const auto &threshold : thresholds) {
      DALI_ENFORCE(0.0 <= threshold,
                   "Threshold value must be >= 0.0. Received: " +
                       std::to_string(threshold));
      DALI_ENFORCE(threshold <= 1.0,
                   "Threshold value must be <= 1.0. Received: " +
                       std::to_string(threshold));
      DALI_ENFORCE(num_attempts_ > 0,
                   "Minimum number of attempts must be greater than zero");

      sample_options_.push_back(std::make_pair(threshold, true));
    }

    if (spec.GetArgument<bool>("allow_no_crop")) {
      sample_options_.push_back(std::make_pair(0.f, false));
    }
  }

  ~RandomBBoxCrop() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void WriteCropToOutput(SampleWorkspace *ws, const Crop &crop) const;

  void WriteBoxesToOutput(
    SampleWorkspace *ws, const BoundingBoxes &bounding_boxes) const;

  void WriteLabelsToOutput(SampleWorkspace *ws, const std::vector<int> &labels) const;

  const std::pair<float, bool> SelectMinimumOverlap() {
    std::uniform_int_distribution<> sampler(
      0, static_cast<int>(sample_options_.size() - 1));
    return sample_options_[sampler(gen_)];
  }

  const float SampleCandidateDimension() {
    std::uniform_real_distribution<float> sampler(scaling_bounds_.min, scaling_bounds_.max);
    return sampler(gen_);
  }

  const bool ValidAspectRatio(float width, float height) const {
    return aspect_ratio_bounds_.Contains(width / height);
  }

  const bool ValidOverlap(const Crop &crop, const BoundingBoxes &boxes, float threshold) const {
    return std::all_of(boxes.begin(), boxes.end(),
                       [&crop, threshold](const BoundingBox &box) {
                         return crop.IntersectionOverUnion(box) >= threshold;
                       });
  }

  const BoundingBoxes RemapBoxes(const Crop &crop, const BoundingBoxes &boxes,
                           float height, float width) const {
    BoundingBoxes remapped_boxes;
    remapped_boxes.reserve(boxes.size());

    for (const auto &box : boxes) {
      remapped_boxes.emplace_back(box.RemapTo(crop));
    }

    return remapped_boxes;
  }

  const Crop SamplePatch(float scaled_height, float scaled_width) {
    std::uniform_real_distribution<float> width_sampler(
      static_cast<float>(0.), 1 - scaled_width);
    std::uniform_real_distribution<float> height_sampler(
      static_cast<float>(0.), 1 - scaled_height);

    const auto left_offset = width_sampler(gen_);
    const auto height_offset = height_sampler(gen_);

    return Crop::FromLtrb(
      left_offset,
      height_offset,
      (left_offset + scaled_width),
      (height_offset + scaled_height));
  }

  const std::pair<BoundingBoxes, std::vector<int>> DiscardBoundingBoxesByCentroid(
      const Crop &crop, const BoundingBoxes &bounding_boxes,
      const std::vector<int> &labels) const {
    DALI_ENFORCE(bounding_boxes.size() == labels.size(),
                 "Labels and bounding boxes should have the same length");
    BoundingBoxes candidate_boxes;
    candidate_boxes.reserve(bounding_boxes.size());

    std::vector<int> candidate_labels;
    candidate_labels.reserve(labels.size());

    // Discard bboxes whose centroid is not in the cropped area
    for (size_t i = 0; i < bounding_boxes.size(); i++) {
      auto coord = bounding_boxes[i].AsLtrb();
      const float x_center = 0.5f * (coord[2] - coord[0]) + coord[0];
      const float y_center = 0.5f * (coord[3] - coord[1]) + coord[1];

      if (crop.Contains(x_center, y_center)) {
        candidate_boxes.push_back(bounding_boxes[i]);
        candidate_labels.push_back(labels[i]);
      }
    }

    return std::make_pair(candidate_boxes, candidate_labels);
  }

  const ProspectiveCrop FindProspectiveCrop(
      const BoundingBoxes &bounding_boxes, const std::vector<int> &labels,
      std::pair<float, bool> minimum_overlap) {
    if (!minimum_overlap.second)
      return ProspectiveCrop(true, Crop::FromLtrb(0, 0, 1, 1), bounding_boxes, labels);

  std::uniform_real_distribution<float> float_dis_(scaling_bounds_.min, scaling_bounds_.max);
    for (int i = 0; i < num_attempts_; ++i) {
      // Image is HWC
      const auto candidate_width = SampleCandidateDimension();
      const auto candidate_height = SampleCandidateDimension();

      if (ValidAspectRatio(candidate_height, candidate_width)) {
        const auto candidate_crop =
            SamplePatch(candidate_height, candidate_width);

        if (ValidOverlap(candidate_crop, bounding_boxes, minimum_overlap.first)) {
          BoundingBoxes candidate_boxes;
          std::vector<int> candidate_labels;

          std::tie(candidate_boxes, candidate_labels) =
                DiscardBoundingBoxesByCentroid(candidate_crop, bounding_boxes, labels);

          if (candidate_boxes.begin() != candidate_boxes.end()) {
            const auto remapped_boxes = RemapBoxes(
              candidate_crop, candidate_boxes, candidate_height, candidate_width);
            return ProspectiveCrop(true, candidate_crop, remapped_boxes, candidate_labels);
          }
        }
      }
    }
    return ProspectiveCrop();
  }

  // float - threshold for IoU, bool - whether to apply crop (false means no cropp)
  std::vector<std::pair<float, bool>> sample_options_;
  const Bounds scaling_bounds_;
  const Bounds aspect_ratio_bounds_;
  const bool ltrb_;
  const int num_attempts_;

 private:
  std::mt19937 gen_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_BBOX_CROP_H_
