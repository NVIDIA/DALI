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

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/fused/crop_cast_permute.h"

namespace dali {

template <typename Backend>
class BBoxCrop : public Operator<CPUBackend> {
  using Overlaps = std::vector<float>;
  using BBox = Rectangle;
  using BBoxes = std::vector<Bbox>;
  using Crop = Rectangle;

 struct Rectangle
 {
    explicit Rectangle()
     : left_(-1)
     , top_(-1)
     , right_(-1)
     , bottom_(-1)
   {
   }

   explicit Rectangle(float left,
                 float top,
                 float right,
                 float bottom)
     : left_(left)
     , top_(top)
     , right_(right)
     , bottom_(bottom)
   {
   }

   bool Contains(float x, float y)
   {
     return x_center >= crop.left_ 
            && x_center =< crop.right_ 
            && y_center >= crop.top_ 
            && y_center =< crop._bottom;
   }

   const float left_, top_, right_, bottom_;
  };

  const static bbox_size = 4;

 public:
  explicit inline BBoxCrop(const OpSpec &spec)
    : num_attempts_(spec.GetArgument<int>("num_attempts")),
    , thresholds_({0.1, 0.3, 0.5, 0.7, 0.9})
  {
    //TODO: @pribalta Pass thresholds as argument?
    //TODO: @pribalta Pass the W/H scaling bounds?
    //TODO: @pribalta Pass the aspect ratio bounds?

  }

  virtual ~BBoxCrop() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override
  {
    const auto& image = ws->Input<CPUBackend>(0);
    const auto& bboxes = ws->Input<CPUBackend>(1);

    const auto minimum_overlap = SelectMinimumOverlap();

    if (minimum_overlap > 0.0)
    {
      const auto crop_candidate = FindCropCandidate(image, bboxes, overlap);

      // TODO what if no candidate is found?

      // TODO Write crop parameters    

      WriteBBoxesToOutput(ws, crop_candidate.second);
    }
    else
    {
      ws->Output<CPUBackend>(0)->Copy(images, 0);
      ws->Output<CPUBackend>(1)->Copy(bboxes, 0);
    }
  }

 private:
  void WriteBBoxesToOutput(Workspace<Backend> *ws, const BBoxes& valid_bboxes)
  {
    auto *bbox_out = ws->Output<CPUBackend>(1);

    bbox_out->Resize({crop_candidate.second.size(), bbox_size});
    auto *bbox_out_data = bbox_out->mutable_data<float>();

    for (int i = 0; i < valid_bboxes.size(); ++i)
    {
      auto *output = bbox_out_data + i * bbox_size;
      output[0] = valid_bboxes[i].left_; 
      output[1] = valid_bboxes[i].top_; 
      output[2] = valid_bboxes[i].right_; 
      output[3] = valid_bboxes[i].bottom_; 
    }
  }

  Bboxes FilterBboxByCentroid(const Crop& crop, const Tensor<Backend>& bboxes)
  {
    Bboxes result;
    result.reserve(bboxes.dim(0));

    // Discard bboxes whose centroid is not in the cropped area
    for (int i = 0; i < bboxes.dim(0); ++i)
    {
      const auto* bbox = bboxes.data<float>() + (i * bbox_size);

      const float x_center = 0.5 * (bbox[0] + bbox[2]);
      const float y_center = 0.5 * (bbox[1] + bbox[3]);

      if (crop.Contains(x_center, y_center)
      {
        result.emplace_back(BBox(bbox[0],
                                 bbox[1],
                                 bbox[2],
                                 bbox[3]));
      }

      return result;
    }
  }

  Crop CropPatch(float width, float height) const
  {
    std::uniform_real_distribution<float> width_sampler(0., 1. - width),
                                          height_sampler(0., 1. - height);

    const auto left_offset = width_sampler(rd_);
    const auto height_offset = height_sampler(rd_);

    // Crop is LEFT/TOP/RIGHT/BOTTOM
    return Crop(left_offset,
                height_offset,
                left_offset + width,
                height_offset + height);
  }

  float ValidAspectRatio(float width, float height) const
  {
    const float aspect_ratio = width / height;

    return aspect_ratio >= 0.5 && aspect_ratio <= 2;
  }

  float SelectMinimumOverlap() const
  {
    static std::uniform_int_distribution<> sampler(0, thresholds_.size());

    return thresholds_(sampler(rd_));
  }

  float Rescale(unsigned int k) const
  {
    static std::uniform_real_distribution<> sampler(0.3, 1.0);

    return sampler(rd_) * k;
  }

  float CalculateOverlap(const Crop& crop_candidate, const BBox& bbox)
  {
    // TODO
    return 0.0;
  }

  bool ValidOverlap(const Crop& crop_candidate, const BBoxes& bboxes, float minimum_overlap)
  {
    for (const auto& bbox : bboxes)
    {
      if (CalculateOverlap(crop_candidate, bbox) <= minimum_overlap)
      {
        return false;
      }
    }

    return true;
  }

  std::pair<Crop, BBoxes> FindCropCandidate(const Tensor<Backend>& image,
                        const Tensor<Backend>& bboxes,
                        float minimum_overlap)
  {
    for (int i = 0; i < num_attempts_; ++i)
    {
      // Input is HWC
      const auto target_height = Rescale(image.dim(0));
      const auto target_width = Rescale(image.dim(1));
      
      if (ValidAspectRatio(target_width, target_height))
      {
        const auto candidate_crop = CropPatch(target_width, target_height);

        auto candidate_bboxes = FilterBboxByCentroid(candidate_crop, bboxes);

        // TODO: Need to rescale and clamp the boxes

        if (ValidOverlap(crop_candidate, candidate_bboxes, minimum_overlap))
        {
          return std::make_pair(crop_candidate, candidate_bboxes);
        }
      }
    }

    return std::make_pair({}, {});
  }

  const unsigned int num_attemtps_;
  const std::vector<float> thresholds_;
  std::random_device rd_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_

