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

class BBoxCrop : public Operator<CPUBackend>
{ 
  const static unsigned int bbox_size_ = 4;

  struct Rectangle
 {
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

   bool Contains(float x, float y) const
   {
     return x >= left_ &&
            x <= right_ &&
            y >= top_ && 
            y <= bottom_;
   }

   const float left_, top_, right_, bottom_;
  };

  explicit inline BBoxCrop(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , num_attempts_(spec.GetArgument<int>("num_attempts"))
    , thresholds_({0.1, 0.3, 0.5, 0.7, 0.9})
  {
    //TODO: @pribalta Pass thresholds as argument?
    //TODO: @pribalta Pass the W/H scaling bounds?
    //TODO: @pribalta Pass the aspect ratio bounds?

  }

  virtual ~BBoxCrop() = default;

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx)
  {
    const auto& image = ws->Input<CPUBackend>(0);
    const auto& bounding_boxes = ws->Input<CPUBackend>(1);

    // TODO if overlap is 0 we bail
    
    const auto prospective_crop = FindProspectiveCrop(image,
                                                bounding_boxes,
                                                SelectMinimumOverlap());

    WriteCropToOutput(ws, prospective_crop.first);
    WriteBoxesToOutput(ws, prospective_crop.second);  
  }

  void WriteCropToOutput(SampleWorkspace *ws, const Rectangle& crop)
  {
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

  void WriteBoxesToOutput(SampleWorkspace *ws, const std::vector<Rectangle>& bounding_boxes)
  {
    auto *bbox_out = ws->Output<CPUBackend>(1);
    bbox_out->Resize({static_cast<int>(bounding_boxes.size()), bbox_size_});
    
    auto *bbox_out_data = bbox_out->mutable_data<float>();
    for (size_t i = 0; i < bounding_boxes.size(); ++i)
    {
      auto *output = bbox_out_data + i * bbox_size_;
      output[0] = bounding_boxes[i].left_; 
      output[1] = bounding_boxes[i].top_; 
      output[2] = bounding_boxes[i].right_; 
      output[3] = bounding_boxes[i].bottom_; 
    }
  }

  float SelectMinimumOverlap()
  {
    static std::uniform_int_distribution<> sampler(0, thresholds_.size());
     return thresholds_[sampler(rd_)];
  }
  
  float Rescale(unsigned int k)
  {
    static std::uniform_real_distribution<> sampler(0.3, 1.0);
    return sampler(rd_) * k;
  }

  float ValidAspectRatio(float width, float height) const
  {
    const float aspect_ratio = width / height;
    return aspect_ratio >= 0.5 && aspect_ratio <= 2;
  }
  
  Rectangle SamplePatch(float height, float width)
  {
    std::uniform_real_distribution<float> width_sampler(0., 1. - width),
                                          height_sampler(0., 1. - height);

    const auto left_offset = width_sampler(rd_);
    const auto height_offset = height_sampler(rd_);
    
    // Patch is LEFT/TOP/RIGHT/BOTTOM
    return Rectangle(left_offset,
                     height_offset,
                     left_offset + width,
                     height_offset + height);
  }

  std::vector<Rectangle> DiscardBoundingBoxesByCentroid(const Rectangle& crop, const Tensor<CPUBackend>& bounding_boxes)
  {
    std::vector<Rectangle> result;
    result.reserve(bounding_boxes.dim(0));

    // Discard bboxes whose centroid is not in the cropped area
    for (int i = 0; i < bounding_boxes.dim(0); ++i)
    {
      const auto* box = bounding_boxes.data<float>() + (i * bbox_size_);

      const float x_center = 0.5 * (box[0] + box[2]);
      const float y_center = 0.5 * (box[1] + box[3]);

      if (crop.Contains(x_center, y_center))
      {
        result.emplace_back(Rectangle(box[0], box[1], box[2], box[3]));
      }
    }

    return result;
  }

  float CalculateOverlap(const Rectangle& crop_candidate, const Rectangle& box)
  {
    // TODO
    return 0.0;
  }

  bool ValidOverlap(const Rectangle& crop_candidate, const std::vector<Rectangle>& bounding_boxes, float minimum_overlap)
  {
    for (const auto& box : bounding_boxes)
    {
      if (CalculateOverlap(crop_candidate, box) <= minimum_overlap)
      {
        return false;
      }
    }
     return true;
  }

  std::pair<Rectangle, std::vector<Rectangle>> FindProspectiveCrop(const Tensor<CPUBackend>& image, const Tensor<CPUBackend>& bounding_boxes, float minimum_overlap)
  {
    for (size_t i = 0; i < num_attempts_; ++i)
    {
      // Image is HWC
      const auto rescaled_height = Rescale(image.dim(0));
      const auto rescaled_width = Rescale(image.dim(1));

      if (ValidAspectRatio(rescaled_height, rescaled_width))
      {
        const auto candidate_crop = SamplePatch(rescaled_height, rescaled_width);
        const auto candidate_bounding_boxes = DiscardBoundingBoxesByCentroid(candidate_crop, bounding_boxes);

        if (ValidOverlap(candidate_crop, candidate_bounding_boxes, minimum_overlap))
        {
          return std::make_pair(candidate_crop, candidate_bounding_boxes);
        }
      }
    }
    
    // TODO what if we dont find anything?
  }

  const unsigned int num_attempts_;
  const std::vector<float> thresholds_;
  std::random_device rd_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_


