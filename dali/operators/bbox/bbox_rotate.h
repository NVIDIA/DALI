// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_BBOX_BBOX_ROTATE_H_
#define DALI_OPERATORS_BBOX_BBOX_ROTATE_H_

#include "dali/core/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

template <typename Backend>
class BBoxRotate : public StatelessOperator<Backend> {
 public:
  enum class Mode {
    Expand,
    Halfway,
    Fixed
  };

  explicit inline BBoxRotate(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        bbox_normalized_(spec.GetArgument<bool>("bbox_normalized")),
        keep_size_(spec.GetArgument<bool>("keep_size")),
        remove_threshold_(spec.GetArgument<float>("remove_threshold")) {
    const auto &mode_str = spec.GetArgument<std::string>("mode");
    if (mode_str == "expand") {
      mode_ = Mode::Expand;
    } else if (mode_str == "halfway") {
      mode_ = Mode::Halfway;
    } else if (mode_str == "fixed") {
      mode_ = Mode::Fixed;
    } else {
      DALI_FAIL("Unknown mode: ", mode_str, ". Supported modes are: expand, halfway, fixed.");
    }
    const auto &bbox_layout = spec.GetArgument<TensorLayout>("bbox_layout");
    if (bbox_layout == "xyWH") {
      use_ltrb_ = false;
    } else if (bbox_layout == "xyXY") {
      use_ltrb_ = true;
    } else {
      DALI_FAIL("Unknown bbox_layout: ", bbox_layout, ". Supported layouts are: xyWH, xyXY.");
    }

    const auto &shape_layout = spec.GetArgument<dali::TensorLayout>("shape_layout");
    shape_wh_index_.first = shape_layout.find('W');
    shape_wh_index_.second = shape_layout.find('H');
    if (shape_wh_index_.first == -1 || shape_wh_index_.second == -1) {
      DALI_FAIL("shape_layout does not contain 'W' and/or 'H'");
    }
  }

 protected:
  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const TensorListShape<-1> boxShapeList = ws.GetInputShape(0);
    TensorListShape<1> buffer_shape(boxShapeList.num_samples());
    for (int i = 0; i < ws.GetRequestedBatchSize(0); i++) {
      const auto &sampleShape = boxShapeList[i];
      if (sampleShape.size() != 2 || sampleShape[1] != 4) {
        DALI_ERROR("First Input to fn.bbox_rotate should be [N,4], got ", sampleShape);
      }
      buffer_shape.tensor_shape_span(i)[0] = sampleShape[0] * 8;  // num_boxes * 4 corners
    }
    bbox_rotate_buffer_.Resize(buffer_shape, DALI_FLOAT);

    const auto numInput = ws.NumInput();
    output_desc.resize(numInput);
    output_desc[0].shape = boxShapeList;
    output_desc[0].type = DALI_FLOAT;
    if (numInput == 2) {
      const auto &labelShapeList = ws.GetInputShape(1);
      output_desc[1].shape = labelShapeList;
      output_desc[1].type = DALI_INT32;
      for (int i = 0; i < labelShapeList.size(); i++) {
        if (labelShapeList[i].size() != 1) {
          DALI_FAIL("Label input to fn.bbox_rotate should be [N], got ", labelShapeList[i]);
        }
        if (labelShapeList[i][0] != boxShapeList[i][0]) {
          DALI_FAIL("Number of labels must match number of boxes. Got: ", labelShapeList[i][0],
                    " and ", boxShapeList[i][0]);
        }
      }
    }

    return true;
  }

  void RunImpl(Workspace &ws) override;

  bool bbox_normalized_;
  bool use_ltrb_;
  bool keep_size_;
  float remove_threshold_;
  std::pair<int, int> shape_wh_index_;
  Mode mode_ = Mode::Expand;
  TensorList<Backend> bbox_rotate_buffer_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BBOX_ROTATE_H_
