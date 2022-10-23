// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNasdasdM_REMAP_H_
#define DALI_KERNasdasdM_REMAP_H_

#include "dali/core/cuda_stream_pool.h"
#include "dali/kernels/imgproc/geom/remap.h"
#include "dali/kernels/imgproc/geom/remap_npp.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "include/dali/core/geom/mat.h"
#include "include/dali/core/span.h"
#include "include/dali/core/static_switch.h"

namespace dali::remap {

#define REMAP_SUPPORTED_TYPES (uint8_t)
//#define REMAP_SUPPORTED_TYPES (uint8_t, int16_t, uint16_t, float)

template<typename Backend>
class Remap : public Operator<Backend> { //TODO SequenceOperator
 public:
  explicit Remap(const OpSpec &spec) : Operator<Backend>(spec) {}


  ~Remap() override = default;
  DISABLE_COPY_MOVE_ASSIGN(Remap);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    TYPE_SWITCH(input.type(), type2id, InputType, REMAP_SUPPORTED_TYPES, (
            {
              return SetupImplTyped<InputType>(output_desc, ws);
            }
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
  }


 protected:
  USE_OPERATOR_MEMBERS();
  bool shift_pixels_ = false;
  float shift_value_ = 0;
  std::vector<DALIInterpType> interps_;
  std::vector<dali::kernels::Roi<2>> rois_;

 private:
  template<typename T>
  bool SetupImplTyped(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
    const auto &input = ws.template Input<Backend>(0);
    DALI_ENFORCE(input.shape().ndim == 3, "Input has to be a HWC image.");

    AcquireArguments<T>(ws, input.shape());

    // If ROI is defined, it determines the output shape.
    // If ROI is not defined, the output shape is the same as input.
    output_desc.resize(1);
    output_desc[0].shape = rois_.empty() ?
                           input.shape() : dali::kernels::ShapeFromRoi(make_cspan(rois_), 3);
    output_desc[0].type = input.type();


    return true;
  }


  template<typename DataType, int ndims>
  void AcquireArguments(const Workspace &ws, TensorListShape<ndims> input_shape) {
    auto curr_batch_size = ws.GetInputBatchSize(0);

    // Interpolation setting
    auto interp = spec_.template GetArgument<DALIInterpType>("interp", &ws);
    interps_ = {static_cast<size_t>(curr_batch_size), interp};

    // Pixel origin setting
    auto pixel_origin = spec_.template GetArgument<std::string>("pixel_origin", &ws);
    if (pixel_origin == "corner") {
      shift_pixels_ = true;
      shift_value_ = -.5f;
    } else if (pixel_origin != "center") {
      DALI_FAIL(
              R"msg("Undefined pixel_origin parameter. Please choose one of: ["center", "corner"])msg");
    }

    // ROI setting
    bool has_roi_start = spec_.ArgumentDefined("roi_start");
    bool has_roi_end = spec_.ArgumentDefined("roi_end");
    DALI_ENFORCE(has_roi_start == has_roi_end,
                 "``roi_start`` and ``roi_end`` must be specified together");
    bool has_roi = has_roi_start && has_roi_end;
    bool roi_relative = spec_.template GetArgument<bool>("roi_relative");
    std::vector<float> roi_start, roi_end;
    if (has_roi) {
      GetShapeLikeArgument<float>(roi_start, spec_, "roi_start", ws, curr_batch_size, 2);
      GetShapeLikeArgument<float>(roi_end, spec_, "roi_end", ws, curr_batch_size, 2);
    }
    assert(roi_start.size() == roi_end.size());
    for (int i = 0; i < roi_start.size(); i += 2) {
      dali::kernels::Roi<2> r = {{roi_start[i], roi_start[i + 1]},
                                 {roi_end[i],   roi_end[i + 1]}};
      if (roi_relative) {
        r.lo.x *= input_shape.template tensor_shape(i * .5)[1];
        r.lo.y *= input_shape.template tensor_shape(i * .5)[0];
        r.hi.x *= input_shape.template tensor_shape(i * .5)[1];
        r.hi.y *= input_shape.template tensor_shape(i * .5)[0];
      }
      rois_.emplace_back(r);
    }
  }
};

}  // namespace dali::remap


#endif