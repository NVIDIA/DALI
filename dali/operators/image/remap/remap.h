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
class Remap : public Operator<Backend> {
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
  bool shift_pixels_{};
  std::vector<DALIInterpType> interps_;
  std::vector<boundary::Boundary<any>> borders_;

 private:
  template<typename T>
  bool SetupImplTyped(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
    const auto &input = ws.template Input<Backend>(0);
    DALI_ENFORCE(input.shape().ndim == 3, "Input has to be a HWC image.");

    // Output image has the same shape as input image
    output_desc.resize(1);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();

    AcquireArguments<T>(ws);

    return true;
  }


  template<typename DataType>
  void AcquireArguments(const Workspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);

    this->template GetPerSampleArgument(interps_, "interp", ws, curr_batch_size);

    //TODO borders
//    std::vector<std::string> border_modes;
//    this->template GetPerSampleArgument(border_modes, "border", ws, curr_batch_size);
//    std::vector<DataType> border_values;
//    this->template GetPerSampleArgument(border_values, "border_value", ws, curr_batch_size);
//    borders_.resize(curr_batch_size);
//    assert(border_values.size() == borders_.size());
//    for (int i = 0; i < curr_batch_size; i++) {
//      borders_[i] = {border_modes[i], {border_values[i]}};
//    }

    shift_pixels_ = spec_.template GetArgument<bool>("pixel_origin", &ws);//TODO
  }
};

}  // namespace dali::remap


#endif