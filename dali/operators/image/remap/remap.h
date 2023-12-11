// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_REMAP_H_
#define DALI_OPERATORS_IMAGE_REMAP_REMAP_H_

#include <string>
#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/kernels/imgproc/geom/remap.h"
#include "dali/kernels/imgproc/geom/remap_npp.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "include/dali/core/geom/mat.h"
#include "include/dali/core/span.h"
#include "include/dali/core/static_switch.h"


namespace dali {
namespace remap {

#define REMAP_SUPPORTED_TYPES (uint8_t, int16_t, uint16_t, float)

template <typename Backend>
class Remap : public SequenceOperator<Backend, StatelessOperator> {
 public:
  using Base = SequenceOperator<Backend, StatelessOperator>;
  explicit Remap(const OpSpec &spec) : Base(spec) {}


  ~Remap() override = default;
  DISABLE_COPY_MOVE_ASSIGN(Remap);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.template Input<Backend>(0);

    AcquireArguments(ws);

    // The output shape is the same as input
    output_desc.resize(1);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();

    return true;
  }


 protected:
  USE_OPERATOR_MEMBERS();
  bool shift_pixels_ = false;
  float shift_value_ = 0;
  std::vector<DALIInterpType> interps_;

 private:
  void AcquireArguments(const Workspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);

    // Interpolation setting
    auto interp = spec_.template GetArgument<DALIInterpType>("interp", &ws);
    interps_.resize(static_cast<size_t>(curr_batch_size), interp);

    // Pixel origin setting
    auto pixel_origin = spec_.template GetArgument<std::string>("pixel_origin", &ws);
    if (pixel_origin == "corner") {
      shift_pixels_ = true;
      shift_value_ = -.5f;
    } else if (pixel_origin != "center") {
      DALI_FAIL(
              R"msg("Undefined pixel_origin parameter. Please choose one of: ["center", "corner"])msg");
    }
  }
};

}  // namespace remap
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_REMAP_H_
