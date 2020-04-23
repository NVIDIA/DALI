// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/coord/coord_flip.h"

namespace dali {

DALI_SCHEMA(CoordFlip)
    .DocStr(
        R"code(Transforms normalized coordinates (range [0.0, 1.0]) so that they map to the same place after
horizontal or/and vertical flip of the input they refer to.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<TensorLayout>(
      "layout",
      R"code(Determines the layout of the coordinates.
  Possible values are:

  ``x`` (horizontal position), ``y`` (vertical position), ``z`` (depthwise position),

Note: If left empty, ``"xy"`` or ``"xyz"`` will be assumed, depending on the number of dimensions.
)code",
      TensorLayout{""})
    .AddOptionalArg("horizontal", R"code(Perform flip along horizontal axis.)code", 1, true)
    .AddOptionalArg("vertical", R"code(Perform flip along vertical axis.)code", 0, true)
    .AddOptionalArg("depthwise", R"code(Perform flip along depthwise axis.)code", 0, true);

class CoordFlipCPU : public CoordFlip<CPUBackend> {
 public:
  explicit CoordFlipCPU(const OpSpec &spec)
      : CoordFlip<CPUBackend>(spec) {}

  ~CoordFlipCPU() override = default;
  DISABLE_COPY_MOVE_ASSIGN(CoordFlipCPU);

  void RunImpl(workspace_t<CPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;
  using CoordFlip<CPUBackend>::layout_;
};

void CoordFlipCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Input is expected to be float");

  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &thread_pool = ws.GetThreadPool();

  if (layout_.empty()) {
    layout_ = ndim_ == 2 ? "xy" : "xyz";
  }

  int x_dim = layout_.find('x');
  DALI_ENFORCE(x_dim >= 0, "Dimension \"x\" not found in the layout");

  int y_dim = layout_.find('y');
  if (ndim_ > 1)
    DALI_ENFORCE(y_dim >= 0, "Dimension \"y\" not found in the layout");

  int z_dim = layout_.find('z');
  if (ndim_ > 2)
    DALI_ENFORCE(z_dim >= 0, "Dimension \"z\" not found in the layout");

  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
    bool horizontal_flip = spec_.GetArgument<int>("horizontal", &ws, sample_id);
    bool vertical_flip = spec_.GetArgument<int>("vertical", &ws, sample_id);
    bool depthwise_flip = spec_.GetArgument<int>("depthwise", &ws, sample_id);
    std::array<bool, 3> flip_dim = {false, false, false};

    if (horizontal_flip) {
      flip_dim[x_dim] = horizontal_flip;
    }

    if (vertical_flip) {
      flip_dim[y_dim] = vertical_flip;
    }

    if (depthwise_flip) {
      flip_dim[z_dim] = depthwise_flip;
    }

    thread_pool.DoWorkWithID(
        [this, &input, &output, sample_id, flip_dim](int thread_id) {
          const auto *in = input[sample_id].data<float>();
          auto *out = output[sample_id].mutable_data<float>();
          auto in_size = volume(input[sample_id].shape());
          int d = 0;
          int64_t i = 0;
          for (; i < in_size; i++, d++) {
            if (d == ndim_) d = 0;
            assert(in[i] >= 0.0f && in[i] <= 1.0f);
            out[i] = flip_dim[d] ? 1.0f - in[i] : in[i];
          }
        });
  }
  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(CoordFlip, CoordFlipCPU, CPU);

}  // namespace dali
