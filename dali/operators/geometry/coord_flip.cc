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

#include "dali/operators/geometry/coord_flip.h"

namespace dali {

DALI_SCHEMA(CoordFlip)
    .DocStr(
        R"code(Transforms vectors or points by flipping (reflecting) their coordinates with
respect to a given center.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<TensorLayout>(
      "layout",
      R"code(Determines the order of coordinates in the input.

The string should consist of the following characters:

  - "x" (horizontal coordinate),
  - "y" (vertical coordinate),
  - "z" (depthwise coordinate),

.. note::
  If left empty, depending on the number of dimensions, the "x", "xy",
  or "xyz" values are assumed.
)code",
      TensorLayout{""})
    .AddOptionalArg("flip_x", R"code(Flip the horizontal (x) coordinate.)code", 1, true)
    .AddOptionalArg("flip_y", R"code(Flip the vertical (y) coordinate.)code", 0, true)
    .AddOptionalArg("flip_z", R"code(Flip the depthwise (z) coordinate.)code", 0, true)
    .AddOptionalArg("center_x", R"code(Flip center in the horizontal axis.)code", 0.5f, true)
    .AddOptionalArg("center_y", R"code(Flip center in the vertical axis.)code", 0.5f, true)
    .AddOptionalArg("center_z", R"code(Flip center in the depthwise axis.)code", 0.5f, true);


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
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &thread_pool = ws.GetThreadPool();
  auto curr_batch_size = ws.GetInputBatchSize(0);

  for (int sample_id = 0; sample_id < curr_batch_size; sample_id++) {
    std::array<bool, 3> flip_dim = {false, false, false};
    flip_dim[x_dim_] = spec_.GetArgument<int>("flip_x", &ws, sample_id);
    flip_dim[y_dim_] = spec_.GetArgument<int>("flip_y", &ws, sample_id);
    flip_dim[z_dim_] = spec_.GetArgument<int>("flip_z", &ws, sample_id);

    std::array<float, 3> mirrored_origin = {1.0f, 1.0f, 1.0f};
    mirrored_origin[x_dim_] = 2.0f * spec_.GetArgument<float>("center_x", &ws, sample_id);
    mirrored_origin[y_dim_] = 2.0f * spec_.GetArgument<float>("center_y", &ws, sample_id);
    mirrored_origin[z_dim_] = 2.0f * spec_.GetArgument<float>("center_z", &ws, sample_id);

    auto in_size = volume(input[sample_id].shape());
    thread_pool.AddWork(
        [this, &input, in_size, &output, sample_id, flip_dim, mirrored_origin](int thread_id) {
          const auto *in = input[sample_id].data<float>();
          auto *out = output[sample_id].mutable_data<float>();
          int d = 0;
          int64_t i = 0;
          for (; i < in_size; i++, d++) {
            if (d == ndim_) d = 0;
            auto in_val = in[i];
            out[i] = flip_dim[d] ? mirrored_origin[d] - in_val : in_val;
          }
        }, in_size);
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(CoordFlip, CoordFlipCPU, CPU);

}  // namespace dali
