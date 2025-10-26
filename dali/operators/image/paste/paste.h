// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PASTE_PASTE_H_
#define DALI_OPERATORS_IMAGE_PASTE_PASTE_H_

#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Paste : public StatelessOperator<Backend> {
 public:
  // 6 values: in_H, in_W, out_H, out_W, paste_y, paste_x
  static const int NUM_INDICES = 6;

  explicit Paste(const OpSpec &spec)
      : StatelessOperator<Backend>(spec), C_(spec.GetArgument<int>("n_channels")) {
    // Kind of arbitrary, we need to set some limit here
    // because we use static shared memory for storing
    // fill value array
    DALI_ENFORCE(C_ <= 1024, "n_channels of more than 1024 is not supported");
    std::vector<uint8_t> rgb;
    GetSingleOrRepeatedArg(spec, rgb, "fill_value", C_);
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      fill_value_.set_order(cudaStreamLegacy);
    } else {
      // Disable pinned memory for CPU backend for no-gpu compatibility
      fill_value_.set_pinned(false);
      input_ptrs_.set_pinned(false);
      output_ptrs_.set_pinned(false);
      in_out_dims_paste_yx_.set_pinned(false);
    }
    fill_value_.Copy(rgb);

    input_ptrs_.reserve(max_batch_size_ * sizeof(uint8_t *));
    output_ptrs_.reserve(max_batch_size_ * sizeof(uint8_t *));
    in_out_dims_paste_yx_.reserve(max_batch_size_ * sizeof(int) * NUM_INDICES);
  }

  virtual inline ~Paste() = default;

 protected:
  USE_OPERATOR_MEMBERS();

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    input_ptrs_.set_type<const uint8_t *>();
    output_ptrs_.set_type<uint8_t *>();
    in_out_dims_paste_yx_.set_type<int>();
    input_ptrs_.Resize({curr_batch_size});
    output_ptrs_.Resize({curr_batch_size});
    in_out_dims_paste_yx_.Resize({curr_batch_size * NUM_INDICES});
    return false;
  }

  void SetupSampleParams(Workspace &ws) {
    auto &input = ws.Input<Backend>(0);
    auto &output = ws.Output<Backend>(0);
    auto curr_batch_size = ws.GetInputBatchSize(0);

    std::vector<TensorShape<>> output_shape(curr_batch_size);

    for (int i = 0; i < curr_batch_size; ++i) {
      auto input_shape = input.tensor_shape(i);
      DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

      int H = input_shape[0];
      int W = input_shape[1];
      C_ = input_shape[2];

      float ratio = spec_.template GetArgument<float>("ratio", &ws, i);
      DALI_ENFORCE(ratio >= 1., "ratio of less than 1 is not supported");

      int new_H = static_cast<int>(ratio * H);
      int new_W = static_cast<int>(ratio * W);

      int min_canvas_size_ = spec_.template GetArgument<float>("min_canvas_size", &ws, i);
      DALI_ENFORCE(min_canvas_size_ >= 0., "min_canvas_size_ of less than 0 is not supported");

      new_H = std::max(new_H, static_cast<int>(min_canvas_size_));
      new_W = std::max(new_W, static_cast<int>(min_canvas_size_));

      output_shape[i] = {new_H, new_W, C_};

      float paste_x_ = spec_.template GetArgument<float>("paste_x", &ws, i);
      float paste_y_ = spec_.template GetArgument<float>("paste_y", &ws, i);
      DALI_ENFORCE(0 <= paste_x_ && paste_x_ <= 1, "paste_x must be in range [0, 1]");
      DALI_ENFORCE(0 <= paste_y_ && paste_y_ <= 1, "paste_y must be in range [0, 1]");
      int paste_x = paste_x_ * (new_W - W);
      int paste_y = paste_y_ * (new_H - H);

      int sample_dims_paste_yx[] = {H, W, new_H, new_W, paste_y, paste_x};
      int *sample_data = in_out_dims_paste_yx_.template mutable_data<int>() + (i * NUM_INDICES);
      std::copy(sample_dims_paste_yx, sample_dims_paste_yx + NUM_INDICES, sample_data);
    }

    output.Resize(output_shape, input.type());
    output.SetLayout("HWC");
  }

  void SetupGPUPointers(Workspace &ws);

  void RunHelper(Workspace &ws);

  void RunImpl(Workspace &ws) override {
    SetupSampleParams(ws);
    SetupGPUPointers(ws);
    RunHelper(ws);
  }

  // Op parameters
  int C_;
  Tensor<Backend> fill_value_;

  Tensor<CPUBackend> input_ptrs_, output_ptrs_, in_out_dims_paste_yx_;
  Tensor<GPUBackend> input_ptrs_gpu_, output_ptrs_gpu_, in_out_dims_paste_yx_gpu_;

  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PASTE_PASTE_H_
