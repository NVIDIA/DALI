// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include "dali/operators/generic/one_hot.h"

namespace dali {

namespace detail {

struct SampleDesc {
  uint64_t outer_vol, inner_vol, output_vol, inner_vol_classes;
  void *out = nullptr;
  const void *in = nullptr;
};

template <typename OutputType, typename InputType>
__global__ void PopulateOneHot(OutputType on_value, OutputType off_value,
                               const SampleDesc *samples) {
  uint64_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto &sample = samples[blockIdx.y];
  if (out_index >= sample.output_vol) {
    return;
  }
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  uint64_t i = out_index / sample.inner_vol_classes;
  uint64_t j = out_index % sample.inner_vol;
  uint64_t in_index = i * sample.inner_vol + j;
  unsigned int in_val = in[in_index];
  uint64_t on_out_index = i * sample.inner_vol_classes + in_val * sample.inner_vol + j;
  out[out_index] = on_out_index == out_index ? on_value : off_value;
}

}  // namespace detail

class OneHotGPU : public OneHot<GPUBackend> {
 public:
  explicit OneHotGPU(const OpSpec &spec) : OneHot<GPUBackend>(spec) {
    int64_t samples_size = batch_size_ * sizeof(detail::SampleDesc);
    scratch_mem_.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
    scratch_mem_.Resize({samples_size});
  }

  ~OneHotGPU() override = default;
  DISABLE_COPY_MOVE_ASSIGN(OneHotGPU);

  void RunImpl(workspace_t<GPUBackend> &ws) override;

  template<typename OutputType, typename InputType>
  void RunImplTyped(workspace_t<GPUBackend> &ws, int placement_axis);

  USE_OPERATOR_MEMBERS();

 private:
  std::vector<detail::SampleDesc> sample_descs_;
  Tensor<GPUBackend> scratch_mem_;
};

void OneHotGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);
  int output_sample_dim = output.shape().sample_dim();
  int placement_axis = get_placement_axis(output_sample_dim);
  output.SetLayout(GetOutputLayout(ws, placement_axis, output_sample_dim));
  TYPE_SWITCH(input.type().id(), type2id, InputType, ONE_HOT_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, ONE_HOT_TYPES, (
      RunImplTyped<OutputType, InputType>(ws, placement_axis);
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)); );       // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())); );     // NOLINT
}

template <typename OutputType, typename InputType>
void OneHotGPU::RunImplTyped(workspace_t<GPUBackend> &ws, int axis) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);

  sample_descs_.clear();
  sample_descs_.reserve(batch_size_);

  uint64_t max_out_vol = 1;
  const auto &shape = output.shape();
  for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
    detail::SampleDesc sample;
    auto output_shape = shape[sample_id];
    sample.outer_vol = volume(output_shape.begin(), output_shape.begin() + axis);
    sample.inner_vol = volume(output_shape.begin() + axis + 1, output_shape.end());
    sample.inner_vol_classes = sample.inner_vol * num_classes_;
    sample.output_vol = sample.outer_vol * sample.inner_vol_classes;
    sample.out = output.mutable_tensor<OutputType>(sample_id);
    sample.in = input.tensor<InputType>(sample_id);
    sample_descs_.push_back(sample);
    max_out_vol = std::max(max_out_vol, sample.output_vol);
  }

  auto stream = ws.stream();

  scratch_mem_.Copy(sample_descs_, stream);
  const auto scratch_mem_gpu = scratch_mem_.data<detail::SampleDesc>();

  const int block = 256;
  auto block_size = (max_out_vol + (block - 1)) / block;
  dim3 grid(block_size, batch_size_);

  OutputType on_value = on_value_, off_value = off_value_;
  detail::PopulateOneHot<OutputType, InputType><<<grid, block>>>(
    on_value, off_value, scratch_mem_gpu);
}

DALI_REGISTER_OPERATOR(OneHot, OneHotGPU, GPU);

}  // namespace dali
