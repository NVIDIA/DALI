// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/generic/one_hot.cuh"

namespace dali {

class OneHotGPU : public OneHot<GPUBackend> {
 public:
  explicit OneHotGPU(const OpSpec &spec) : OneHot<GPUBackend>(spec) {
    scratch_mem_.set_type<uint8_t>();
  }

  ~OneHotGPU() override = default;

  USE_OPERATOR_MEMBERS();

 protected:
  void RunImpl(Workspace &ws) override;
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  template<typename OutputType, typename InputType>
  void RunImplTyped(Workspace &ws, int placement_axis);

 private:
  std::vector<one_hot::SampleDesc> sample_descs_;
  Tensor<GPUBackend> scratch_mem_;
  int recent_n_samples_ = 0;
};

bool OneHotGPU::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  int num_samples = input.shape().num_samples();
  if (num_samples != recent_n_samples_) {
    recent_n_samples_ = num_samples;
    int64_t samples_size = num_samples * sizeof(one_hot::SampleDesc);
    scratch_mem_.Resize({samples_size});
  }
  sample_descs_.clear();
  sample_descs_.reserve(num_samples);
  return OneHot<GPUBackend>::SetupImpl(output_desc, ws);
}

void OneHotGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  int output_sample_dim = output.shape().sample_dim();
  int placement_axis = get_placement_axis(output_sample_dim);
  output.SetLayout(GetOutputLayout(ws, placement_axis, output_sample_dim));
  TYPE_SWITCH(input.type(), type2id, InputType, ONE_HOT_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, ONE_HOT_TYPES, (
      RunImplTyped<OutputType, InputType>(ws, placement_axis);
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)); );       // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())); );     // NOLINT
}

template <typename OutputType, typename InputType>
void OneHotGPU::RunImplTyped(Workspace &ws, int axis) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  int num_samples = input.shape().num_samples();

  uint64_t max_out_vol = 1;
  const auto &shape = output.shape();
  for (int sample_id = 0; sample_id < num_samples; ++sample_id) {
    one_hot::SampleDesc sample;
    auto output_shape = shape.tensor_shape_span(sample_id);
    auto outer_vol = volume(output_shape.begin(), output_shape.begin() + axis);
    sample.inner_vol = volume(output_shape.begin() + axis + 1, output_shape.end());
    sample.inner_vol_classes = sample.inner_vol * num_classes_;
    sample.output_vol = outer_vol * sample.inner_vol_classes;
    sample.out = output.mutable_tensor<OutputType>(sample_id);
    sample.in = input.tensor<InputType>(sample_id);
    sample_descs_.push_back(sample);
    max_out_vol = std::max(max_out_vol, sample.output_vol);
  }

  auto stream = ws.stream();

  scratch_mem_.Copy(sample_descs_, stream);
  const auto *scratch_mem_gpu = scratch_mem_.data<one_hot::SampleDesc>();

  const int block = 256;
  auto grid = one_hot::gridHelper(max_out_vol, num_samples, block);

  one_hot::PopulateOneHot<OutputType, InputType><<<grid, block, 0, stream>>>(
    on_value_, off_value_, scratch_mem_gpu);
}

DALI_REGISTER_OPERATOR(OneHot, OneHotGPU, GPU);

}  // namespace dali
