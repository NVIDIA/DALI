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

#include "dali/operators/audio/preemphasis_filter_op.h"
#include <vector>
#include "dali/pipeline/data/types.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {

namespace {

template <typename OutputType, typename InputType>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  float coeff;
  int64_t size;
};

using BorderType = PreemphasisFilter<GPUBackend>::BorderType;

template <typename OutputType, typename InputType>
void __global__ PreemphasisFilterKernel(const SampleDescriptor<OutputType, InputType> *samples,
                                        BorderType border_type) {
  const auto &sample = samples[blockIdx.y];
  int64_t block_size = blockDim.x;
  int64_t block_start = block_size * blockIdx.x;
  int64_t grid_stride = block_size * gridDim.x;

  int64_t k = block_start + threadIdx.x;
  if (k >= sample.size)
    return;

  if (k == 0) {
    if (border_type == BorderType::Zero) {
      sample.out[k] = sample.in[k];
    } else {
      // BorderType::Reflect or BorderType::Clamp
      InputType border = (border_type == BorderType::Reflect) ? sample.in[1] : sample.in[0];
      sample.out[k] = sample.in[k] - sample.coeff * border;
    }
    k += grid_stride;
  }
  for (; k < sample.size; k += grid_stride)
    sample.out[k] = sample.in[k] - sample.coeff * sample.in[k-1];
}

}  // namespace

class PreemphasisFilterGPU : public PreemphasisFilter<GPUBackend> {
 public:
  explicit PreemphasisFilterGPU(const OpSpec &spec) : PreemphasisFilter<GPUBackend>(spec) {
    // void is OK here, pointer sizes are the same size
    int64_t sz = max_batch_size_ * sizeof(SampleDescriptor<void, void>);
    scratch_mem_.Resize({sz}, DALI_UINT8);
  }
  void RunImpl(Workspace &ws) override;

 private:
  template <typename OutputType, typename InputType>
  void RunImplTyped(Workspace &ws);

  Tensor<GPUBackend> scratch_mem_;
};

template <typename OutputType, typename InputType>
void PreemphasisFilterGPU::RunImplTyped(Workspace &ws) {
  using SampleDesc = SampleDescriptor<OutputType, InputType>;
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  auto curr_batch_size = ws.GetInputBatchSize(0);

  auto stream = ws.stream();
  auto scratch = kernels::DynamicScratchpad(AccessOrder(stream));
  auto samples_cpu = scratch.Allocate<mm::memory_kind::pinned, SampleDesc>(curr_batch_size);
  for (int sample_idx = 0; sample_idx < curr_batch_size; sample_idx++) {
    auto &sample = samples_cpu[sample_idx];
    sample.in = input.tensor<InputType>(sample_idx);
    sample.out = output.mutable_tensor<OutputType>(sample_idx);
    sample.size = volume(input.tensor_shape(sample_idx));
    sample.coeff = preemph_coeff_[sample_idx];
  }

  int64_t sz = curr_batch_size * sizeof(SampleDesc);
  scratch_mem_.Resize({sz}, DALI_UINT8);
  auto sample_descs_gpu = reinterpret_cast<SampleDesc*>(scratch_mem_.mutable_data<uint8_t>());

  CUDA_CALL(
    cudaMemcpyAsync(sample_descs_gpu, samples_cpu, sz, cudaMemcpyHostToDevice, stream));

  int block = 256;
  auto blocks_per_sample = std::max(32, 1024 / curr_batch_size);
  dim3 grid(blocks_per_sample, curr_batch_size);
  PreemphasisFilterKernel<<<grid, block, 0, stream>>>(sample_descs_gpu, border_type_);
}

void PreemphasisFilterGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, InputType, PREEMPH_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, PREEMPH_TYPES, (
      RunImplTyped<OutputType, InputType>(ws);
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())));  // NOLINT
}

DALI_REGISTER_OPERATOR(PreemphasisFilter, PreemphasisFilterGPU, GPU);

}  // namespace dali
