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

#include "dali/operators/geometry/coord_flip.h"
#include <utility>
#include <vector>

namespace dali {

namespace {

template <typename T = float>
struct SampleDesc {
  float* out = nullptr;
  const float* in = nullptr;
  int64_t size = 0;
  uint8_t flip_dim_mask = 0;
  float mirrored_origin[3];
};

template <typename T = float>
__global__ void CoordFlipKernel(const SampleDesc<T>* samples, int ndim) {
  int64_t block_size = blockDim.x;
  int64_t grid_size = gridDim.x * block_size;
  int sample_idx = blockIdx.y;
  const auto &sample = samples[sample_idx];
  int64_t offset = block_size * blockIdx.x;
  int64_t tid = threadIdx.x;
  for (int64_t idx = offset + tid; idx < sample.size; idx += grid_size) {
    int d = idx % ndim;
    bool flip = sample.flip_dim_mask & (1 << d);
    sample.out[idx] = flip ? sample.mirrored_origin[d] - sample.in[idx] : sample.in[idx];
  }
}

}  // namespace

class CoordFlipGPU : public CoordFlip<GPUBackend> {
 public:
  explicit CoordFlipGPU(const OpSpec &spec)
      : CoordFlip<GPUBackend>(spec) {}

  ~CoordFlipGPU() override = default;
  DISABLE_COPY_MOVE_ASSIGN(CoordFlipGPU);

  void RunImpl(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<GPUBackend>::RunImpl;
  using CoordFlip<GPUBackend>::layout_;

 private:
  std::vector<SampleDesc<float>> sample_descs_;
  Tensor<GPUBackend> scratchpad_;
};

void CoordFlipGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  auto curr_batch_size = ws.GetInputBatchSize(0);

  sample_descs_.clear();
  sample_descs_.reserve(max_batch_size_);
  for (int sample_id = 0; sample_id < curr_batch_size; sample_id++) {
    SampleDesc<float> sample_desc;
    sample_desc.in = input.tensor<float>(sample_id);
    sample_desc.out = output.mutable_tensor<float>(sample_id);
    sample_desc.size = volume(input.tensor_shape(sample_id));
    assert(sample_desc.size == volume(output.tensor_shape(sample_id)));

    bool flip_x = spec_.GetArgument<int>("flip_x", &ws, sample_id);
    bool flip_y = spec_.GetArgument<int>("flip_y", &ws, sample_id);
    bool flip_z = spec_.GetArgument<int>("flip_z", &ws, sample_id);

    if (flip_x) {
      sample_desc.flip_dim_mask |= (1 << x_dim_);
    }

    if (flip_y) {
      sample_desc.flip_dim_mask |= (1 << y_dim_);
    }

    if (flip_z) {
      sample_desc.flip_dim_mask |= (1 << z_dim_);
    }

    sample_desc.mirrored_origin[x_dim_] =
        2.0f * spec_.GetArgument<float>("center_x", &ws, sample_id);
    sample_desc.mirrored_origin[y_dim_] =
        2.0f * spec_.GetArgument<float>("center_y", &ws, sample_id);
    sample_desc.mirrored_origin[z_dim_] =
        2.0f * spec_.GetArgument<float>("center_z", &ws, sample_id);

    sample_descs_.emplace_back(std::move(sample_desc));
  }

  int64_t sz = curr_batch_size * sizeof(SampleDesc<float>);
  scratchpad_.Resize({sz}, DALI_UINT8);
  auto sample_descs_gpu_ = reinterpret_cast<SampleDesc<float>*>(
      scratchpad_.mutable_data<uint8_t>());
  auto stream = ws.stream();
  CUDA_CALL(
    cudaMemcpyAsync(sample_descs_gpu_, sample_descs_.data(), sz, cudaMemcpyHostToDevice, stream));

  int block = 1024;
  auto blocks_per_sample = std::max(32, 1024 / curr_batch_size);
  dim3 grid(blocks_per_sample, curr_batch_size);
  CoordFlipKernel<<<grid, block, 0, stream>>>(sample_descs_gpu_, ndim_);
}

DALI_REGISTER_OPERATOR(CoordFlip, CoordFlipGPU, GPU);

}  // namespace dali
