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

namespace {

template <typename T = float>
struct SampleDesc {
  float* out = nullptr;
  const float* in = nullptr;
  int64_t size = 0;
  uint8_t flip_dim_mask = 0;
};

template <typename T = float>
__global__ void CoordFlipKernel(const SampleDesc<T>* samples, int ndim) {
  int64_t block_size = blockDim.y * blockDim.x;
  int64_t grid_size = gridDim.x * block_size;
  int sample_idx = blockIdx.y;
  const auto &sample = samples[sample_idx];
  int64_t offset = block_size * blockIdx.x;
  int64_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int64_t idx = offset + tid; idx < sample.size; idx += grid_size) {
    int d = idx % ndim;
    bool flip = static_cast<bool>(sample.flip_dim_mask & (1 << d));
    sample.out[idx] = flip ? T(1) - sample.in[idx] : sample.in[idx];
  }
}

}  // namespace

class CoordFlipGPU : public CoordFlip<GPUBackend> {
 public:
  explicit CoordFlipGPU(const OpSpec &spec)
      : CoordFlip<GPUBackend>(spec) {}

  ~CoordFlipGPU() override = default;
  DISABLE_COPY_MOVE_ASSIGN(CoordFlipGPU);

  void RunImpl(workspace_t<GPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<GPUBackend>::RunImpl;
  using CoordFlip<GPUBackend>::layout_;

 private:
  std::vector<SampleDesc<float>> sample_descs_;
  Tensor<GPUBackend> scratchpad_;
};

void CoordFlipGPU::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Input is expected to be float");

  auto &output = ws.OutputRef<GPUBackend>(0);

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

  sample_descs_.clear();
  sample_descs_.reserve(batch_size_);
  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
    SampleDesc<float> sample_desc;
    sample_desc.in = input.tensor<float>(sample_id);
    sample_desc.out = output.mutable_tensor<float>(sample_id);
    sample_desc.size = volume(input.tensor_shape(sample_id));
    assert(sample_desc.size == volume(output.tensor_shape(sample_id)));

    bool horizontal_flip = spec_.GetArgument<int>("horizontal", &ws, sample_id);
    bool vertical_flip = spec_.GetArgument<int>("vertical", &ws, sample_id);
    bool depthwise_flip = spec_.GetArgument<int>("depthwise", &ws, sample_id);

    if (horizontal_flip) {
      sample_desc.flip_dim_mask |= (1 << x_dim);
    }

    if (vertical_flip) {
      sample_desc.flip_dim_mask |= (1 << y_dim);
    }

    if (depthwise_flip) {
      sample_desc.flip_dim_mask |= (1 << z_dim);
    }
    sample_descs_.emplace_back(std::move(sample_desc));
  }

  scratchpad_.set_type(TypeInfo::Create<uint8_t>());
  int64_t sz = batch_size_ * sizeof(SampleDesc<float>);
  scratchpad_.Resize({sz});
  auto sample_descs_gpu_ = reinterpret_cast<SampleDesc<float>*>(scratchpad_.mutable_data<uint8_t>());
  auto stream = ws.stream();
  CUDA_CALL(
    cudaMemcpyAsync(sample_descs_gpu_, sample_descs_.data(), sz, cudaMemcpyHostToDevice, stream));

  dim3 block(32, 32);
  auto blocks_per_sample = std::max(32, 1024 / batch_size_);
  dim3 grid(blocks_per_sample, batch_size_);
  CoordFlipKernel<<<grid, block, 0, stream>>>(sample_descs_gpu_, ndim_);
}

DALI_REGISTER_OPERATOR(CoordFlip, CoordFlipGPU, GPU);

}  // namespace dali
