// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/common/cast.cuh"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/operators/generic/cast.h"


namespace dali {

class CastGPU : public Cast<GPUBackend> {
 public:
  explicit CastGPU(const OpSpec &spec) : Cast<GPUBackend>{spec}, block_setup_{block_volume_scale} {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) override;
  void RunImpl(DeviceWorkspace &ws) override;

  ~CastGPU() override = default;

 protected:
  void PrepareBlocks(const DeviceWorkspace &ws);

  static const int block_volume_scale = 4;

 private:
  using GpuBlockSetup = kernels::BlockSetup<1, -1>;

  GpuBlockSetup block_setup_;
  std::vector<kernels::CastSampleDesc> samples_;

  USE_OPERATOR_MEMBERS();
};

bool CastGPU::SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) {
  PrepareBlocks(ws);
  return Cast<GPUBackend>::SetupImpl(output_desc, ws);
}

void CastGPU::PrepareBlocks(const DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  const auto &input_shape = input.shape();
  std::array<std::pair<int, int>, 1> collapse_groups = {{{0, input_shape.sample_dim()}}};
  auto collapsed_shape = collapse_dims<1>(input.shape(), collapse_groups);

  block_setup_.SetupBlocks(collapsed_shape, true);
}

void CastGPU::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  const auto &input_shape = input.shape();
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  kernels::DynamicScratchpad scratchpad({}, ws.stream());
  auto num_samples = input_shape.num_samples();
  samples_.resize(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    samples_[sample_id].output = output.raw_mutable_tensor(sample_id);
    samples_[sample_id].input = input.raw_tensor(sample_id);
  }

  std::vector<kernels::CastSampleBlockDesc> params_host(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    params_host[sample_id].sample_size = volume(input.tensor_shape(sample_id));
  }

  auto blocks = block_setup_.Blocks();
  for (int block_id = 0, sample_id = -1; block_id < blocks.size(); block_id++) {
    if (blocks[block_id].sample_idx != sample_id) {
      sample_id++;
      params_host[sample_id].first_block = block_id;
    }
  }

  kernels::CastSampleBlockDesc *params_dev;
  kernels::CastSampleDesc *samples_dev;
  std::tie(params_dev, samples_dev) = scratchpad.ToContiguousGPU(ws.stream(),
                                                                 params_host, samples_);

  DALIDataType itype = input.type();
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  TYPE_SWITCH(output_type_, type2id, OType, CAST_ALLOWED_TYPES, (
    TYPE_SWITCH(itype, type2id, IType, CAST_ALLOWED_TYPES, (
      kernels::BinSearchCastKernel<OType, IType>
          <<<grid_dim, block_dim, 0, ws.stream()>>>(samples_dev, params_dev,
            num_samples, block_volume_scale);
    ), DALI_FAIL(make_string("Invalid input type: ", itype)););  // NOLINT(whitespace/parens)
  ), DALI_FAIL(make_string("Invalid output type: ", output_type_)););  // NOLINT(whitespace/parens)
}

DALI_REGISTER_OPERATOR(Cast, CastGPU, GPU);

}  // namespace dali
