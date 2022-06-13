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
  TensorListShape<1> collapsed_shape_;

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
  if (input.sample_dim() > 0) {
    std::array<std::pair<int, int>, 1> collapse_groups = {{{0, input_shape.sample_dim()}}};
    collapsed_shape_ = collapse_dims<1>(input_shape, collapse_groups);
  } else {
    collapsed_shape_ = uniform_list_shape(input_shape.num_samples(), TensorShape<1>{1});
  }

  block_setup_.SetupBlocks(collapsed_shape_, true);
}

void CastGPU::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  kernels::DynamicScratchpad scratchpad({}, ws.stream());
  auto num_samples = collapsed_shape_.num_samples();

  // Get rid of empty samples at the end of the batch
  for (; num_samples > 0 && collapsed_shape_[num_samples - 1][0] == 0; num_samples--) {}


  samples_.resize(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    samples_[sample_id].output = output.raw_mutable_tensor(sample_id);
    samples_[sample_id].input = input.raw_tensor(sample_id);
  }

  std::vector<kernels::CastSampleBlockDesc> params_host(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    params_host[sample_id].sample_size = collapsed_shape_[sample_id][0];
  }

  auto blocks = block_setup_.Blocks();

  // Calculate id of the earliest block that should process given sample
  for (int block_id = 0, sample_id = -1; block_id < blocks.size(); block_id++) {
    // In case of an empty sample, the block descriptor is not generated for it.
    // We mark all the empty samples as using currently selected block_id, and the kernel chooses
    // the latest (rightmost) sample that is marked with its block id for processing.
    while (sample_id < blocks[block_id].sample_idx) {
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
