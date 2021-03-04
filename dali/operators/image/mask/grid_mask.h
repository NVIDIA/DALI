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

#ifndef DALI_OPERATORS_IMAGE_MASK_GRID_MASK_H_
#define DALI_OPERATORS_IMAGE_MASK_GRID_MASK_H_

#include <vector>
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class GridMask : public Operator<Backend> {
 public:
  explicit GridMask(const OpSpec &spec) : Operator<Backend>(spec) { }

 protected:
  bool CanInferOutputs() const override { return true; }
  void GetArguments(const workspace_t<Backend> &ws) {
    int batch_size = ws.GetInputBatchSize(0);
    this->GetPerSampleArgument(tile_, "tile", ws, batch_size);
    this->GetPerSampleArgument(ratio_, "ratio", ws, batch_size);
    this->GetPerSampleArgument(angle_, "angle", ws, batch_size);
    this->GetPerSampleArgument(shift_x_, "shift_x", ws, batch_size);
    this->GetPerSampleArgument(shift_y_, "shift_y", ws, batch_size);
    for (auto t : tile_)
      DALI_ENFORCE(t > 0, "Tile argument must be positive");
  }
  std::vector<int> tile_;
  std::vector<float> ratio_;
  std::vector<float> angle_;
  std::vector<float> shift_x_;
  std::vector<float> shift_y_;
  kernels::KernelManager kernel_manager_;
};


class GridMaskCpu : public GridMask<CPUBackend> {
 public:
  explicit GridMaskCpu(const OpSpec &spec) : GridMask(spec) { }
  using Operator<CPUBackend>::RunImpl;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;
};

class GridMaskGpu : public GridMask<GPUBackend> {
 public:
  explicit GridMaskGpu(const OpSpec &spec) : GridMask(spec) { }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<GPUBackend> &ws) override;
  void RunImpl(workspace_t<GPUBackend> &ws) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_MASK_GRID_MASK_H_
