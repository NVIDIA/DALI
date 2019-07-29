// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_

#include "dali/pipeline/operators/displacement/new_warp.h"
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/alloc.h"

namespace dali {

template <typename Derived>
class NewWarp<GPUBackend> : public Operator<GPUBackend>
 public:
  using MyType = Derived;
  constexpr MyType &This() { return static_cast<MyType&>(*this); }
  constexpr const MyType &This() const { return static_cast<MyType&>(*this); }

  using Backend = GPUBackend;
  using Workspace = DeviceWorkspace;

  void RunImpl(Workspace* ws, const int idx) override;

  bool InferOutputs(
      std::vector<kernels::TensorListShape<>> &shapes,
      std::vector<TypeInfo> &types, DeviceWorkspace &ws) {
    shapes.resize(1);
    types.resize(1);

    SetupMappingParams(ws);
    Setup(shapes[0], types[0], ws);
    return true;
  }


  virtual void Setup(kernels::TensorListShape<> &shape,
                     DALIDataType &type,
                     DeviceWorkspace &ws) = 0;

  DALIDataType input_type;
  DALIDataType output_type;

  void RunImpl(DeviceWorkspace* ws, const int idx) {
    InferOutputs(shapes);

    auto input = ws->Input<GPUBackend>(0);
  }

  template <typename OutputType, typename InputType>
  void Run(TensorListView<OutputType> &output);

 protected:
  kernels::KernelManager kmgr_;
  Tensor<GPUBackend> params_gpu_;
  bool has_scalar_params_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_IMPL_CUH_
