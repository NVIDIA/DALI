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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/operators/generic/transpose/transpose.h"
#include "dali/kernels/transpose/transpose_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/core/error_handling.h"

namespace dali {

class TransposeGPU : public Transpose<GPUBackend> {
 public:
  using Kernel = kernels::TransposeGPU;

  explicit inline TransposeGPU(const OpSpec &spec) : Transpose(spec) {
    kmgr_.Resize<Kernel>(1, 1);
  }

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  bool SetupImpl(vector<OutputDesc> &descs, const DeviceWorkspace &ws) override {
    Transpose<GPUBackend>::SetupImpl(descs, ws);
    const auto &input = ws.template InputRef<GPUBackend>(0);

    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_.Setup<Kernel>(0, ctx, input.shape(), make_span(perm_), input.type().size());

    return true;
  }

  void RunImpl(DeviceWorkspace &ws) override {
    const auto &input = ws.InputRef<GPUBackend>(0);
    auto &output = ws.OutputRef<GPUBackend>(0);

    output.SetLayout(output_layout_);
    GetData(in_data_, input);
    GetData(out_data_, output);

    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_.Run<Kernel>(0, 0, ctx, out_data_.data(), in_data_.data());
  }

 private:
  void GetData(vector<void *> &data, TensorList<GPUBackend> &tl) {
    int N = tl.ntensor();
    data.resize(N);
    for (int i = 0; i < N; i++) {
      data[i] = tl.raw_mutable_tensor(i);
    }
  }

  void GetData(vector<const void *> &data, const TensorList<GPUBackend> &tl) {
    int N = tl.ntensor();
    data.resize(N);
    for (int i = 0; i < N; i++) {
      data[i] = tl.raw_tensor(i);
    }
  }

  kernels::KernelManager kmgr_;
  vector<const void *> in_data_;
  vector<void *> out_data_;
};

DALI_REGISTER_OPERATOR(Transpose, TransposeGPU, GPU);

}  // namespace dali
