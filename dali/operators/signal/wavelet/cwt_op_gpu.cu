// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/signal/wavelets/cwt_args.h"
#include "dali/kernels/signal/wavelets/cwt_gpu.h"
#include "dali/operators/signal/wavelets/cwt_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Cwt)
  .DocStr(R"(Performs continuous wavelet transform on a 1D signal (for example, audio).

Result values of transform are computed for all specified scales.
Input data is expected to be one channel (shape being ``(nsamples,)``, ``(nsamples, 1)``
) of type float32.)")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("a", R"(List of scale coefficients of type float32.)", DALIDataType::DALI_FLOAT_VEC)
  .AddArg("wavelet", R"(Name of mother wavelet. Currently supported wavelets' names are:
- HAAR - Haar wavelet
- GAUS - Gaussian wavelet
- MEXH - Mexican hat wavelet
- MORL - Morlet wavelet
- SHAN - Shannon wavleet
- FBSP - Frequency B-spline wavelet)", DALIDataType::DALI_WAVELET_NAME)
  .AddArg("wavelet_args", R"(Additional arguments for mother wavelet. They are passed
as list of float32 values.
- HAAR - none
- GAUS - n (order of derivative)
- MEXH - sigma
- MORL - none
- SHAN - fb (bandwidth parameter > 0), fc (center frequency > 0)
- FBSP - m (order parameter >= 1), fb (bandwidth parameter > 0), fc (center frequency > 0)
)", DALIDataType::DALI_FLOAT_VEC);

template <typename T>
struct CwtImplGPU : public OpImplBase<GPUBackend> {
 public:
  using CwtArgs = kernels::signal::wavelets::CwtArgs<T>;
  using CwtKernel = kernels::signal::wavelets::CwtGpu<T>;

  explicit CwtImplGPU(CwtArgs args) : args_(std::move(args)) {
    kmgr_cwt_.Resize<CwtKernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto in_view = view<const T>(input);

    auto type = type2id<T>::value;

    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();

    auto &req = kmgr_cwt_.Setup<CwtKernel>(0, ctx, in_view);
    output_desc.resize(1);
    output_desc[0].type = type;
    output_desc[0].shape = req.output_shapes[0];

    return true;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);

    auto in_view = view<const T>(input);
    auto out_view = view<T>(output);

    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();

    kmgr_cwt_.Run<CwtKernel>(0, ctx, out_view, in_view, args_);
  }

 private:
  CwtArgs args_;
  kernels::KernelManager kmgr_cwt_;
  std::vector<OutputDesc> cwt_out_desc_;
  TensorList<GPUBackend> cwt_out_;
};

DALI_REGISTER_OPERATOR(Cwt, Cwt<GPUBackend>, GPU);

}  // namespace dali
