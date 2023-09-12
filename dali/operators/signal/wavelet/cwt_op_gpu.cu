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
#include "dali/core/dev_buffer.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/signal/wavelet/cwt_args.h"
#include "dali/kernels/signal/wavelet/cwt_gpu.h"
#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"
#include "dali/operators/signal/wavelet/cwt_op.h"
#include "dali/operators/signal/wavelet/wavelet_run.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/op_schema.h"

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
- FBSP - Frequency B-spline wavelet)",
            DALIDataType::DALI_WAVELET_NAME)
    .AddArg("wavelet_args", R"(Additional arguments for mother wavelet. They are passed
as list of float32 values.
- HAAR - none
- GAUS - n (order of derivative)
- MEXH - sigma
- MORL - none
- SHAN - fb (bandwidth parameter > 0), fc (center frequency > 0)
- FBSP - m (order parameter >= 1), fb (bandwidth parameter > 0), fc (center frequency > 0)
)",
            DALIDataType::DALI_FLOAT_VEC);

template <typename T>
struct CwtImplGPU : public OpImplBase<GPUBackend> {
 public:
  using CwtArgs = kernels::signal::CwtArgs<T>;
  using CwtKernel = kernels::signal::CwtGpu<T>;

  template <template <typename> class W>
  using WvltKernel = kernels::signal::WaveletGpu<T, W>;

  explicit CwtImplGPU(CwtArgs args) : args_(std::move(args)) {
    ResizeWaveletKernelForName<T>(args_.wavelet, kmgr_wvlt_);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  dali::kernels::signal::WaveletSpan<T> GetDefaultSpan() {
    dali::kernels::signal::WaveletSpan<T> def_span;
    def_span.begin = -1.0f;
    def_span.end = 1.0f;
    def_span.sampling_rate = 1000;
    return def_span;
  }

 private:
  CwtArgs args_;
  kernels::KernelManager kmgr_wvlt_;
  // std::vector<OutputDesc> wvlt_out_desc_;
  // TensorList<GPUBackend> wvlt_out_;

  // kernels::KernelManager kmgr_cwt_;
  // std::vector<OutputDesc> cwt_out_desc_;
  // TensorList<GPUBackend> cwt_out_;
};

template <typename T>
bool CwtImplGPU<T>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto type = type2id<T>::value;

  dali::kernels::signal::WaveletSpan<T> def_span = GetDefaultSpan();

  TensorListShape<> a_shape(1, {1});
  a_shape.set_tensor_shape(0, {args_.a.size()});
  TensorListView<StorageGPU, const T> a_view = make_tensor_list_gpu((T *)nullptr, a_shape);

  TensorListShape<> b_shape(1, {1});
  b_shape.set_tensor_shape(0, {1});
  TensorListView<StorageGPU, const T> b_view = make_tensor_list_gpu((T *)nullptr, b_shape);

  auto &req = SetupWaveletKernelForName(args_.wavelet, kmgr_wvlt_, ctx, a_view, b_view, def_span,
                                        args_.wavelet_args);

  // wvlt_out_desc_.resize(1);
  // wvlt_out_desc_[0].type = type;
  // wvlt_out_desc_[0].shape = req.output_shapes[0];

  output_desc.resize(1);
  output_desc[0].type = type;
  output_desc[0].shape = req.output_shapes[0];
  return true;
}

template <typename T>
void CwtImplGPU<T>::RunImpl(Workspace &ws) {
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto &output = ws.Output<GPUBackend>(0);

  DeviceBuffer<T> a_buffer;
  a_buffer.from_host(args_.a);
  TensorListShape<> a_shape(1, {1});
  a_shape.set_tensor_shape(0, {args_.a.size()});
  TensorListView<StorageGPU, const T> a_view = make_tensor_list_gpu(a_buffer.data(), a_shape);

  std::vector<T> zero;
  zero.push_back(0);
  DeviceBuffer<T> b_buffer;
  b_buffer.from_host(zero);
  TensorListShape<> b_shape(1, {1});
  b_shape.set_tensor_shape(0, {1});
  TensorListView<StorageGPU, const T> b_view = make_tensor_list_gpu(b_buffer.data(), b_shape);

  dali::kernels::signal::WaveletSpan<T> def_span = GetDefaultSpan();
  auto out_view = view<T>(output);
  RunWaveletKernelForName(args_.wavelet, kmgr_wvlt_, ctx, out_view, a_view, b_view, def_span);
}

template <>
bool Cwt<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  const auto &input = ws.Input<GPUBackend>(0);
  auto type = input.type();
  TYPE_SWITCH(type, type2id, T, (float), (
      using Impl = CwtImplGPU<T>;
      if (!impl_ || type != type_) {
        impl_ = std::make_unique<Impl>(args_);
        type_ = type;
      }
  ), DALI_FAIL(make_string("Unsupported data type: ", type)));  // NOLINT

  impl_->SetupImpl(output_desc, ws);
  return true;
}

template <>
void Cwt<GPUBackend>::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Cwt, Cwt<GPUBackend>, GPU);

}  // namespace dali
