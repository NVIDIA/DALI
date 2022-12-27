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

#include "dali/operators/audio/mfcc/mfcc.h"
#include <vector>
#include "dali/kernels/signal/dct/dct_gpu.h"
#include "dali/core/static_switch.h"

namespace dali {

namespace detail {
template <>
DLL_PUBLIC  void LifterCoeffs<GPUBackend>::Calculate(int64_t target_length, float lifter,
                                                     cudaStream_t stream)  {
  // If different lifter argument, clear previous coefficients
  if (lifter_ != lifter) {
    coeffs_.clear();
    lifter_ = lifter;
  }

  // 0 means no liftering
  if (lifter_ == 0.0f)
    return;

  // Calculate remaining coefficients (if necessary)
  if (static_cast<int64_t>(coeffs_.size()) < target_length) {
    int start_idx = coeffs_.size();
    int added_length = target_length - start_idx;
    coeffs_.resize(target_length, stream);
    std::vector<float> new_coeffs(added_length);
    CalculateCoeffs(new_coeffs.data(), start_idx, added_length);
    CUDA_CALL(
      cudaMemcpyAsync(&coeffs_.data()[start_idx], new_coeffs.data(), added_length * sizeof(float),
                      cudaMemcpyHostToDevice, stream));
  }
}

template <typename T>
std::vector<OutputDesc> SetupKernel(kernels::KernelManager &kmgr, kernels::KernelContext &ctx,
                                    const TensorList<GPUBackend> &input,
                                    span<const MFCC<GPUBackend>::DctArgs> args, int axis) {
  using Kernel = kernels::signal::dct::Dct1DGpu<T>;
  kmgr.Resize<Kernel>(1);
  auto in_view = view<const T>(input);
  auto &req = kmgr.Setup<Kernel>(0, ctx, in_view, args, axis);
  return {{req.output_shapes[0], input.type()}};
}

}  // namespace detail

template<>
bool MFCC<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                 const Workspace &ws) {
  GetArguments(ws);
  ctx_.gpu.stream = ws.stream();
  auto &input = ws.Input<GPUBackend>(0);

  auto in_shape = input.shape();
  int ndim = in_shape.sample_dim();
  DALI_ENFORCE(axis_ >= 0 && axis_ < ndim,
               make_string("Axis ", axis_, " is out of bounds [0,", ndim, ")"));

  TYPE_SWITCH(input.type(), type2id, T, MFCC_SUPPORTED_TYPES, (
    output_desc = detail::SetupKernel<T>(kmgr_, ctx_, input, make_cspan(args_), axis_);
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
  int64_t max_ndct = 0;
  for (int i = 0; i < output_desc[0].shape.num_samples(); ++i) {
    int64_t ndct = output_desc[0].shape[i][axis_];
    if (ndct > max_ndct)
      max_ndct = ndct;
  }
  lifter_coeffs_.Calculate(max_ndct, lifter_, ws.stream());
  return true;
}

template<>
void MFCC<GPUBackend>::RunImpl(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, T, MFCC_SUPPORTED_TYPES, (
    using Kernel = kernels::signal::dct::Dct1DGpu<T>;
    auto in_view = view<const T>(input);
    auto out_view = view<T>(ws.Output<GPUBackend>(0));
    auto lifter_view = make_tensor_gpu<1>(lifter_coeffs_.data(),
                                          {static_cast<int64_t>(lifter_coeffs_.size())});
    kmgr_.Run<Kernel>(0, ctx_, out_view, in_view, lifter_view);
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
}

DALI_REGISTER_OPERATOR(MFCC, MFCC<GPUBackend>, GPU);

}  // namespace dali
