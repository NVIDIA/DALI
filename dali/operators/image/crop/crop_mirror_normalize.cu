// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/image/crop/crop_mirror_normalize.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
bool CropMirrorNormalize<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                const Workspace &ws) {
  auto curr_batch_size = ws.GetInputBatchSize(0);
  output_desc.resize(1);
  SetupCommonImpl(ws);
  const auto &input = ws.Input<GPUBackend>(0);
  int ndim = input.shape().sample_dim();
  TYPE_SWITCH(input_type_, type2id, InputType, CMN_IN_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, CMN_OUT_TYPES, (
      VALUE_SWITCH(ndim, Dims, CMN_NDIMS, (
        using Kernel = kernels::SliceFlipNormalizePermutePadGpu<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        auto &kernel_sample_args = std::any_cast<std::vector<Args>&>(kernel_sample_args_);
        output_desc[0].type = output_type_;
        output_desc[0].shape.resize(curr_batch_size, Dims);
        kmgr_.Resize<Kernel>(1);

        kernels::KernelContext ctx;
        ctx.gpu.stream = ws.stream();
        auto in_view = view<const InputType, Dims>(input);
        auto &req = kmgr_.Setup<Kernel>(0, ctx, in_view, kernel_sample_args);
        output_desc[0].shape = req.output_shapes[0];
      ), DALI_FAIL(make_string("Not supported number of dimensions:", ndim));); // NOLINT
    ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
  ), DALI_FAIL(make_string("Not supported input type:", input_type_));); // NOLINT
  return true;
}

template<>
void CropMirrorNormalize<GPUBackend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(output_layout_);
  int ndim = input.shape().sample_dim();
  TYPE_SWITCH(input_type_, type2id, InputType, CMN_IN_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, CMN_OUT_TYPES, (
      VALUE_SWITCH(ndim, Dims, CMN_NDIMS, (
        using Kernel = kernels::SliceFlipNormalizePermutePadGpu<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        auto in_view = view<const InputType, Dims>(input);
        auto out_view = view<OutputType, Dims>(output);
        kernels::KernelContext ctx;
        ctx.gpu.stream = ws.stream();
        auto &kernel_sample_args = std::any_cast<std::vector<Args>&>(kernel_sample_args_);
        kmgr_.Run<Kernel>(0, ctx, out_view, in_view, kernel_sample_args);
      ), DALI_FAIL(make_string("Not supported number of dimensions:", ndim));); // NOLINT
    ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
  ), DALI_FAIL(make_string("Not supported input type:", input_type_));); // NOLINT
}

}  // namespace dali
