// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/crop/crop_mirror_normalize.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cpu.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(CropMirrorNormalize)
  .DocStr(R"code(Performs fused cropping, normalization, format conversion
(NHWC to NCHW) if desired, and type casting.

Normalization takes the input images and produces the output by using the following formula::

  output = scale * (input - mean) / std + shift

.. note::
    If no cropping arguments are specified, only mirroring and normalization will occur.
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<DALIImageType>("image_type", "Image type", nullptr)
  .DeprecateArg("image_type")  // deprecated since 0.24dev
  .AddOptionalArg("dtype",
       R"code(Output data type.

Supported types: ``FLOAT``, ``FLOAT16``, ``INT8``, ``UINT8``.

If not set, the input type is used.)code", DALI_FLOAT)
  .DeprecateArgInFavorOf("output_dtype", "dtype")  // deprecated since 0.24dev
  .AddOptionalArg("output_layout",
    R"code(Tensor data layout for the output.)code", TensorLayout("CHW"))
  .AddOptionalArg("pad_output",
    R"code(Determines whether to pad the output to the number of channels as a power of 2).)code", false)
  .AddOptionalArg("mirror",
    R"code(If nonzero, the image will be flipped (mirrored) horizontally.)code",
    0, true)
  .AddOptionalArg("mean",
    R"code(Mean pixel values for image normalization.)code",
    std::vector<float>{0.0f})
  .AddOptionalArg("std",
    R"code(Standard deviation values for image normalization.)code",
    std::vector<float>{1.0f})
  .AddOptionalArg("scale", R"(The value by which the result is multiplied.

This argument is useful when using integer outputs to improve dynamic range utilization.)",
    1.0f)
  .AddOptionalArg("shift", R"(The value added to the (scaled) result.

This argument is useful when using unsigned integer outputs to improve dynamic range utilization.)",
    0.0f)
  .AddParent("CropAttr")
  .AddParent("OutOfBoundsAttr");

DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<CPUBackend>, CPU);

template <>
bool CropMirrorNormalize<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                const HostWorkspace &ws) {
  output_desc.resize(1);
  SetupCommonImpl(ws);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int ndim = in_shape.sample_dim();
  int nsamples = in_shape.size();
  auto nthreads = ws.GetThreadPool().NumThreads();
  TYPE_SWITCH(input_type_, type2id, InputType, CMN_IN_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, CMN_OUT_TYPES, (
      VALUE_SWITCH(ndim, Dims, CMN_NDIMS, (
        using Kernel = kernels::SliceFlipNormalizePermutePadCpu<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        output_desc[0].type = TypeInfo::Create<OutputType>();
        output_desc[0].shape.resize(nsamples, Dims);
        kmgr_.Resize<Kernel>(nthreads, nsamples);
        // Do the kernel setup
        auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
          auto in_view = view<const InputType, Dims>(input[sample_idx]);
          kernels::KernelContext ctx;
          auto &req = kmgr_.Setup<Kernel>(sample_idx, ctx, in_view, kernel_sample_args[sample_idx]);
          output_desc[0].shape.set_tensor_shape(sample_idx, req.output_shapes[0][0].shape);
        }
      ), DALI_FAIL(make_string("Not supported number of dimensions:", ndim));); // NOLINT
    ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
  ), DALI_FAIL(make_string("Not supported input type:", input_type_));); // NOLINT

  return true;
}

template <>
void CropMirrorNormalize<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  output.SetLayout(output_layout_);
  auto in_shape = input.shape();
  auto out_shape = output.shape();
  int ndim = in_shape.sample_dim();
  int nsamples = in_shape.size();
  auto& thread_pool = ws.GetThreadPool();
  TYPE_SWITCH(input_type_, type2id, InputType, CMN_IN_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, CMN_OUT_TYPES, (
      VALUE_SWITCH(ndim, Dims, CMN_NDIMS, (
        using Kernel = kernels::SliceFlipNormalizePermutePadCpu<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
        for (int sample_id = 0; sample_id < nsamples; sample_id++) {
          thread_pool.AddWork(
            [this, &input, &output, &kernel_sample_args, sample_id](int thread_id) {
              auto in_view = view<const InputType, Dims>(input[sample_id]);
              auto out_view = view<OutputType, Dims>(output[sample_id]);
              auto &args = kernel_sample_args[sample_id];
              kernels::KernelContext ctx;
              kmgr_.Run<Kernel>(thread_id, sample_id, ctx, out_view, in_view, args);
            }, out_shape.tensor_size(sample_id));
        }
      ), DALI_FAIL(make_string("Not supported number of dimensions:", ndim));); // NOLINT
    ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
  ), DALI_FAIL(make_string("Not supported input type:", input_type_));); // NOLINT
  thread_pool.RunAll();
}

}  // namespace dali
