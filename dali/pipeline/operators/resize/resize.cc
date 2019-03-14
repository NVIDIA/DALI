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

#include "dali/pipeline/operators/resize/resize.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddOptionalArg("resize_x", "The length of the X dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_y` is left at 0, then the op will keep "
      "the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_y", "The length of the Y dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_x` is left at 0, then the op will keep "
      "the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_shorter", "The length of the shorter dimension of the resized image. "
      "This option is mutually exclusive with `resize_longer`, `resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_longer", "The length of the longer dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`,`resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image.", 0.f, true);

DALI_SCHEMA(Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .AllowMultipleInputSets()
  .AddOptionalArg("save_attrs",
      R"code(Save reshape attributes for testing.)code", false)
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");

template<>
Resize<CPUBackend>::Resize(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , ResizeAttr(spec)
    , ResizeBase(spec) {
  per_sample_meta_.resize(num_threads_);
  resample_params_.resize(num_threads_);
  out_shape_.resize(num_threads_);
  Initialize(num_threads_);

  save_attrs_ = spec_.HasArgument("save_attrs");
  outputs_per_idx_ = save_attrs_ ? 2 : 1;
}

template <>
void Resize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  const int thread_idx = ws->thread_idx();
  per_sample_meta_[thread_idx] = GetTransfomMeta(ws, spec_);
  resample_params_[thread_idx] = GetResamplingParams(per_sample_meta_[thread_idx]);
}

template <>
void Resize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const int thread_idx = ws->thread_idx();
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(outputs_per_idx_ * idx);

  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");
  DALI_ENFORCE(input.ndim() == 3, "Resize expects 3-dimensional tensor input.");
  if (input.GetLayout() != DALI_UNKNOWN) {
    DALI_ENFORCE(input.GetLayout() == DALI_NHWC,
                 "Resize expects interleaved channel layout (NHWC)");
  }

  RunCPU(output, input, thread_idx);

  if (save_attrs_) {
    auto &attr_output = ws->Output<CPUBackend>(outputs_per_idx_ * idx + 1);
    auto &in_shape = input.shape();

    attr_output.Resize(Dims{2});
    int *t = attr_output.mutable_data<int>();
    t[0] = in_shape[0];
    t[1] = in_shape[1];
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);

}  // namespace dali
