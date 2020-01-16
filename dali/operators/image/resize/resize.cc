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

#include "dali/operators/image/resize/resize.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddOptionalArg("resize_x", R"code(The length of the X dimension of the resized image.
This option is mutually exclusive with `resize_shorter`.
If the `resize_y` is left at 0, then the op will keep
the aspect ratio of the original image.)code", 0.f, true)
  .AddOptionalArg("resize_y", R"code(The length of the Y dimension of the resized image.
This option is mutually exclusive with `resize_shorter`.
If the `resize_x` is left at 0, then the op will keep
the aspect ratio of the original image.)code", 0.f, true)
  .AddOptionalArg("resize_shorter", R"code(The length of the shorter dimension of the resized image.
This option is mutually exclusive with `resize_longer`, `resize_x` and `resize_y`.
The op will keep the aspect ratio of the original image.
The longer dimension can be bounded by setting the `max_size` argument.
See `max_size` argument doc for more info.)code", 0.f, true)
  .AddOptionalArg("resize_longer", R"code(The length of the longer dimension of the resized image.
This option is mutually exclusive with `resize_shorter`,`resize_x` and `resize_y`.
The op will keep the aspect ratio of the original image.)code", 0.f, true)
  .AddOptionalArg("max_size", R"code(Maximum size of the longer dimension when resizing with `resize_shorter`.
When set with `resize_shorter`, the shortest dimension will be resized to `resize_shorter` iff
the longest dimension is smaller or equal to `max_size`. If not, the shortest dimension is resized to
satisfy the constraint ``longest_dim == max_size``.
Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

Example:

Original image = ``400x1200``.

Resized with:

* ``resize_shorter = 200``  (``max_size`` not set) => ``200x600``
* ``resize_shorter = 200``, ``max_size =  400``    => ``132x400``
* ``resize_shorter = 200``, ``max_size = 1000``    => ``200x600``)code", std::vector<float>{0.f, 0.f}, false);

DALI_SCHEMA(Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .InputLayout(0, "HWC")
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
void Resize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace &ws) {
  const int thread_idx = ws.thread_idx();
  per_sample_meta_[thread_idx] = GetTransfomMeta(&ws, spec_);
  resample_params_[thread_idx] = GetResamplingParams(per_sample_meta_[thread_idx]);
}

template <>
void Resize<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  const int thread_idx = ws.thread_idx();
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");
  DALI_ENFORCE(input.ndim() == 3, "Resize expects 3-dimensional tensor input.");
  if (!input.GetLayout().empty()) {
    DALI_ENFORCE(ImageLayoutInfo::IsChannelLast(input.GetLayout()),
                 "Resize expects interleaved channel layout aka (N)HWC");
  }

  RunCPU(output, input, thread_idx);
  output.SetLayout(InputLayout(ws, 0));

  if (save_attrs_) {
    auto &attr_output = ws.Output<CPUBackend>(1);
    auto &in_shape = input.shape();

    attr_output.Resize({2});
    int *t = attr_output.mutable_data<int>();
    t[0] = in_shape[0];
    t[1] = in_shape[1];
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);

}  // namespace dali
