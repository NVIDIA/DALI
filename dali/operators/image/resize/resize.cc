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
namespace detail {
  kernels::ResamplingParams2D GetResamplingParams(
    const TransformMeta &meta, kernels::FilterDesc min_filter, kernels::FilterDesc mag_filter) {
    kernels::ResamplingParams2D params;
    params[0].output_size = meta.rsz_h;
    params[1].output_size = meta.rsz_w;
    params[0].min_filter = params[1].min_filter = min_filter;
    params[0].mag_filter = params[1].mag_filter = mag_filter;
    return params;
  }
}  // namespace detail

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
  .InputLayout(0, {"HWC", "FHWC" /*, "DHWC", "FDHWC" */ })
  .AddOptionalArg("save_attrs",
      R"code(Save reshape attributes for testing.)code", false)
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");

template<>
Resize<CPUBackend>::Resize(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , ResizeAttr(spec)
    , ResizeBase<CPUBackend>(spec) {
  per_sample_meta_.resize(num_threads_);
  resample_params_.resize(num_threads_);
  InitializeCPU(num_threads_);

  save_attrs_ = spec_.HasArgument("save_attrs");
}

template <>
void Resize<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const int thread_idx = ws.thread_idx();
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);

  RunResize(ws, output, input);
  output.SetLayout(input.GetLayout());

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
