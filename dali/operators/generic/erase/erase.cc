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

#include "dali/operators/generic/erase/erase.h"
#include <memory>
#include "dali/operators/generic/erase/erase_utils.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/erase/erase_cpu.h"
#include "dali/kernels/kernel_manager.h"

#define ERASE_SUPPORTED_NDIMS (1, 2, 3, 4, 5)
#define ERASE_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                               uint64_t, int64_t, float, float16)

namespace dali {

DALI_SCHEMA(Erase)
  .DocStr(R"code(Erases one or multiple regions from the image.

The region is specified by an *anchor* (starting point) and a *shape* (dimensions).
Only the relevant dimensions are specified.
Non-specified dimensions are treated as if the whole range of the axis was provided.
To specify multiple regions, *anchor* and *shape* represent multiple points consecutively
(e.g. *anchor* = (y0, x0, y0, y1,...) and *shape* = (h0, w0, h1, w1,...)).
The arguments anchor and shape are interpreted according to the value of the argument *axis_names*
(or alternatively the value of the argument *axes*). If no *axis_names*/*axes* arguments are provided,
all the dimensions except 'C' (channels) must be specified.

**Example 1:**

*anchor* = (10, 20), *shape* = (190, 200), *axis_names* = "HW", *fill_value* = 0

input: *layout* = "HWC", *shape* = (300, 300, 3)

The erase region covers the range from 10 to 200 in the vertical dimension (heigth) and goes from
20 to 220 in the horizontal dimension (width). The range for the channel dimension goes from 0 to 3, as it
was not specified. That is::

    output[y, x, c] = 0               if 20 <= x < 220 and 10 <= y < 200
    output[y, x, c] = input[y, x, c]  otherwise

**Example 2:**

*anchor* = (10, 250), *shape* = (20, 30), *axis_names* = "W", *fill_value* = (118, 185, 0)

input: *layout* = "HWC", *shape* = (300, 300, 3)

Two erase regions are provided, covering two vertical bands ranging from x=(10, 30) and x=(250, 280) respectively.
Each pixel in the erased regions is filled with a multi-channel value (118, 185, 0). That is::

    output[y, x, :] = (118, 185, 0)   if 10 <= x < 30 or 250 <= x < 280
    output[y, x, :] = input[y, x, :]  otherwise

**Example 3:**

*anchor* = (0.15, 0.15), *shape* = (0.3, 0.3), *axis_names* = "HW", *fill_value* = 100, *normalized* = True

input: *layout* = "HWC", *shape* = (300, 300, 3)

One erase region with normalized coordinates in the height and width dimensions is provided. A single fill value is provided
for all the channels. The coordinates can be transformed to the absolute by multiplying by the input shape. That is::

    output[y, x, c] = 100             if 0.15 * 300 <= x < (0.3 + 0.15) * 300 and 0.15 * 300 <= y < (0.3 + 0.15) * 300
    output[y, x, c] = input[y, x, c]  otherwise

**Example 4:**

*anchor* = (0.15, 0.15), *shape* = (20, 30), *normalized_anchor* = True, *normalized_shape* = False

input: *layout* = "HWC", *shape* = (300, 300, 3)

One erase region with an anchor specified in normalized coordinates and shape in absolute coordinates.
Since no *axis_names* is provided, the *anchor* and *shape* must contain all dimensions except 'C' (channels)"::

    output[y, x, c] = 0               if 0.15 * 300 <= x < (0.15 * 300) + 20 and (0.15 * 300) <= y < (0.15 * 300) + 30
    output[y, x, c] = input[y, x, c]  otherwise
)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<float>("anchor",
    R"code(Coordinates for anchor or starting point of the erase region. Only the coordinates of the relevant dimensions
(specified by *axis_names* or *axes*) should be provided.)code",
    vector<float>(), true)
  .AddOptionalArg<float>("shape",
    R"code(Values for shape or dimensions of the erase region. Only the coordinates of the relevant dimensions
(specified by *axis_names* or *axes*) should be provided.)code",
    vector<float>(), true)
  .AddOptionalArg("axes",
    R"code(Order of dimensions used for *anchor* and *shape* arguments, as dimension indexes. For instance, axes=(1, 0)
means the coordinates in anchor and *shape* refer to axes 1 and 0, in that particular order)code",
    std::vector<int>{1, 0})
  .AddOptionalArg("axis_names",
    R"code(Order of dimensions used for anchor and shape arguments, as described in the layout.
For instance, axis_names="HW" means that the coordinates in *anchor* and *shape* refer to dimensions H (heigth)
and W (width), in that particular order.
If provided, *axis_names* takes higher priority than *axes*)code",
    "HW")
  .AddOptionalArg("fill_value",
    R"code(Value to fill the erased region. Might be specified as a single value (e.g. 0) or a multi-channel value
(e.g. (200, 210, 220)). If a multi-channel fill value is provided, the input layout should contain a channel dimension 'C')code",
    std::vector<float>{0, })
  .AddOptionalArg("normalized_anchor",
    R"code(Whether or not the anchor argument should be interpreted as normalized (range [0.0, 1.0])
or absolute coordinates. It is mutually exclusive with providing a value for *normalized*)code",
    false)
  .AddOptionalArg("normalized_shape",
    R"code(Whether or not the shape argument should be interpreted as normalized (range [0.0, 1.0])
or absolute coordinates. It is mutually exclusive with providing a value for *normalized*.)code",
    false)
  .AddOptionalArg("normalized",
    R"code(Whether or not the anchor and shape arguments should be interpreted as normalized (range [0.0, 1.0])
or absolute coordinates. It is mutually exclusive with providing a value for *normalized_shape* and
*normalized_anchor* separately.)code",
    false)
  .AddOptionalArg("centered_anchor",
    R"code(If True, the anchors refer to the center of the region instead of the top-left corner,
resulting in centered erased regions at the specified *anchor*.)code",
    false)
  .AllowSequences()
  .SupportVolumetric();

template <typename T, int Dims>
class EraseImplCpu : public detail::OpImplBase<CPUBackend> {
 public:
  using EraseKernel = kernels::EraseCpu<T, Dims>;

  explicit EraseImplCpu(const OpSpec& spec)
      : spec_(spec), batch_size_(spec.GetArgument<int>("batch_size")) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  OpSpec spec_;
  int batch_size_ = 0;
  std::vector<int> axes_;
  std::vector<kernels::EraseArgs<T, Dims>> args_;
  kernels::KernelManager kmgr_;
};

template <typename T, int Dims>
bool EraseImplCpu<T, Dims>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto layout = input.GetLayout();
  auto type = input.type();
  auto shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().size();

  args_ = detail::GetEraseArgs<T, Dims>(spec_, ws, shape, layout);

  kmgr_.Initialize<EraseKernel>();
  kmgr_.Resize<EraseKernel>(nthreads, nsamples);
  // the setup step is not necessary for this kernel (output and input shapes are the same)

  output_desc.resize(1);
  output_desc[0] = {shape, input.type()};
  return true;
}


template <typename T, int Dims>
void EraseImplCpu<T, Dims>::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();

  for (int i = 0; i < nsamples; i++) {
    thread_pool.DoWorkWithID(
      [this, &input, &output, i](int thread_id) {
        kernels::KernelContext ctx;
        auto in_view = view<const T, Dims>(input[i]);
        auto out_view = view<T, Dims>(output[i]);
        kmgr_.Run<EraseKernel>(thread_id, i, ctx, out_view, in_view, args_[i]);
      });
  }
  thread_pool.WaitForWork();
}

template <>
bool Erase<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                  const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  TYPE_SWITCH(input.type().id(), type2id, T, ERASE_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, ERASE_SUPPORTED_NDIMS, (
      impl_ = std::make_unique<EraseImplCpu<T, Dims>>(spec_);
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  assert(impl_ != nullptr);
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void Erase<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Erase, Erase<CPUBackend>, CPU);

}  // namespace dali
