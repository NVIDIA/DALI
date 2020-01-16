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

#include <map>
#include <vector>
#include "dali/operators/generic/pad.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cpu.h"

namespace dali {

DALI_SCHEMA(Pad)
  .DocStr(R"code(Pads all samples with `fill_value` in the given `axes`,
to match the size of the biggest dimension on those axes in the batch.
The element padding axes is specified with the argument `axes`.
Supported types are integer and floating point numeric types.

Examples:

- `1-D` samples, `fill_value` = -1, `axes` = (0,)

::

  input  = [[3,   4,   2,   5,   4],
            [2,   2],
            [3, 199,   5]];
  output = [[3,   4,   2,   5,   4],
            [2,   2,  -1,  -1,  -1],
            [3, 199,   5,  -1,  -1]]

- `1-D` samples, `fill_value` = -1, `axes` = (0,), `shape` = (7,)

::

  input  = [[3,   4,   2,   5,   4],
            [2,   2],
            [3, 199,   5]];
  output = [[3,   4,   2,   5,   4,  -1,  -1],
            [2,   2,  -1,  -1,  -1,  -1,  -1],
            [3, 199,   5,  -1,  -1,  -1,  -1]]

- `1-D` samples, `fill_value` = -1, `axes` = (0,), `align` = (4,)

::

  input  = [[3,   4,   2,   5,   4],
            [2,   2],
            [3, 199,   5]];
  output = [[3,   4,   2,   5,   4,  -1,  -1,  -1],
            [2,   2,  -1,  -1,  -1,  -1,  -1,  -1],
            [3, 199,   5,  -1,  -1,  -1,  -1,  -1]]


- `2-D` samples, `fill_value` = 42, `axes` = (1,)

::

  input  = [[[1,  2,  3,  4],
             [5,  6,  7,  8]],
            [[1,  2],
             [4,  5]]]
  output = [[[1,  2,  3,  4],
             [5,  6,  7,  8]],
            [[1,  2, 42, 42],
             [4,  5, 42, 42]]]

- `2-D` samples, `fill_value` = 0, `axes` = (0, 1), `align` = (4, 5)

::

  input  = [[[1,  2,  3,  4],
             [5,  6,  7,  8],
             [9, 10, 11, 12]],
            [[1, 2],
             [4, 5]]]
  output = [[[1,  2,  3,  4,  0],
             [5,  6,  7,  8,  0],
             [9, 10, 11, 12,  0],
             [0,  0,  0,  0,  0]],
            [[1,  2,  0,  0,  0],
             [4,  5,  0,  0,  0],
             [0,  0,  0,  0,  0],
             [0,  0,  0,  0,  0]]]

- `2-D` samples, `fill_value` = 0, `axes` = (0, 1), `align` = (1, 2), `shape` = (4, -1)

::

  input  = [[[1,  2,  3],
             [4,  5,  6]],
            [[1, 2],
             [4, 5],
             [6, 7]]]
  output = [[[1,  2,  3,  0],
             [4,  5,  6,  0],
             [0,  0,  0,  0],
             [0,  0,  0,  0]],
            [[1,  2,  0,  0],
             [4,  5,  0,  0],
             [6,  7,  0,  0],
             [0,  0,  0,  0]]])code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("fill_value",
    R"code(The value to pad the batch with)code",
    0.0f)
  .AddOptionalArg<int>("axes",
    R"code(Indices of the axes on which the batch samples will be padded.
Indexes are zero-based with 0 being the outer-most dimension of the tensor.
Arguments *axis_names* and *axes* are mutually exclusive.
If *axes* and *axis_names* are empty or not provided, the output will be padded on all the axes)code",
    std::vector<int>())
  .AddOptionalArg<TensorLayout>("axis_names",
    R"code(Names of the axes on which the batch samples will be padded.
Dimension names should correspond to dimensions in the input layout.
Arguments *axis_names* and *axes* are mutually exclusive.
If *axes* and *axis_names* are empty or not provided, the output will be padded on all the axes)code",
    "")
  .AddOptionalArg<int>("align",
    R"code(If specified, determines the alignment on those dimensions that are padded. That is,
the padded length on `axis = axes[i]` should be a multiple of `align[i]`. If a single integer value
is provided, the alignment restrictions are applied to all the padded axes.
If a non-negative value for *shape* is specified, the alignment shall be ignored on that axis)code",
    std::vector<int>())
  .AddOptionalArg<int>("shape",
    R"code(The extents of the output shape in the axes specified by *axes* or *axis_names*;
specifying -1 for an axis restores the default behavior of extending the axis to accommodate the
(aligned) size of the largest sample in the batch)code",
    vector<int>());

template <>
bool Pad<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const workspace_t<CPUBackend> &ws) {
  output_desc.resize(1);
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  auto in_layout = input.GetLayout();
  int ndim = in_shape.sample_dim();
  int nsamples = in_shape.num_samples();
  auto nthreads = ws.GetThreadPool().size();

  TYPE_SWITCH(input.type().id(), type2id, T, PAD_SUPPORTED_TYPES, (
    VALUE_SWITCH(ndim, Dims, PAD_SUPPORTED_NDIMS, (
      using Kernel = kernels::SliceFlipNormalizePermutePadCpu<T, T, Dims>;
      using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;

      kmgr_.Resize<Kernel>(nthreads, nsamples);
      output_desc[0].type = TypeInfo::Create<T>();
      output_desc[0].shape.resize(nsamples, Dims);

      auto in_view = view<const T, Dims>(input);
      auto &kernel_sample_args = FillArgs<Args>(in_shape, in_layout);
      for (int i = 0; i < nsamples; i++) {
        auto in_view = view<const T, Dims>(input[i]);
        kernels::KernelContext ctx;
        auto req = kmgr_.Setup<Kernel>(i, ctx, in_view, kernel_sample_args[i]);
        output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", ndim)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void Pad<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  int nsamples = input.size();
  int ndim = input.shape().sample_dim();
  auto& thread_pool = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), type2id, T, PAD_SUPPORTED_TYPES, (
    VALUE_SWITCH(ndim, Dims, PAD_SUPPORTED_NDIMS, (
      using Kernel = kernels::SliceFlipNormalizePermutePadCpu<T, T, Dims>;
      using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;

      for (int i = 0; i < nsamples; i++) {
        thread_pool.DoWorkWithID(
          [this, &input, &output, i](int thread_id) {
            kernels::KernelContext ctx;
            auto in_view = view<const T, Dims>(input[i]);
            auto out_view = view<T, Dims>(output[i]);
            auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
            kmgr_.Run<Kernel>(thread_id, i, ctx, out_view, in_view, kernel_sample_args[i]);
          });
      }
      thread_pool.WaitForWork();
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", ndim)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
}

DALI_REGISTER_OPERATOR(Pad, Pad<CPUBackend>, CPU);

}  // namespace dali
