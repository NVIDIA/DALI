// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/operators/generic/transpose/transpose.h"
#include "dali/pipeline/data/views.h"

namespace dali {

#define TRANSPOSE_ALLOWED_TYPES                                                                    \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
    double)

class TransposeCPU : public Transpose<CPUBackend> {
 public:
  explicit inline TransposeCPU(const OpSpec &spec) : Transpose(spec) {}

  void RunImpl(HostWorkspace& ws) {
        const auto& input = ws.InputRef<CPUBackend>(0);
    auto& output = ws.OutputRef<CPUBackend>(0);
    output.SetLayout(output_layout_);
    auto& thread_pool = ws.GetThreadPool();
    auto input_type = input.type().id();

    auto out_shape = output.shape();
    int nsamples = out_shape.num_samples();

    TYPE_SWITCH(input_type, type2id, T, TRANSPOSE_ALLOWED_TYPES, (
      for (int i = 0; i < nsamples; i++) {
        thread_pool.AddWork(
          [this, &input, &output, i](int thread_id) {
            TensorShape<> src_ts = input.shape()[i];
            auto dst_ts = permute(src_ts, perm_);
            kernels::TransposeGrouped(
                TensorView<StorageCPU, T>{output[i].mutable_data<T>(), dst_ts},
                TensorView<StorageCPU, const T>{input[i].data<T>(), src_ts}, make_cspan(perm_));
          }, out_shape.tensor_size(i));
      }
    ), DALI_FAIL(make_string("Unsupported input type: ", input_type)));  // NOLINT
    thread_pool.RunAll();
  }
};

DALI_REGISTER_OPERATOR(Transpose, TransposeCPU, CPU);

DALI_SCHEMA(Transpose)
    .DocStr(R"code(Transposes the tensors by reordering the dimensions based on
the ``perm`` parameter.

Destination dimension ``i`` is obtained from source dimension ``perm[i]``.

For example, for a source image with ``HWC`` layout, ``shape = (100, 200, 3)``,
and ``perm = [2, 0, 1]``, it will produce a destination image with ``CHW``
layout and ``shape = (3, 100, 200)``, holding the equality:

.. math:: dst(x_2, x_0, x_1) = src(x_0, x_1, x_2)

which is equivalent to:

.. math:: dst(x_{perm[0]}, x_{perm[1]}, x_{perm[2]}) = src(x_0, x_1, x_2)

for all valid coordinates.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddArg("perm",
            R"code(Permutation of the dimensions of the input, for example, [2, 0, 1].)code",
            DALI_INT_VEC)
    .AddOptionalArg(
        "transpose_layout",
        R"code(When set to True, the axis names in the output data layout are permuted according
to ``perm``, Otherwise, the input layout is copied to the output.

If ``output_layout`` is set, this argument is ignored.)code",
        true)
    .AddOptionalArg(
        "output_layout",
        R"code(Explicitly sets the output data layout.

If this argument is specified, ``transpose_layout`` is ignored.)code",
        TensorLayout(""));

}  // namespace dali
