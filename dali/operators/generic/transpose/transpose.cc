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

#include "dali/kernels/common/transpose.h"
#include "dali/core/static_switch.h"
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
    TYPE_SWITCH(input_type, type2id, T, TRANSPOSE_ALLOWED_TYPES, (
      for (int i = 0; i < batch_size_; i++) {
        thread_pool.DoWorkWithID([this, &input, &output, i](int thread_id) {
          SmallVector<int64_t, transpose_detail::kStaticShapeElements> src_shape;
          transpose_detail::VecInt perm;
          for (auto s : input.shape().tensor_shape(i)) {
            src_shape.push_back(s);
          }
          for (auto p : perm_) {
            perm.push_back(p);
          }
          transpose_detail::PrepareArguments(src_shape, perm);
          auto dst_shape = kernels::Permute(src_shape, perm);
          TensorShape<> src_ts(src_shape.begin(), src_shape.end());
          TensorShape<> dst_ts(dst_shape.begin(), dst_shape.end());
          kernels::TransposeGrouped(TensorView<StorageCPU, T>{output[i].mutable_data<T>(), dst_ts},
                                      TensorView<StorageCPU, const T>{input[i].data<T>(), src_ts},
                                      make_cspan(perm));
      });
    }),
    DALI_FAIL("Input type not supported."));
    thread_pool.WaitForWork();
  }
};

DALI_REGISTER_OPERATOR(Transpose, TransposeCPU, CPU);

DALI_SCHEMA(Transpose)
    .DocStr("Transpose tensor dimension to a new permutated dimension specified by `perm`.")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddArg("perm",
            R"code(Permutation of the dimensions of the input (e.g. [2, 0, 1]).)code", DALI_INT_VEC)
    .AddOptionalArg(
        "transpose_layout",
        R"code(When set to true, the output data layout will be transposed according to perm.
Otherwise, the input layout is copied to the output)code",
        true)
    .AddOptionalArg(
        "output_layout",
        R"code(If provided, sets output data layout, overriding any `transpose_layout` setting)code",
        "");

}  // namespace dali
