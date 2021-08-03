// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/generic/slice/subscript.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/slice/slice_cpu.h"

namespace dali {

#define INDEX_ARGS(idx) \
     AddOptionalArg<int>("at_" #idx, "Position index", nullptr, true) \
    .AddOptionalArg<int>("lo_" #idx, "Range start", nullptr, true) \
    .AddOptionalArg<int>("hi_" #idx, "Range end", nullptr, true) \
    .AddOptionalArg<int>("step_" #idx, "Range step", nullptr, true)


DALI_SCHEMA(TensorSubscript)
    .MakeDocHidden()
    .DocStr(R"(Applies NumPy-like indexing to a batch of tensors.)")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<int>("num_subscripts",
      "Number of subscripts supplied, including full-range - used for validation only.", nullptr)
    .INDEX_ARGS(0)
    .INDEX_ARGS(1)
    .INDEX_ARGS(2)
    .INDEX_ARGS(3)
    .INDEX_ARGS(4)
    .INDEX_ARGS(5)
    .INDEX_ARGS(6)
    .INDEX_ARGS(7)
    .INDEX_ARGS(8)
    .INDEX_ARGS(9)
    .INDEX_ARGS(10)
    .INDEX_ARGS(11)
    .INDEX_ARGS(12)
    .INDEX_ARGS(13)
    .INDEX_ARGS(14)
    .INDEX_ARGS(15)
    .INDEX_ARGS(16)
    .INDEX_ARGS(17)
    .INDEX_ARGS(18)
    .INDEX_ARGS(19)
    .INDEX_ARGS(20)
    .INDEX_ARGS(21)
    .INDEX_ARGS(22)
    .INDEX_ARGS(23)
    .INDEX_ARGS(24)
    .INDEX_ARGS(25)
    .INDEX_ARGS(26)
    .INDEX_ARGS(27)
    .INDEX_ARGS(28)
    .INDEX_ARGS(29)
    .INDEX_ARGS(30)
    .INDEX_ARGS(31);

template <>
template <int ndim, int element_size>
void TensorSubscript<CPUBackend>::RunTyped(HostWorkspace &ws) {
  auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  int N = input.ntensor();
  using T = kernels::type_of_size<element_size>;
  ThreadPool &tp = ws.GetThreadPool();

  kernels::SliceCPU<T, T, ndim> K;
  TensorView<StorageCPU, const T, ndim> tv_in;
  TensorView<StorageCPU, T, ndim> tv_out;

  kernels::KernelContext ctx;
  for (int i = 0; i < N; i++) {
    tv_in.shape = simplified_in_shape_[i];
    tv_in.data = static_cast<const T*>(input.raw_tensor(i));
    tv_out.shape = simplified_out_shape_[i];
    tv_out.data = static_cast<T*>(output.raw_mutable_tensor(i));
    kernels::SliceArgs<T, ndim> args;
    args.anchor = simplified_anchor_[i].to_static<ndim>();
    args.shape = tv_out.shape;
    K.Schedule(ctx, tv_out, tv_in, args, tp);
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(TensorSubscript, TensorSubscript<CPUBackend>, CPU);

DALI_SCHEMA(SubscriptDimCheck)
    .MakeDocHidden()
    .DocStr(R"(Checks that the input has at least `num_subscripts` dimensions.

This operator is used internally when all indices are empty (:) and just verifieis
that the input has sufficient number of dimensions and passes through the input.)")
    .NumInput(1)
    .NumOutput(1)
    .PassThrough({{0, 0}})
    .AddArg("num_subscripts",
      "Number of subscripts supplied, which is the minimum required in the input.", DALI_INT32);


template <typename Backend>
struct SubscriptDimCheck : public Operator<Backend> {
  explicit SubscriptDimCheck(const OpSpec &spec) : Operator<Backend>(spec) {
    num_subscripts_ = spec.GetArgument<int>("num_subscripts");
  }

  bool SetupImpl(vector<OutputDesc> &desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    auto &in = ws.template InputRef<Backend>(0);
    DALI_ENFORCE(num_subscripts_ <= in.sample_dim(), make_string("Too many indices (",
      num_subscripts_, ") for a ", in.sample_dim(), "-D tensor."));
    auto &out = ws.template OutputRef<Backend>(0);
    out.ShareData(&in);
  }

  int num_subscripts_ = 0;
};

DALI_REGISTER_OPERATOR(SubscriptDimCheck, SubscriptDimCheck<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(SubscriptDimCheck, SubscriptDimCheck<GPUBackend>, GPU);

}  // namespace dali
