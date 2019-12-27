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

#include "dali/operators/erase/erase.h"
#include <memory>
#include "dali/operators/erase/erase_utils.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/erase/erase_cpu.h"
#include "dali/kernels/kernel_manager.h"

#define ERASE_SUPPORTED_NDIMS (1, 2, 3, 4, 5)
#define ERASE_SUPPORTED_TYPES (float, uint8)

namespace dali {

DALI_SCHEMA(Erase)
  .DocStr(R"code(TODO....)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<float>("anchor",
    R"code()code",
    vector<float>(), true)
  .AddOptionalArg<float>("shape",
    R"code()code",
    vector<float>(), true)
  .AddOptionalArg("axes",
    R"code(Order of dimensions used for anchor and shape arguments, as dimension indexes)code",
    std::vector<int>{0, 1})
  .AddOptionalArg("axis_names",
    R"code(Order of dimensions used for anchor and shape arguments, as described in layout.
If provided, `axis_names` takes higher priority than `axes`)code",
    "HW")
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

  args_ = detail::GetEraseArgs<T, Dims>(spec_, shape, layout);

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
