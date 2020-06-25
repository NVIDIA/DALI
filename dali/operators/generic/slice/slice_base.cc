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

#include "dali/operators/generic/slice/slice_base.h"
#include <memory>
#include <queue>
#include <vector>
#include "dali/kernels/slice/slice_cpu.h"

namespace dali {

DALI_SCHEMA(SliceBase)
    .DocStr(R"code(Base implementation for `Slice`, `Crop` and related operators)code")
    .MakeInternal()
    .AddDeprecatedArg("output_dtype", "", "dtype", false)
    .AddOptionalArg("dtype",
       R"code(Output data type. By default same data type as the input will be used.
Supported types: `FLOAT`, `FLOAT16`, and `UINT8`)code",
      DALI_NO_TYPE);


template <typename OutputType, typename InputType, int Dims>
class SliceBaseCpu : public OpImplBase<CPUBackend> {
 public:
  using Kernel = kernels::SliceCPU<OutputType, InputType, Dims>;
  using SliceArgs = kernels::SliceArgs<OutputType, Dims>;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

  std::vector<SliceArgs>& Args() { return args_; }

 private:
  std::vector<SliceArgs> args_;
  kernels::KernelManager kmgr_;

  // Priority queue used to schedule larger samples first to maximize CPU utilization
  // (avoid posting a big sample last and have all the other threads idle while it's processed)
  using VolumeSampleIdPair = std::pair<int64_t, int>;  // volume(out_shape), sample_idx
  using SampleQueue = std::priority_queue<VolumeSampleIdPair, std::vector<VolumeSampleIdPair>>;
  SampleQueue sample_queue_;
};

template <typename OutputType, typename InputType, int Dims>
bool SliceBaseCpu<OutputType, InputType, Dims>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                          const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = in_shape.num_samples();
  auto nthreads = ws.GetThreadPool().size();
  assert(nsamples == static_cast<int>(args_.size()));
  output_desc.resize(1);
  output_desc[0].type = TypeInfo::Create<OutputType>();
  output_desc[0].shape.resize(nsamples, Dims);

  kmgr_.Resize<Kernel>(nthreads, nsamples);
  kernels::KernelContext ctx;
  auto in_view = view<const InputType, Dims>(input);
  assert(sample_queue_.empty());
  for (int i = 0; i < nsamples; i++) {
    auto in_view = view<const InputType, Dims>(input[i]);
    auto req = kmgr_.Setup<Kernel>(i, ctx, in_view, args_[i]);
    auto out_shape = req.output_shapes[0][0].shape;
    output_desc[0].shape.set_tensor_shape(i, out_shape);
    // Larger samples will be scheduled first
    sample_queue_.push({volume(out_shape), i});
  }
  return true;
}

template <typename OutputType, typename InputType, int Dims>
void SliceBaseCpu<OutputType, InputType, Dims>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();
  while (!sample_queue_.empty()) {
    int sample_idx = sample_queue_.top().second;
    sample_queue_.pop();
    thread_pool.DoWorkWithID(
      [this, &input, &output, sample_idx](int thread_id) {
        auto in_view = view<const InputType, Dims>(input[sample_idx]);
        auto out_view = view<OutputType, Dims>(output[sample_idx]);
        kernels::KernelContext ctx;
        kmgr_.Run<Kernel>(thread_id, sample_idx, ctx, out_view, in_view, args_[sample_idx]);
      });
  }
  thread_pool.WaitForWork();
  output.SetLayout(input.GetLayout());
}

template <>
bool SliceBase<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto input_type = input.type().id();
  auto ndim = input.shape().sample_dim();

  if (!impl_ || input_type_ != input_type || ndim != ndim_) {
    impl_.reset();
    input_type_ = input_type;
    ndim_ = ndim;
  }
  auto output_type = output_type_ == DALI_NO_TYPE ? input_type_ : output_type_;

  VALUE_SWITCH(ndim_, Dims, SLICE_DIMS, (
    TYPE_SWITCH(input_type_, type2id, InputType, SLICE_TYPES, (
      if (input_type_ == output_type) {
        using Impl = SliceBaseCpu<InputType, InputType, Dims>;
        if (!impl_)
          impl_ = std::make_unique<Impl>();
        FillArgs(reinterpret_cast<Impl*>(impl_.get())->Args(), ws);
      } else {
        TYPE_SWITCH(output_type, type2id, OutputType, (float, float16, uint8_t), (
          using Impl = SliceBaseCpu<OutputType, InputType, Dims>;
          if (!impl_)
            impl_ = std::make_unique<Impl>();
          FillArgs(reinterpret_cast<Impl*>(impl_.get())->Args(), ws);
        ), DALI_FAIL(make_string("Not supported output type: ", output_type));); // NOLINT
      }
    ), DALI_FAIL(make_string("Not supported input type: ", input_type_)););  // NOLINT
  ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void SliceBase<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

}  // namespace dali
