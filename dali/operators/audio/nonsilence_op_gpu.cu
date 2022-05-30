// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/common/find/find_first_last_gpu.cuh"
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"
#include "dali/kernels/signal/moving_mean_square_gpu.h"
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace {

struct threshold_cutoff_db {
  float thresh_;

  DALI_HOST_DEV DALI_FORCEINLINE threshold_cutoff_db() = default;
  DALI_HOST_DEV DALI_FORCEINLINE explicit threshold_cutoff_db(float cutoff_db, float ref) {
    kernels::signal::DecibelToMagnitude<float> db2mag(10.f, ref);
    thresh_ = db2mag(cutoff_db);
  }

  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(float x) const noexcept {
    return x >= thresh_;
  }
};

template <typename Idx>
struct nonsilent_region_fmt {
  int offset;  // adds win_len - 1 to the length because we are not sure where exactly within the
               // window the silence starts
  DALI_HOST_DEV DALI_FORCEINLINE nonsilent_region_fmt() = default;
  DALI_HOST_DEV DALI_FORCEINLINE explicit nonsilent_region_fmt(int mms_window_len)
      : offset(mms_window_len - 1) {}

  using Pair = using kernels::find_first_last::pair_idx<Idx>;
  DALI_HOST_DEV DALI_FORCEINLINE Pair operator()(Pair x) const noexcept {
    auto len = (x.b + 1) - x.a;  // last + 1 - b
    return {len + offset};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE Pair neutral() const noexcept {
    return {0, 0};  // empty range
  }
};

using FindBeginLengthSampleDesc =
    kernels::find_first_last::SampleDesc<float, int32_t, threshold_cutoff_db,
                                         nonsilent_region_fmt<int32_t>>;

__global__ void InitPredicates(FindBeginLengthSampleDesc *sample_descs_gpu, float *ref_pow,
                               float *cutoff_db, int nsamples) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nsamples;
       idx += blockDim.x * gridDim.x) {
    sample_descs_gpu[idx].predicate = threshold_cutoff_db(cutoff_db[idx], ref_pow[idx]);
  }
}

}  // namespace

class NonsilenceOperatorGpu : public NonsilenceOperator<GPUBackend> {
 public:
  explicit NonsilenceOperatorGpu(const OpSpec &spec) :
          NonsilenceOperator<GPUBackend>(spec) {}

  ~NonsilenceOperatorGpu() override = default;
  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorGpu);

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override {
    auto dtype = ws.template Input<GPUBackend>(0).type();
    // TYPE_SWITCH(dtype, type2id, T, NONSILENCE_TYPES, (
      RunImplTyped<float>(ws);
   // ), DALI_FAIL(make_string("Unsupported input type: ", dtype));)  // NOLINT
  }

 private:
  kernels::MaxGPU<float, float> max_kernel;
  kernels::find_first_last::FindFirstLastGPU find_first_last_kernel;

  template <typename T>
  void CalcMMS(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 1> &mms,
               const TensorListView<StorageGPU, const T, 1> &in) {
    kernels::signal::MovingMeanSquareGpu<T> kernel;
    kernels::signal::MovingMeanSquareArgs args{window_length_, reset_interval_};
    kernel.Run(ctx, mms, in, args);
  }

  void CalcMax(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 0> &max,
               const TensorListView<StorageGPU, float, 1> &in) {
    std::array<int, 1> axes = { 0 };
    max_kernel.Setup(ctx, in.shape, make_cspan(axes), false, false);
    max_kernel.Run(ctx, max, in);
  }

  void CalcNonsilentRegion(kernels::KernelContext &ctx,
                           TensorListView<StorageGPU, int32_t, 0> &begin,
                           TensorListView<StorageGPU, int32_t, 0> &len,
                           TensorListView<StorageGPU, float, 1> &mms) {
    int nsamples = mms.num_samples();
    auto sample_descs_cpu =
        ctx.scratchpad->Allocate<mm::memory_kind::pinned, FindBeginLengthSampleDesc>(nsamples);
    for (int i = 0; i < nsamples; i++) {
      auto &sample = sample_descs_cpu[i];
      sample.a_ptr = begin[i].data;
      sample.b_ptr = len[i].data;
      sample.in = mms[i].data;
      sample.len = mms[i].shape[0];
      sample.format = nonsilent_region_fmt<int32_t>(window_length_);
    }

    if (!reference_max_) {  // otherwise it gets initialized by InitPredicates
      for (int i = 0; i < nsamples; i++) {
        sample_descs_cpu[i].predicates = threshold_cutoff_db(cutoff_db_[i], reference_power_[i]);
      }
    }
    auto sample_descs_gpu = ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

    if (reference_max_) {  // need to initialize predicates if using max as reference
      TensorListShape<0> scalar(nsamples);
      auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
      CalcMax(ctx, max_mms, mms);

      auto cutoff_db = ctx.scratchpad->Allocate<mm::memory_kind::pinned, float>(nsamples);
      for (int i = 0; i < nsamples; i++)
        cutoff_db[i] = cutoff_db_[i];
      auto cutoff_db_gpu = ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(cutoff_db, nsamples));

      static constexpr int kBlockSize = 256;
      int grid = div_ceil(nsamples, kBlockSize);
      InitPredicates<<<grid, kBlockSize, 0, ctx.gpu.stream>>>(sample_descs_gpu, max_mms,
                                                              cutoff_db_gpu, nsamples);
    }

    find_first_last_kernel.template Run(ctx, sample_descs_gpu, nsamples);
  }

  template <typename T>
  void RunImplTyped(workspace_t<GPUBackend> &ws) {
    kernels::DynamicScratchpad scratchpad({}, ws.stream());
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    ctx.scratchpad = &scratchpad;

    auto input = view<const T, 1>(ws.template Input<GPUBackend>(0));
    int nsamples = input.shape.num_samples();
    auto out_begin = view<int32_t>(ws.template Output<GPUBackend>(0));
    auto out_end = view<int32_t>(ws.template Output<GPUBackend>(1));

    // 1. Compute MMS
    auto mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 1>(input.shape);
    CalcMMS(ctx, mms, input);

    // 2. Find the region that above or equal to the cutoff_db
    CalcNonsilentRegion(ctx, out_begin, out_end, mms);
  }
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);


}  // namespace dali
