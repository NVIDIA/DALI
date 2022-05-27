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

__global__
void InitPredicates(threshold_cutoff_db *predicates, float *ref_pow, float *cutoff_db, int N) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < N;
       idx += blockDim.x * gridDim.x) {
    predicates[idx] = threshold_cutoff_db(cutoff_db[idx], ref_pow[idx]);
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
      // RunImplTyped<float>(ws);
   // ), DALI_FAIL(make_string("Unsupported input type: ", dtype));)  // NOLINT
  }

 private:
  kernels::MaxGPU<float, float> max_kernel;
  kernels::find::FindFirstLastGPU find_first_last_kernel;

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

  // void CalcNonsilentRegionRefMax(kernels::KernelContext &ctx,
  //                                TensorListView<StorageGPU, int32_t, 0> &begin,
  //                                TensorListView<StorageGPU, int32_t, 0> &len,
  //                                TensorListView<StorageGPU, float, 1> &mms,
  //                                span<float> cutoff_db, span<float> ref_pow = {}) {
  //   int nsamples = mms.num_samples();
  //   TensorListShape<0> scalar(nsamples);

  //   using Predicate = threshold_cutoff_db;

  //   if (ref_pow.empty()) {
  //     auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
  //     CalcMax(ctx, max_mms, mms);

  //     // InitPredicates takes contiguous mem
  //     // Sanity check that AllocTensorList allocated a contiguous chunk
  //     assert(max_mms.is_contiguous());
  //     float* max_mms_data = max_mms[0].data;

  //     Predicate* predicates = ctx.scratchpad->Allocate<mm::memory_kind::device, Predicate>(nsamples);
  //     static constexpr int kBlockSize = 256;
  //     int grid = div_ceil(nsamples, kBlockSize);
  //     InitPredicates<<<grid, kBlockSize, 0, ctx.cuda.stream>>>(predicates, max_mms_data, cutoff_db.data(), nsamples);
  //   }



  //   using OutFormat = kernels::find::begin_length<int32_t>;
  //   find_first_last_kernel.template Run<float, int32_t, Predicate, OutFormat>(
  //       ctx, begin, len, mms, predicates);
  // }

  // void CalcNonsilentRegionRefConstant(kernels::KernelContext &ctx,
  //                                     TensorListView<StorageGPU, int32_t, 0> &begin,
  //                                     TensorListView<StorageGPU, int32_t, 0> &len,
  //                                     TensorListView<StorageGPU, float, 1> &mms,
  //                                     float ref_pow) {
  //   int nsamples = mms.num_samples();
  //   TensorListShape<0> scalar(nsamples);
  //   auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
  //   CalcMax(ctx, max_mms, mms);

  //   using Predicate = threshold_val<float>;
  //   // no need for it to be pinned, since we copy those predicates to sample descriptors (pinned) in
  //   // the kernel
  //   span<Predicate> predicates{ctx.scratchpad->Allocate<mm::memory_kind::host, Predicate>(nsamples),
  //                              nsamples};
  //   for (int i = 0; i < nsamples; i++) {
  //     predicates[i].value_ = ref_pow;
  //   }

  //   using OutFormat = kernels::find::begin_length<int32_t>;
  //   find_first_last_kernel.template Run<float, int32_t, Predicate, OutFormat>(ctx, begin, len, mms,
  //                                                                             predicates);
  // }


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

    auto mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 1>(input.shape);
    CalcMMS(ctx, mms, input);

    // if (reference_max_) {
    //   TensorListShape<0> scalar(nsamples);
    //   auto max_mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
    //   CalcMax(ctx, max_mms, mms);
    // }
  }
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);


}  // namespace dali
