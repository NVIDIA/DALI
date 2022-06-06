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
#include "dali/kernels/signal/decibel/decibel_calculator.h"
#include "dali/kernels/signal/moving_mean_square_gpu.h"
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace {

/**
 * @brief Threshold predicate for the nonsilent region.
 */
template <typename T>
struct NotLess {
  T threshold;

  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(T x) const noexcept {
    return x >= threshold;
  }
};

using kernels::find_first_last::idx_pair;

/**
 * @brief Coordinate post-processor. Converts first/last to begin/length.
 *        Begin it's calculated as first - window, meaning that we start counting from the
 *        beginning of the sliding window.
 */
template <typename Idx>
struct NonsilentRegion {
  Idx window;

  DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> operator()(idx_pair<Idx> x) const noexcept {
    Idx start = x.a - window;
    return {start, x.b + 1 - start};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> neutral() const noexcept {
    return  idx_pair<Idx>{0, 0};  // empty range
  }
};

struct InitPredicateSampleArgs {
  NotLess<float> *predicate;
  float *ref_pow;
  float cutoff_db;
};

__global__ void InitPredicates(InitPredicateSampleArgs *sample_params, int nsamples) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nsamples;
       idx += blockDim.x * gridDim.x) {
    auto &params = sample_params[idx];
    kernels::signal::DecibelToMagnitude<float> db2mag(10.f, *params.ref_pow);
    auto threshold = db2mag(params.cutoff_db);
    *(params.predicate) = {threshold};
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
    TYPE_SWITCH(dtype, type2id, T, NONSILENCE_TYPES, (
      RunImplTyped<T>(ws);
    ), DALI_FAIL(make_string("Unsupported input type: ", dtype));)  // NOLINT
  }

 private:
  kernels::MaxGPU<float, float> max_kernel;
  using Predicate = NotLess<float>;
  kernels::find_first_last::FindFirstLastGPU<float, int32_t, Predicate, NonsilentRegion<int32_t>>
      find_first_last_;

  /**
   * @brief Computes mean moving mean squre of an audio signal
   */
  template <typename T>
  void CalcMMS(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 1> &mms,
               const TensorListView<StorageGPU, const T, 1> &in) {
    kernels::signal::MovingMeanSquareGpu<T> kernel;
    kernels::signal::MovingMeanSquareArgs args{window_length_, reset_interval_};
    kernel.Run(ctx, mms, in, args);
  }

  /**
   * @brief Calculates the maximum value of an input
   */
  void CalcMax(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 0> &max,
               const TensorListView<StorageGPU, float, 1> &in) {
    std::array<int, 1> axes = { 0 };
    max_kernel.Setup(ctx, in.shape, make_cspan(axes), false, false);
    max_kernel.Run(ctx, max, in);
  }

  /**
   * @brief Calculates the beginning and length of the nonsilent region
   */
  void CalcNonsilentRegion(kernels::KernelContext &ctx,
                           TensorListView<StorageGPU, int32_t, 0> &begin,
                           TensorListView<StorageGPU, int32_t, 0> &len,
                           TensorListView<StorageGPU, float, 1> &mms) {
    int nsamples = mms.num_samples();

    TensorListShape<0> scalar_sh(nsamples);
    auto predicates =
        ctx.scratchpad->AllocTensorList<mm::memory_kind::device, NotLess<float>>(scalar_sh);

    if (!reference_max_) {
      auto predicates_cpu =
        ctx.scratchpad->AllocTensorList<mm::memory_kind::pinned, NotLess<float>>(scalar_sh);
      for (int i = 0; i < nsamples; i++) {
        kernels::signal::DecibelToMagnitude<float> db2mag(10.f, reference_power_[i]);
        (*predicates_cpu.data[i]).threshold = db2mag(cutoff_db_[i]);
      }
      assert(predicates_cpu.is_contiguous());
      assert(predicates.is_contiguous());
      CUDA_CALL(cudaMemcpyAsync(predicates.data[0], predicates_cpu.data[0],
                                sizeof(Predicate) * nsamples, cudaMemcpyHostToDevice,
                                ctx.gpu.stream));
    } else {
      // Predicates need to initialized on the device, because reference power is on the GPU
      // (needs to be calculated as a pre-step)
      auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float>(scalar_sh);
      CalcMax(ctx, max_mms, mms);

      auto init_predicate_args =
          ctx.scratchpad->Allocate<mm::memory_kind::pinned, InitPredicateSampleArgs>(nsamples);

      for (int i = 0; i < nsamples; i++) {
        auto &sample = init_predicate_args[i];
        sample.predicate = predicates.data[i];
        sample.ref_pow = max_mms.data[i];
        sample.cutoff_db = cutoff_db_[i];
      }
      auto init_predicate_args_gpu =
          ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(init_predicate_args, nsamples));

      constexpr int kBlockSize = 256;
      int grid = div_ceil(nsamples, kBlockSize);
      InitPredicates<<<grid, kBlockSize, 0, ctx.gpu.stream>>>(init_predicate_args_gpu, nsamples);
    }

    // Initialize post-processors
    // NonsilentRegion postprocessor converts first/last to begin/length
    // begin is calculated as `first - window`, that is, the beginning of the sliding window for the
    // first sample that goes above the threshold.
    auto postprocessors_cpu =
        ctx.scratchpad->Allocate<mm::memory_kind::pinned, NonsilentRegion<int32_t>>(nsamples);
    for (int i = 0; i < nsamples; i++) {
      postprocessors_cpu[i].window = window_length_;
    }
    auto postprocessors_data =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(postprocessors_cpu, nsamples));
    TensorListView<StorageGPU, NonsilentRegion<int32_t>, 0> postprocessors{postprocessors_data,
                                                                           scalar_sh};

    find_first_last_.Run(ctx, begin, len, mms, predicates, postprocessors);
  }

  template <typename T>
  void RunImplTyped(workspace_t<GPUBackend> &ws) {
    kernels::DynamicScratchpad scratchpad({}, ws.stream());
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    ctx.scratchpad = &scratchpad;

    auto input = view<const T, 1>(ws.template Input<GPUBackend>(0));
    int nsamples = input.shape.num_samples();
    auto out_begin = view<int32_t, 0>(ws.template Output<GPUBackend>(0));
    auto out_len = view<int32_t, 0>(ws.template Output<GPUBackend>(1));

    // 1. Compute MMS
    auto mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 1>(input.shape);
    CalcMMS(ctx, mms, input);

    // 2. Find the non silent region as the begin and length of the region where the energy is above
    // a given value.
    CalcNonsilentRegion(ctx, out_begin, out_len, mms);
  }
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);


}  // namespace dali
