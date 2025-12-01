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

#include "dali/kernels/reduce/find_region.cuh"
#include "dali/kernels/reduce/reduce_gpu.h"
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
  T value_;

  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(T x) const noexcept {
    return x >= value_;
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
    *(params.predicate) = {db2mag(params.cutoff_db)};
  }
}

template <typename Idx>
struct NonsilentRegionPostprocessArgs {
  Idx *begin;
  Idx *len;
  const i64vec2 *region;
  Idx input_len;
};

template <typename Idx = int64_t>
__global__ void NonsilentRegionPostprocess(NonsilentRegionPostprocessArgs<Idx> *args, int nsamples,
                                           int window) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nsamples;
       idx += blockDim.x * gridDim.x) {
    auto &sample_args = args[idx];
    auto region = *(sample_args.region);
    // `begin` is calculated as `first - window + 1`, meaning that we start counting from the
    // beginning of the sliding window. The length of the region needs to be extended by the same
    // amount. We also limit the region to the bounds of the input.
    Idx start = cuda_max<Idx>(region.x - window + 1, 0);
    Idx end = region.y;
    *(sample_args.begin) = start;
    *(sample_args.len) = end - start;
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
  void RunImpl(Workspace &ws) override {
    auto dtype = ws.Input<GPUBackend>(0).type();
    TYPE_SWITCH(dtype, type2id, T, (NONSILENCE_TYPES),
      (RunImplTyped<T>(ws);),
      (DALI_FAIL(
          make_string("Unsupported input type: ", dtype,
                      "\nSupported types are : ", ListTypeNames<NONSILENCE_TYPES>()));));
  }

 private:
  kernels::MaxGPU<float, float> max_kernel_;
  using Predicate = NotLess<float>;
  kernels::FindRegionGPU<float, Predicate> find_region_;

  /**
   * @brief Computes mean moving mean square of an audio signal
   */
  template <typename T>
  void CalcMMS(TensorListView<StorageGPU, float, 1> &mms,
               const TensorListView<StorageGPU, const T, 1> &in,
               cudaStream_t stream) {
    kernels::DynamicScratchpad scratchpad(stream);
    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    ctx.scratchpad = &scratchpad;

    kernels::signal::MovingMeanSquareGpu<T> kernel;
    kernels::signal::MovingMeanSquareArgs args{window_length_, -1};
    kernel.Run(ctx, mms, in, args);
  }

  /**
   * @brief Calculates the maximum value of an input
   */
  void CalcMax(TensorListView<StorageGPU, float, 0> &max,
               const TensorListView<StorageGPU, float, 1> &in,
               cudaStream_t stream) {
    kernels::DynamicScratchpad scratchpad(stream);
    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    ctx.scratchpad = &scratchpad;

    std::array<int, 1> axes = { 0 };
    max_kernel_.Setup(ctx, in.shape, make_cspan(axes), false, false);
    max_kernel_.Run(ctx, max, in);
  }

  /**
   * @brief Calculates the beginning and length of the nonsilent region
   */
  void CalcNonsilentRegion(TensorListView<StorageGPU, int32_t, 0> &begin,
                           TensorListView<StorageGPU, int32_t, 0> &len,
                           TensorListView<StorageGPU, float, 1> &mms,
                           cudaStream_t stream) {
    kernels::DynamicScratchpad scratchpad(stream);
    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    ctx.scratchpad = &scratchpad;

    int nsamples = mms.num_samples();
    int G = div_ceil(nsamples, 256), B = 256;  // for the simpler kernels

    TensorListShape<0> scalar_sh(nsamples);
    auto predicates =
        ctx.scratchpad->AllocTensorList<mm::memory_kind::device, NotLess<float>>(scalar_sh);

    if (!reference_max_) {
      auto predicates_cpu =
        ctx.scratchpad->AllocTensorList<mm::memory_kind::pinned, NotLess<float>>(scalar_sh);
      for (int i = 0; i < nsamples; i++) {
        kernels::signal::DecibelToMagnitude<float> db2mag(10.f, reference_power_[i]);
        (*predicates_cpu.data[i]) = {db2mag(cutoff_db_[i])};
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
      CalcMax(max_mms, mms, stream);

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

      InitPredicates<<<G, B, 0, ctx.gpu.stream>>>(init_predicate_args, nsamples);
    }

    auto region_tmp = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, i64vec2>(scalar_sh);
    find_region_.Setup(ctx, mms.shape);
    find_region_.Run(ctx, region_tmp, mms, predicates);

    auto postprocess_args =
        ctx.scratchpad->Allocate<mm::memory_kind::pinned, NonsilentRegionPostprocessArgs<int32_t>>(
            nsamples);
    for (int i = 0; i < nsamples; i++) {
      auto &sample = postprocess_args[i];
      sample.region = region_tmp[i].data;
      sample.begin = begin[i].data;
      sample.len = len[i].data;
      sample.input_len = mms[i].shape[0];
    }
    auto postprocess_args_gpu =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(postprocess_args, nsamples));

    // Postprocess: Deinterleave and adjust region to start of the window
    NonsilentRegionPostprocess<int32_t>
        <<<G, B, 0, ctx.gpu.stream>>>(postprocess_args_gpu, nsamples, window_length_);
  }

  template <typename T>
  void RunImplTyped(Workspace &ws) {
    auto input = view<const T, 1>(ws.Input<GPUBackend>(0));
    int nsamples = input.shape.num_samples();
    auto out_begin = view<int32_t, 0>(ws.Output<GPUBackend>(0));
    auto out_len = view<int32_t, 0>(ws.Output<GPUBackend>(1));

    // 1. Compute MMS
    kernels::DynamicScratchpad scratchpad(ws.stream());
    auto mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 1>(input.shape);
    CalcMMS(mms, input, ws.stream());

    // 2. Find the non silent region as the begin and length of the region where the energy is above
    // a given value.
    CalcNonsilentRegion(out_begin, out_len, mms, ws.stream());
  }
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);


}  // namespace dali
