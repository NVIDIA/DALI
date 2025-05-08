// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_REDUCE_GPU_IMPL_CUH_
#define DALI_KERNELS_REDUCE_REDUCE_GPU_IMPL_CUH_

/** @file
 *
 * This file contains a template for implementing directional reductions and
 * an implementation of Sum for GPU backend.
 */

#include <cassert>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/reduce/reduce_axes_gpu_impl.cuh"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_setup_utils.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/partition.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/core/traits.h"

namespace dali {
namespace kernels {

/// @brief Implementation details of reduction kernels
namespace reduce_impl {

enum class ReductionKind {
  All,     ///< Reduce contiguous memory
  Sample,  ///< Reduce samples separately
  Block,   ///< Reduce contiguous samples separately
  Inner,   ///< Reduce inner dimension
  Middle,  ///< Reduce middle or outer dimension
  None,    ///< No reduction - can happen when reduced dimension has unit extent
  Fold,    ///< Reduce samples, keep spatial extents
};

/**
 * @brief Describes the size of a sample (or sometimes of all samples combined) reduced by
 *        a single stage.
 *
 * Example:
 * calculate mean pixel values for a video (frames, height, width, channels).
 * 20 frames, 640x480, 3 channels
 *
 * Stage input shape: [20, 307200, 3]
 * Stage output shape: [20, 512, 3]
 * outer = 20
 * inner = 3
 * reduced_in = 307200
 * reduced_out = 512     # cannot reduce 307200 elements in one stage
 */
struct ReductionShape {
  /// Combined volume of the outer dimensions (relative to the ones reduced in this stage)
  int64_t outer;
  /// Combined volume of the inner dimensions (relative to the ones reduced in this stage)
  int64_t inner;
  /// Input volume of the dimension reduced by this stage
  int64_t reduced_in;
  /// Output volume of the dimension reduced by this stage
  int64_t reduced_out;

  int64_t input_elements() const {
    return outer * inner * reduced_in;
  }

  int64_t output_elements() const {
    return outer * inner * reduced_out;
  }
};

struct ReductionStage {
  ReductionKind kind;
  int axis;
  int index;
  bool is_last = false;

  vector<ReductionShape> shape;
  vector<int64_t> input_offsets, output_offsets;
  vector<int> sample_indices;

  int num_samples() const {
    return shape.size();
  }

  int64_t input_elements() const {
    int64_t n = 0;
    for (auto &rs : shape)
      n += rs.input_elements();
    return n;
  }

  int64_t output_elements() const {
    int64_t n = 0;
    for (auto &rs : shape)
      n += rs.output_elements();
    return n;
  }

  void calculate_offsets() {
    int64_t ofs = 0;
    int n_in = num_samples();
    input_offsets.resize(n_in);
    for (int i = 0; i < n_in; i++) {
      input_offsets[i] = ofs;
      ofs += shape[i].input_elements();
    }

    // if the stage produces contiguous output or just one sample, we only
    // need one output pointer - otherwise we need a pointer for each sample separately
    int n_out = kind == ReductionKind::All ||     // only one sample in output
                kind == ReductionKind::Sample ||  // contiguous output
                kind == ReductionKind::Block ||   // contiguous output
                kind == ReductionKind::Fold       // only one sample in output
                ? 1 : num_samples();

    output_offsets.resize(n_out);
    ofs = 0;
    for (int i = 0; i < n_out; i++) {
      output_offsets[i] = ofs;
      ofs += shape[i].output_elements();
    }
  }
};


struct TempBufferSizes {
  int64_t io_buffers = 0;
  int64_t param_buffers = 0;
  int64_t host_param_buffers = 0;
  static constexpr int kStageParamAlignment = 64;

  void AddStage(TempBufferSizes sz) {
    if (sz.io_buffers > io_buffers)
      io_buffers = sz.io_buffers;
    if (sz.param_buffers > param_buffers)
      param_buffers = sz.param_buffers;
    host_param_buffers = align_up(host_param_buffers, kStageParamAlignment) + sz.param_buffers;
  }

  template <typename T>
  void AddParam(int nsamples, size_t alignment = alignof(T)) {
    param_buffers = align_up(param_buffers, alignment) + nsamples * sizeof(T);
    host_param_buffers = align_up(host_param_buffers, alignment) + nsamples * sizeof(T);
  }

  template <typename T>
  void AddIO(int nsamples, size_t alignment = alignof(T)) {
    io_buffers = align_up(io_buffers, alignment) + nsamples * sizeof(T);
  }
};


/**
 * @brief Manages working memory for multi-stage operations
 *
 * The layout is:
 * Host:  parameters0 parameters1, ...., parametersN
 * GPU:   max_parameters I/O ......... O/I
 *
 * Every other stage the roles of inputs and outputs are switched, so the outputs of the
 * previous stage become the inputs of current one.
 *
 * The host parameter buffers are not reused across stages, because they can be still in use
 * by H2D copies
 */
struct WorkArea {
  char *host_memory = nullptr, *gpu_memory = nullptr;

  // Next parameter offset within this stage
  int64_t next_param_offset = 0;
  // Used for pinned (host) buffers, which cannot be imnmediately overwritten
  int64_t stage_param_offset = 0;
  int stage_index = 0;

  void BeginStage(int stage_idx) {
    stage_param_offset = align_up(stage_param_offset + next_param_offset,
                                  TempBufferSizes::kStageParamAlignment);
    next_param_offset = 0;
    stage_index = stage_idx;
  }

  template <typename T>
  T *ParamBuffer(int64_t n, size_t alignment = alignof(T)) {
    next_param_offset = align_up(next_param_offset, alignment);
    T *out = reinterpret_cast<T *>(host_memory + stage_param_offset + next_param_offset);
    next_param_offset += n * sizeof(T);
    assert(next_param_offset + stage_param_offset <= buffer_sizes.host_param_buffers);
    return out;
  }

  template <typename T>
  const T *InputBuffer(int64_t elements) {
    return IOBuffer<const T>(elements, (stage_index & 1) == 1);
  }

  template <typename T>
  T *OutputBuffer(int64_t elements) {
    return IOBuffer<T>(elements, (stage_index & 1) == 0);
  }

  template <typename T>
  T *IOBuffer(int64_t elements, bool alloc_at_end) {
    assert(elements * sizeof(T) <= buffer_sizes.io_buffers);
    int64_t offset;
    if (alloc_at_end) {
      offset = buffer_sizes.io_buffers + buffer_sizes.param_buffers - elements * sizeof(T);
      offset &= -alignof(T);  // we're allocating from the end, so align _down_
    } else {
      offset = buffer_sizes.param_buffers;
      offset = align_up(offset, alignof(T));
    }
    assert(offset >= buffer_sizes.param_buffers &&
           offset + elements*sizeof(T) <= buffer_sizes.io_buffers + buffer_sizes.param_buffers);
    return reinterpret_cast<T *>(gpu_memory + offset);
  }

  void CopyParamsToDevice(cudaStream_t stream) {
    if (next_param_offset) {
      char *host_param_start = host_memory + stage_param_offset;
      CUDA_CALL(cudaMemcpyAsync(gpu_memory, host_param_start, next_param_offset,
                                cudaMemcpyHostToDevice, stream));
    }
  }

  template <typename T>
  T *GetDeviceParam(T *host_param) const {
    if (host_param == nullptr)
      return nullptr;

    auto raw_ptr = reinterpret_cast<const char*>(host_param);
    char *host_param_start = host_memory + stage_param_offset;
    assert(raw_ptr >= host_param_start &&
           raw_ptr < host_param_start + next_param_offset);
    return reinterpret_cast<T*>(gpu_memory + (raw_ptr - host_param_start));
  }

  TempBufferSizes buffer_sizes;
};

template <bool value>
using bool_const = std::integral_constant<bool, value>;

template <int value>
using int_const = std::integral_constant<int, value>;


template <typename ReduceImpl>
auto GetPreprocessorHelper(
      std::true_type, const ReduceImpl *impl, int sample_idx, bool reduce_batch)
      ->decltype(impl->GetPreprocessorImpl(sample_idx, reduce_batch)) {
  return impl->GetPreprocessorImpl(sample_idx, reduce_batch);
}

template <int non_reduced_dims, typename ReduceImpl>
auto GetPreprocessorBanksHelper(
      std::true_type, const ReduceImpl *impl, WorkArea *wa, int reduced_axis)
      ->decltype(impl->GetPreprocessorBanksImpl(*wa, reduced_axis, int_const<non_reduced_dims>())) {
  return impl->GetPreprocessorBanksImpl(*wa, reduced_axis, int_const<non_reduced_dims>());
}

template <typename ReduceImpl>
auto GetPostprocessorHelper(
      std::true_type, const ReduceImpl *impl, int sample_idx, bool reduce_batch)
      ->decltype(impl->GetPostprocessorImpl(sample_idx, reduce_batch)) {
  return impl->GetPostprocessorImpl(sample_idx, reduce_batch);
}

template <bool do_preprocess>
inline identity GetPreprocessorHelper(bool_const<do_preprocess>, ...) { return {}; }
template <bool do_postprocess>
inline identity GetPostprocessorHelper(bool_const<do_postprocess>, ...) { return {}; }
template <int non_reduced_dims, bool do_preprocess>
inline IdentityPreprocessor<non_reduced_dims> *
GetPreprocessorBanksHelper(bool_const<do_preprocess>, ...) {
  return nullptr;
}

template <typename ReduceImplBaseType, int non_reduced_dims>
struct PreprocessorBankType {
  using type = decltype(*std::declval<ReduceImplBaseType&>().
    template GetPreprocessorBanks<true, non_reduced_dims>(
      std::declval<WorkArea&>(), 0));
};

template <typename ReduceImplBaseType>
struct PreprocessorType {
  using type =
    decltype(std::declval<ReduceImplBaseType&>().template GetPreprocessor<true>(0, true));
};

template <typename ReduceImplBaseType>
struct PostprocessorType {
  using type =
    decltype(std::declval<ReduceImplBaseType&>().template GetPostprocessor<true>(0, true));
};

template <typename ReduceImplBaseType, int non_reduced_dims>
using preprocessor_bank_t =
    typename PreprocessorBankType<ReduceImplBaseType, non_reduced_dims>::type;

template <typename ReduceImplBaseType>
using preprocessor_t = typename PreprocessorType<ReduceImplBaseType>::type;

template <typename ReduceImplBaseType>
using postprocessor_t = typename PostprocessorType<ReduceImplBaseType>::type;

/**
 * @brief This is the base class for implementing reductions
 *
 * Implementing reductions:
 * 1. Create a CRTP derived class MyReduction
 * ```{.cpp}
 * template <typename In>
 * class MyReduction : public ReduceImplGPU<float, In, float, MyReduction<In>>
 * ```
 * 2. Define your reduction function (to your class), e.g.:
 * ```{.cpp}
 * reductions::sum GetReduction() const { return {}; }
 * ```
 * 3. Add your pre/postprocessing functions:
 * ```{.cpp}
 * MyPostprocessor GetPortprocessor(int sample_idx, bool reduce_batch) const { ... };
 *
 * MyPreprocessor GetPreprocessor(int sample_idx, bool reduce_batch) const { ... };
 *
 * template <int non_reduced_dims>
 * MyPreprocessorBank<non_reduced_dims> *GetPreprocessorBanksImpl(WorkArea &wa) const { ... };
 * ```
 *
 * You may need to add extra arguments to Run/Setup, in which case just shadow the original
 * function(s) - you should call them at some point in your customized Run/Setup.
 *
 * See mean_stddev_gpu_impl.cuh for usage examples.
 */
template <typename Out, typename In, typename Acc, typename Actual>
class ReduceImplGPU {
 public:
  /// returns a reference to the actual derived class instance
  Actual &This() noexcept { return static_cast<Actual &>(*this); }

  /// returns a const reference to the actual derived class instance
  const Actual &This() const noexcept { return static_cast<Actual &>(*this); }

  /// Get stage - for testing
  const ReductionStage &GetStage(int idx) const { return stages_[idx]; }
  /// Get number of stages - for testing
  int GetNumStages() const { return stages_.size(); }

  /// True, if reduction across input tensors was requested
  bool ReduceBatch() const { return reduce_batch_; }

  /// Input shape after simplification (with merged dimensions)
  const TensorListShape<> &SimplifiedOutputShape() const { return out_shape_; }

  /// Output shape after simplification (with merged dimensions)
  const TensorListShape<> &SimplifiedInputShape() const { return in_shape_; }

  /// Indices of reduced axes after simplification
  span<const int> SimplifiedAxes() const { return make_span(simplified_axes_); }

  /// Reduction factor for given sample
  int64_t ReducedElements(int sample) const { return reduced_elements_[sample]; }
  /// Total reduction factor - valid when reducing whole batch
  int64_t TotalReducedElements() const { return total_reduced_; }

  /**
   * @brief Sets up the reduction
   *
   * This function determines how many and what kind of reduction stages are necessary, as well
   * as calculates memory requirements for them.
   * The arguments are validated and exception is thrown if the arguments don't describe a valid
   * reduction.
   *
   * @param ctx           kernel context - for compatibility with DALI kernel API
   * @param in_shape      shape of the input tensor
   * @param axes          indices of dimensions to reduce
   * @param keep_dims     if true, the reduced dimensions are present in the output shape, with
   *                      unit extent, otherwise the reduced dimensions are omitted in the output
   *                      shape
   * @param reduce_batch  true, if reducing across tensors within the batch
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes_arg,
                           bool keep_dims,
                           bool reduce_batch) {
    if (in_shape.sample_dim() > 64)
      throw std::range_error("Reduce supports up to 64 dimensions");
    reduce_batch_ = reduce_batch;
    axes_.copy_assign(axes_arg.begin(), axes_arg.end());
    auto axes = make_span(axes_);
    if (reduce_batch)
      CheckBatchReduce(in_shape, axes);
    KernelRequirements req;
    req.output_shapes.resize(1);
    auto &out_shape = req.output_shapes[0];
    CalculateReducedShape(out_shape, in_shape, axes, keep_dims, reduce_batch_);
    reduced_elements_.resize(in_shape.num_samples());
    CalculateReductionFactors(reduced_elements_, in_shape, axes);

    total_reduced_ = 0;
    for (auto e : reduced_elements_)
      total_reduced_ += e;

    Simplify(in_shape, axes);

    InitStages();
    CalculateTempBuffers();
    return req;
  }

  void Run(KernelContext &kctx, const OutListGPU<Out> &out, const InListGPU<In> &in) {
    assert(!stages_.empty());
    Context ctx;
    auto host_mem_size = buffer_sizes_.host_param_buffers;
    auto gpu_mem_size = buffer_sizes_.param_buffers + buffer_sizes_.io_buffers;

    ctx.stream = kctx.gpu.stream;
    ctx.work_area.buffer_sizes = buffer_sizes_;
    ctx.work_area.host_memory = kctx.scratchpad->AllocatePinned<char>(host_mem_size, 64);
    ctx.work_area.gpu_memory = kctx.scratchpad->AllocateGPU<char>(gpu_mem_size, 64);

    ctx.input = reshape(in, in_shape_, true);
    ctx.output = reshape(out, out_shape_, true);

    for (auto &stage : stages_)
      LaunchStage(ctx, stage);
  }

 private:
  void Simplify(const TensorListShape<> &in_shape, span<const int> axes) {
    SimplifyReduction(simplified_axes_, dim_groups_, in_shape, axes);
    collapse_dims(in_shape_, in_shape, dim_groups_);
    CalculateReducedShape(out_shape_, in_shape_, make_span(simplified_axes_), true, reduce_batch_);
  }

  bool HasPreprocessingParams() const {
    return !std::is_empty<preprocessor_bank_t<ReduceImplGPU, 2>>::value;
  }

  bool HasPostprocessingParams() const {
    return !std::is_empty<postprocessor_t<ReduceImplGPU>>::value;
  }

  template <typename ReduceImplBaseType, int non_reduced_dims>
  friend struct PreprocessorBankType;
  template <typename ReduceImplBaseType>
  friend struct PreprocessorType;
  template <typename ReduceImplBaseType>
  friend struct PostprocessorType;

  /*
   * The functions below get the preprocessors and postprocessors.
   * The function uses a helper class and have a default template arugment Derived = Actual
   * to make the return type a depenedent type.
   * To specialize these, define GetPreprocessorImpl, GetPostprocessorsImpl, etc in the
   * Actual class. See `StdDevImplGPU` for usage example.
   */

  template <bool do_preprocess, int non_reduced_dim, typename Derived = Actual>
  auto *GetPreprocessorBanks(WorkArea &wa, int reduced_axis) const {
    return GetPreprocessorBanksHelper<non_reduced_dim>(bool_const<do_preprocess>(),
      static_cast<const Derived *>(this), &wa, reduced_axis);
  }

  template <bool do_preprocess, typename Derived = Actual>
  auto GetPreprocessor(int sample_index, bool reduce_batch) const {
    return GetPreprocessorHelper(
      bool_const<do_preprocess>(), static_cast<const Derived *>(this), sample_index, reduce_batch);
  }

  template <bool do_postprocess, typename Derived = Actual>
  auto GetPostprocessor(int sample_index, bool reduce_batch) const {
    return GetPostprocessorHelper(
      bool_const<do_postprocess>(), static_cast<const Derived *>(this), sample_index, reduce_batch);
  }

  template <bool do_preprocess, typename Derived = Actual>
  auto *GetPreprocessors(WorkArea &wa) const {
    using pp_t = std::remove_reference_t<decltype(GetPreprocessor<do_preprocess>(0, false))>;
    pp_t *pp = nullptr;
    if (!std::is_empty<pp_t>::value) {
      bool batch = ReduceBatch();
      int N = in_shape_.num_samples();
      pp = wa.ParamBuffer<pp_t>(N);
      for (int i = 0; i < N; i++)
        pp[i] = GetPreprocessor<do_preprocess>(batch ? 0 : i, batch);
    }
    return pp;
  }

  template <bool do_postprocess, typename Derived = Actual>
  auto *GetPostprocessors(WorkArea &wa) const {
    using pp_t = std::remove_reference_t<decltype(GetPostprocessor<do_postprocess>(0, false))>;
    pp_t *pp = nullptr;
    if (!std::is_empty<pp_t>::value) {
      bool batch = ReduceBatch();
      int N = in_shape_.num_samples();
      pp = wa.ParamBuffer<pp_t>(N);
      for (int i = 0; i < N; i++)
        pp[i] = GetPostprocessor<do_postprocess>(batch ? 0 : i, batch);
    }
    return pp;
  }

 private:
  /**
   * @brief Calculate the sizes of the temporary buffers required to launch all stages
   */
  void CalculateTempBuffers() {
    int nstages = stages_.size();
    buffer_sizes_ = {};
    for (auto &stage : stages_) {
      TempBufferSizes stage_buffers;
      CalculateTempBuffers(stage_buffers, stage);
      buffer_sizes_.AddStage(stage_buffers);
    }
    buffer_sizes_.param_buffers = align_up(buffer_sizes_.param_buffers, 64);
    buffer_sizes_.host_param_buffers = align_up(buffer_sizes_.host_param_buffers, 64);
  }

  /**
   * @brief Calculates memory requirements of a single stage
   */
  void CalculateTempBuffers(TempBufferSizes &buf_sizes, ReductionStage &stage) {
    const int N = stage.num_samples();

    // Process the possible arguments in the order in which they appear in the kernels:

    // Outputs, Inputs, lengths, preprocessors
    // or
    // Sample descriptors, preprocessors, postprocessors

    switch (stage.kind) {
      case ReductionKind::Middle:
      case ReductionKind::Inner:
        if (stage.index == 0) {  // playing it safe for unlikely event of specialization
          if (stage.is_last)
            buf_sizes.AddParam<ReduceSampleDesc<Out, In>>(N);
          else
            buf_sizes.AddParam<ReduceSampleDesc<Acc, In>>(N);
        } else {
          if (stage.is_last)
            buf_sizes.AddParam<ReduceSampleDesc<Out, Acc>>(N);
          else
            buf_sizes.AddParam<ReduceSampleDesc<Acc, Acc>>(N);
        }
        break;

      case ReductionKind::None:
        buf_sizes.AddParam<Out *>(N);
        // NOTE:  actually, the stage input for all stages but the first would be Out, not In, but
        //        it doesnt affect pointer size, so we're good anyway
        buf_sizes.AddParam<const In *>(N);
        buf_sizes.AddParam<int64_t>(N);
        break;

      case ReductionKind::Sample:
        // single output - no per-sample output pointers
        buf_sizes.AddParam<const In *>(N);
        buf_sizes.AddParam<int64_t>(N);
        break;

      case ReductionKind::Fold:
        // all other arguments are scalar
        buf_sizes.AddParam<const In *>(N);
        break;
    }

    // first stage may require preprocessing
    if (stage.index == 0) {
      if (This().HasPreprocessingParams()) {
        // per-sample preprocessor may need some parameter space
        switch (stage.kind) {
          case ReductionKind::Middle:  // the preprocessor bank is 2D (outer, inner)
            buf_sizes.AddParam<preprocessor_bank_t<ReduceImplGPU, 2>>(N);
            break;
          case ReductionKind::Inner:  // the preprocessor bank is 1D (outer)
          case ReductionKind::None:   // the preprocessor bank is 1D (pointwise)
          case ReductionKind::Fold:   // the preprocessor bank is 1D (pointwise)
            buf_sizes.AddParam<preprocessor_bank_t<ReduceImplGPU, 1>>(N);
            break;
          case ReductionKind::Sample:
            // per sample scalar preprocessor is possible
            buf_sizes.AddParam<preprocessor_t<ReduceImplGPU>>(N);
            break;
          default:
            // no buffer for preprocessor - it's passed by value
            break;
        }
      }
    } else {
      buf_sizes.AddIO<Acc>(stage.input_elements());
    }

    // last stage may require postprocessing
    if (stage.is_last) {
      // per-sample postprocessor may need some parameter space
      if (This().HasPostprocessingParams()) {
        switch (stage.kind) {
          case ReductionKind::Middle:
          case ReductionKind::Inner:
          case ReductionKind::None:
          case ReductionKind::Sample:
          case ReductionKind::Block:
            buf_sizes.AddParam<postprocessor_t<ReduceImplGPU>>(N);
            break;
          default:
            // no buffer required - passed by value
            break;
        }
      }
    } else {
      buf_sizes.AddIO<Acc>(stage.output_elements());
    }
  }

  /**
   * @brief Initializes processing where no per-sample dimension is reduced
   *
   * The reduction is either pre- and post-processing only or a reduction of respective
   * elements across samples.
   */
  void InitPassThrough() {
    stages_.resize(1);
    ReductionStage &stage = stages_.back();
    stage = {};
    stage.index = 0;
    stage.is_last = true;
    stage.kind = reduce_batch_ ? ReductionKind::Fold : ReductionKind::None;
    int N = in_shape_.num_samples();
    stage.shape.resize(N);
    for (int i = 0; i < N; i++) {
      ReductionShape &rs = stage.shape[i];
      rs.inner = volume(in_shape_.tensor_shape_span(i));  // everything goes here into inner
      rs.outer = 1;
      rs.reduced_in = 1;
      rs.reduced_out = (i == 0 || !reduce_batch_) ? 1 : 0;
    }
    stages_.back().is_last = true;
  }

  static int64_t CalcReducedExtent(int64_t to_reduce, int remaining_stages) {
    if (remaining_stages == 0 || to_reduce < 2)
      return 1;
    int log2 = ilog2(to_reduce);
    int pow = log2 * remaining_stages / (remaining_stages + 1);
    if (pow == 0 && to_reduce > 2)  // avoid no-op in the following stage
      pow = 1;  // keep some work for the next stage
    int macroblocks = 1 << pow;
    return macroblocks;
  }

  /**
   * @brief Initializes a full reduction - per-sample or whole batch.
   */
  void InitReduceAll() {
    const int nsamples = in_shape_.num_samples();
    int64_t max_reduced_extent = 0;
    for (int i = 0; i < nsamples; i++) {
      int64_t extent = volume(in_shape_.tensor_shape_span(i));
      if (extent > max_reduced_extent) {
        max_reduced_extent = extent;
      }
    }

    if (max_reduced_extent == 0) {
      stages_.resize(1);
      ReductionStage &stage = stages_.front();
      stage.index = 0;
      // stage 0 is never ReduceAll, because that would require contiguous input
      stage.kind = reduce_batch_ ? ReductionKind::All : ReductionKind::Sample;
      stage.shape.resize(stage.kind == ReductionKind::All ? 1 : nsamples);
      for (auto &rs : stage.shape) {
        rs.outer = 1;
        rs.inner = 1;
        rs.reduced_out = 1;
        rs.reduced_in = 0;
      }
      stage.is_last = true;
      return;
    }

    int log2max = ilog2(max_reduced_extent * (reduce_batch_ ? nsamples : 1));
    int substages = div_ceil(log2max, 15);
    if (substages == 0)  // no reduction, but we need to at least copy
      substages++;

    if (substages == 1 && reduce_batch_ && nsamples > 1) {
      // cannot reduce multiple samples in one stage due to contiguity requirement
      substages++;
    }

    stages_.resize(substages);

    for (int s = 0; s < substages; s++) {
      ReductionStage &stage = stages_[s];
      stage.index = s;
      // stage 0 is never ReduceAll, because that would require contiguous input
      stage.kind = s == 0 ? ReductionKind::Sample
                          : reduce_batch_ ? ReductionKind::All : ReductionKind::Block;
      stage.shape.resize(stage.kind == ReductionKind::All ? 1 : nsamples);
    }

    for (int i = 0; i < nsamples; i++) {
      int64_t r = volume(in_shape_.tensor_shape_span(i));
      auto &rs = stages_[0].shape[i];
      rs.reduced_in = r;
    }

    for (auto &stage : stages_)
      for (auto &rs : stage.shape)
        rs.inner = rs.outer = 1;  // we reduce everything - no inner, no outer

    for (int s = 0; s < substages; s++) {
      int remaining_stages = substages - 1 - s;
      ReductionStage &stage = stages_[s];
      if (stage.kind == ReductionKind::All) {
        // ReduceAll cannot happen at stage 0 unless there's just one sample
        assert(s > 0 || nsamples == 1);
        int64_t r = 0;  // calculate total size
        if (s > 0) {
          for (auto &rs : stages_[s-1].shape)
            r += rs.reduced_out;
        } else {
          r = max_reduced_extent;
        }
        stage.shape[0].reduced_in = r;
        r = CalcReducedExtent(r, remaining_stages);
        stage.shape[0].reduced_out = r;
      } else {
        int64_t max_in = 0;
        if (s > 0) {
          for (int i = 0; i < nsamples;  i++) {
            auto &rs = stage.shape[i];
            int64_t r = stages_[s-1].shape[i].reduced_out;
            rs.reduced_in = r;
            rs.inner = rs.outer = 1;
            if (r > max_in)
              max_in = r;
          }
        } else {
          max_in = max_reduced_extent;
        }

        int64_t reduced_out = CalcReducedExtent(max_in, remaining_stages);

        for (int i = 0; i < nsamples; i++) {
          stage.shape[i].reduced_out = reduced_out;
        }
      }
    }

    stages_.back().is_last = true;
  }

  void CalculateOffsets() {
    for (auto &stage : stages_)
      stage.calculate_offsets();
  }

  /**
   * @brief Defines what kind of reduction stages are necessary to realize the reduction
   *        defined by the parameters passed to Setup and initializes them.
   */
  void InitStages() {
    const int nsamples = in_shape_.num_samples();
    const int in_dim = in_shape_.sample_dim();
    stages_.clear();

    // short-circuit special cases
    if (in_dim == 1) {
      // There are two major special cases:
      // 1. No reduction (or only sample reduction)
      // 2. Total reduction (per- or cross-sample)
      if (simplified_axes_.empty())
        InitPassThrough();
      else
        InitReduceAll();
      CalculateOffsets();
      return;
    }

    vector<int64_t> outer(nsamples, 1), reduced(nsamples, 1), inner(nsamples);

    for (int i = 0; i < nsamples; i++) {
      inner[i] = volume(in_shape_.tensor_shape_span(i));
    }

    int prev_axis = -1;  // no previous axis

    for (int axis : simplified_axes_) {
      // calculate the outer/inner extents for this axis
      for (int i = 0; i < nsamples; i++) {
        auto sample_shape = in_shape_.tensor_shape_span(i);
        int64_t new_outer = 1;
        for (int a = prev_axis+1; a < axis; a++) {
          new_outer *= sample_shape[a];
        }
        reduced[i] = sample_shape[axis];
        outer[i] *= new_outer;
        inner[i] = volume(sample_shape.begin() + axis + 1, sample_shape.end());
      }
      prev_axis = axis;

      int64_t max_reduced_extent = 0;
      int64_t min_reduced_extent = std::numeric_limits<int64_t>::max();
      for (int i = 0; i < nsamples; i++) {
        int64_t extent = in_shape_.tensor_shape_span(i)[axis];
        if (extent > max_reduced_extent) {
          max_reduced_extent = extent;
        }
        if (extent < min_reduced_extent) {
          min_reduced_extent = extent;
        }
      }

      int log2max = ilog2(max_reduced_extent);
      int substages = div_ceil(log2max, 15);
      if (substages == 0) {
        assert(min_reduced_extent <= 1 && max_reduced_extent <= 1);
        // We have a 0-sized reduction, so we need to run a stage that injects the neutral value.
        // If only some samples have 0 size, some waste will occur, but it's an edge case anyway.
        if (min_reduced_extent != 1)
          substages++;
      }

      bool is_inner = axis == in_dim - 1;
      for (int substage = 0; substage < substages; substage++) {
        int remaining_stages = substages - 1 - substage;  // not including this one
        ReductionStage stage;
        stage.kind = is_inner ? ReductionKind::Inner : ReductionKind::Middle;
        stage.shape.resize(nsamples);
        for (int i = 0; i < nsamples; i++) {
          int64_t r = reduced[i];

          stage.shape[i].outer = outer[i];
          stage.shape[i].inner = inner[i];
          stage.shape[i].reduced_in = r;
          stage.axis = axis;

          r = CalcReducedExtent(r, remaining_stages);
          stage.shape[i].reduced_out = r;
          reduced[i] = r;
        }
        stage.index = stages_.size();
        stages_.push_back(std::move(stage));
      }
    }
    if (reduce_batch_ && nsamples > 1) {
      ReductionStage stage;
      stage.kind = ReductionKind::Fold;
      stage.shape.resize(nsamples);
      for (int i = 0; i < nsamples; i ++) {
        stage.shape[i].outer = 1;
        stage.shape[i].inner = stages_.back().shape[i].output_elements();
        stage.shape[i].reduced_in = 1;
        stage.shape[i].reduced_out = (i == 0) ? 1 : 0;
      }
      stage.index = stages_.size();
      stages_.push_back(std::move(stage));
    }
    stages_.back().is_last = true;
    CalculateOffsets();
  }

  /**
   * @brief Defines excution environment of the reduction.
   */
  struct Context {
    cudaStream_t stream;
    WorkArea work_area;
    OutListGPU<Out> output;
    InListGPU<In> input;
  };

  template <ReductionKind kind>
  using ReductionKindTag = std::integral_constant<ReductionKind, kind>;

  /**
   * @brief Launches a reduction stage in environment given by `ctx`
   */
  void LaunchStage(Context &ctx, ReductionStage &stage) {
    ctx.work_area.BeginStage(stage.index);
    VALUE_SWITCH(stage.kind, kind, (
              ReductionKind::All,
              ReductionKind::Sample,
              ReductionKind::Block,
              ReductionKind::Inner,
              ReductionKind::Middle,
              ReductionKind::None,
              ReductionKind::Fold
      ), (  // NOLINT
        if (stage.index == 0) {
          if (stage.is_last)
            LaunchStage<Out, In, true, true>(ctx, stage, ReductionKindTag<kind>());
          else
            LaunchStage<Acc, In, true, false>(ctx, stage, ReductionKindTag<kind>());
        } else {
          if (stage.is_last)
            LaunchStage<Out, Acc, false, true>(ctx, stage, ReductionKindTag<kind>());
          else
            LaunchStage<Acc, Acc, false, false>(ctx, stage, ReductionKindTag<kind>());
        }
      ),  // NOLINT
      (assert(!"This code should be unreachable"))
    );    // NOLINT
  }

  /**
   * @brief Calculates input pointers for a stage and stores them in a host-side parameter buffer.
   *
   * @return Host-side parameter buffer containing the input pointers.
   */
  template <typename StageIn>
  const StageIn *const *InputPtrs(Context &ctx, ReductionStage &stage) const {
    WorkArea &wa = ctx.work_area;
    auto *ptrs = wa.ParamBuffer<const StageIn*>(stage.num_samples());
    if (stage.index > 0) {
      const StageIn *tmp_in = wa.InputBuffer<StageIn>(stage.input_elements());
      for (int i = 0; i < stage.num_samples(); i++)
        ptrs[i] = tmp_in + stage.input_offsets[i];
    } else {
      assert((std::is_same<const StageIn, const In>::value));
      for (int i = 0; i < stage.num_samples(); i++)
        ptrs[i] = reinterpret_cast<const StageIn*>(ctx.input.data[i]);
    }
    return ptrs;
  }

  /**
   * @brief Calculates output pointers for a stage and stores them in a host-side parameter buffer.
   *
   * @return Host-side parameter buffer containing the output pointers.
   */
  template <typename StageOut>
  StageOut *const *OutputPtrs(Context &ctx, const ReductionStage &stage) const {
    WorkArea &wa = ctx.work_area;
    auto *ptrs = wa.ParamBuffer<StageOut*>(stage.num_samples());
    if (!stage.is_last) {
      StageOut *tmp_out = wa.OutputBuffer<StageOut>(stage.output_elements());
      for (int i = 0; i < stage.num_samples(); i++)
        ptrs[i] = tmp_out + stage.output_offsets[i];
    } else {
      assert((std::is_same<StageOut, Out>::value));
      for (int i = 0; i < stage.num_samples(); i++)
        ptrs[i] = reinterpret_cast<StageOut*>(ctx.output.data[i]);
    }
    return ptrs;
  }

  /**
   * @brief Calculates sample descriptors for a stage and stores them in a host-side buffer.
   *
   * @return Host-side parameter buffer containing the descriptors.
   */
  template <typename StageOut, typename StageIn>
  auto PrepareSampleDescs(Context &ctx, const ReductionStage &stage) const {
    using SampleDesc = ReduceSampleDesc<StageOut, StageIn>;
    WorkArea &wa = ctx.work_area;
    SampleDesc *samples = wa.ParamBuffer<SampleDesc>(stage.num_samples());

    if (stage.index > 0) {
      const StageIn *tmp_in = wa.InputBuffer<StageIn>(stage.input_elements());
      for (int i = 0; i < stage.num_samples(); i++)
        samples[i].in = tmp_in + stage.input_offsets[i];
    } else {
      assert((std::is_same<const StageIn, const In>::value));
      for (int i = 0; i < stage.num_samples(); i++)
        samples[i].in = reinterpret_cast<const StageIn*>(ctx.input.data[i]);
    }

    if (!stage.is_last) {
      StageOut *tmp_out = wa.OutputBuffer<StageOut>(stage.output_elements());
      for (int i = 0; i < stage.num_samples(); i++)
        samples[i].out = tmp_out + stage.output_offsets[i];
    } else {
      assert((std::is_same<StageOut, Out>::value));
      for (int i = 0; i < stage.num_samples(); i++)
        samples[i].out = reinterpret_cast<StageOut*>(ctx.output.data[i]);
    }

    for (int i = 0; i < stage.num_samples(); i++) {
      samples[i].n_inner = stage.shape[i].inner;
      samples[i].n_outer = stage.shape[i].outer;
      samples[i].n_reduced = stage.shape[i].reduced_in;
      samples[i].num_macroblocks = stage.shape[i].reduced_out;
      samples[i].macroblock_size = div_ceil(samples[i].n_reduced, samples[i].num_macroblocks);
    }
    return samples;
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::All>) {
    assert(!is_first || ctx.input.is_contiguous());
    WorkArea &wa = ctx.work_area;

    auto pre = GetPreprocessor<is_first>(0, true);
    auto post = GetPostprocessor<is_last>(0, true);

    const StageIn *in = is_first ? reinterpret_cast<const StageIn *>(ctx.input.data[0])
                                 : wa.InputBuffer<StageIn>(stage.input_elements());

    StageOut *out = is_last ? reinterpret_cast<StageOut *>(ctx.output.data[0])
                            : wa.OutputBuffer<StageOut>(stage.output_elements());

    dim3 block(32, 32);
    dim3 grid(stage.shape[0].reduced_out);

    ReduceAllKernel<Acc><<<grid, block, 0, ctx.stream>>>(
      out, in, stage.input_elements(), This().GetReduction(), pre, post);

    CUDA_CALL(cudaGetLastError());
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::Sample>) {
    assert(!is_last || ctx.output.is_contiguous());
    WorkArea &wa = ctx.work_area;

    const StageIn *const *in = InputPtrs<StageIn>(ctx, stage);

    StageOut *out = is_last ? reinterpret_cast<StageOut *>(ctx.output.data[0])
                            : wa.OutputBuffer<StageOut>(stage.output_elements());
    int64_t *sizes = wa.ParamBuffer<int64_t>(stage.num_samples());
    for (int i = 0; i < stage.num_samples(); i++) {
      sizes[i] = stage.shape[i].reduced_in;
      assert(stage.shape[i].reduced_out == stage.shape[0].reduced_out);
    }

    dim3 block(32, 32);
    dim3 grid(stage.shape[0].reduced_out, stage.num_samples());

    auto *pre = GetPreprocessors<is_first>(wa);
    auto *post = GetPostprocessors<is_last>(wa);

    wa.CopyParamsToDevice(ctx.stream);

    auto *gpu_in              = wa.GetDeviceParam(in);
    const int64_t *gpu_sizes  = wa.GetDeviceParam(sizes);
    auto *gpu_pre             = wa.GetDeviceParam(pre);
    auto *gpu_post            = wa.GetDeviceParam(post);

    ReduceAllBatchedKernel<Acc><<<grid, block, 0, ctx.stream>>>(
      out, gpu_in, gpu_sizes, This().GetReduction(), gpu_pre, gpu_post);

    CUDA_CALL(cudaGetLastError());
  }


  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::Block>) {
    assert(!is_last || ctx.output.is_contiguous());
    assert(!is_first && "Block reduction is never the first stage");
    WorkArea &wa = ctx.work_area;

    auto *pre = GetPreprocessors<is_first>(wa);
    auto *post = GetPostprocessors<is_last>(wa);

    const StageIn *in = is_first ? reinterpret_cast<const StageIn *>(ctx.input.data[0])
                                 : wa.InputBuffer<StageIn>(stage.input_elements());

    StageOut *out = is_last ? reinterpret_cast<StageOut *>(ctx.output.data[0])
                            : wa.OutputBuffer<StageOut>(stage.output_elements());

    dim3 block(32, 32);
    dim3 grid(stage.shape[0].reduced_out, stage.num_samples());

    wa.CopyParamsToDevice(ctx.stream);
    auto *gpu_pre   = wa.GetDeviceParam(pre);
    auto *gpu_post  = wa.GetDeviceParam(post);

    int64_t sample_size = stage.shape[0].reduced_in;
    ReduceAllBlockwiseKernel<Acc><<<grid, block, 0, ctx.stream>>>(
      out, in, sample_size, This().GetReduction(), gpu_pre, gpu_post);

    CUDA_CALL(cudaGetLastError());
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::Inner>) {
    using SampleDesc = ReduceSampleDesc<StageOut, StageIn>;
    WorkArea &wa = ctx.work_area;
    SampleDesc *cpu_samples = PrepareSampleDescs<StageOut, StageIn>(ctx, stage);

    // There are four cases, depending on the sizes of the reduced and the outer dimension.
    // We separate the sample into bins by running a stable partiioning on the samples.

    int num_samples = stage.num_samples();

    auto &indices = stage.sample_indices;
    indices.resize(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    auto groups =
      multi_partition(
        indices.begin(), indices.end(),
        [&](int i) {
          return cpu_samples[i].n_reduced == 1;
        },
        [&](int i) {
          return cpu_samples[i].n_reduced < 32 && cpu_samples[i].num_macroblocks == 1;
        },
        [&](int i) {
          return cpu_samples[i].n_reduced < 1024 && cpu_samples[i].num_macroblocks == 1;
        });
    auto none_end = std::get<0>(groups);
    auto small_end = std::get<1>(groups);
    auto medium_end = std::get<2>(groups);

    int num_none = none_end - indices.begin();
    int num_small = small_end - none_end;
    int num_medium = medium_end - small_end;
    int num_large = indices.end() - medium_end;

    auto *pre = GetPreprocessorBanks<is_first, 1>(wa, stage.axis);
    auto *post = GetPostprocessors<is_last>(wa);


    permute_in_place(cpu_samples, indices);
    if (pre)
      permute_in_place(pre, indices);
    if (post)
      permute_in_place(post, indices);

    wa.CopyParamsToDevice(ctx.stream);

    SampleDesc *gpu_samples = wa.GetDeviceParam(cpu_samples);
    auto *gpu_pre = wa.GetDeviceParam(pre);
    auto *gpu_post = wa.GetDeviceParam(post);

    using pre_bank_t = std::remove_cv_t<std::remove_reference_t<decltype(*pre)>>;;
    using post_t = std::remove_cv_t<std::remove_reference_t<decltype(*post)>>;;
    using red_t = std::remove_reference_t<decltype(This().GetReduction())>;

    auto launch_params = [&](auto kernel, int nsamples, int shm_size, int max_block_size) {
      int preferred_block_size = max_block_size;
      int preferred_grid_size;  // unused
      CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(
        &preferred_grid_size,
        &preferred_block_size,
        kernel,
        shm_size,
        max_block_size));

      dim3 block(32, preferred_block_size / 32);
      int gridx = std::max(32, 512/nsamples);
      dim3 grid(gridx, nsamples);
      return std::make_pair(grid, block);
    };

    dim3 grid, block;
    int sample_offset = 0;

    // None
    if (num_none) {
      std::tie(grid, block) = launch_params(
          ReduceFlatNoneKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_none, 0, 256);

      ReduceFlatNoneKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_none;
    }

    // Small
    if (num_small) {
      std::tie(grid, block) = launch_params(
          ReduceInnerSmallKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_small, 0, 256);

      ReduceInnerSmallKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_small;
    }

    // Medium

    if (num_medium) {
      std::tie(grid, block) = launch_params(
          ReduceInnerMediumKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_medium, 0, 256);

      ReduceInnerMediumKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_medium;
    }

    int shm_size = 0x8000;
    // Large

    if (num_large) {
      std::tie(grid, block) = launch_params(
          ReduceInnerLargeKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_large, shm_size, 256);

      ReduceInnerLargeKernel<Acc><<<grid, block, shm_size, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);
    }

    assert((sample_offset + num_large) == num_samples);
    CUDA_CALL(cudaGetLastError());
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::Middle>) {
    using SampleDesc = ReduceSampleDesc<StageOut, StageIn>;
    WorkArea &wa = ctx.work_area;
    SampleDesc *cpu_samples = PrepareSampleDescs<StageOut, StageIn>(ctx, stage);

    // There are three cases, depending on the sizes of the inner and the reduced dimension.
    // We separate the sample into three bins by running a stable partiioning on the samples.

    int num_samples = stage.num_samples();

    auto &indices = stage.sample_indices;
    indices.resize(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    auto groups =
      multi_partition(
        indices.begin(), indices.end(),
        [&](int i) {
         return cpu_samples[i].n_reduced == 0;
        },
        [&](int i) {
         return cpu_samples[i].n_reduced == 1;
        },
        [&](int i) {
         return cpu_samples[i].n_reduced < 1024 && cpu_samples[i].num_macroblocks == 1;
        },
        [&](int i) {
          return cpu_samples[i].n_inner < 32;
        });
    auto degenerate_end = std::get<0>(groups);
    auto none_end = std::get<1>(groups);
    auto middle_small_end = std::get<2>(groups);
    auto middle_large_inner_small_end = std::get<3>(groups);

    int num_degenerate = degenerate_end - indices.begin();
    int num_none = none_end - degenerate_end;
    int num_middle_small = middle_small_end - none_end;
    int num_middle_large_inner_small = middle_large_inner_small_end - middle_small_end;
    int num_middle_large_inner_medium = indices.end() - middle_large_inner_small_end;

    auto *pre = GetPreprocessorBanks<is_first, 2>(wa, stage.axis);
    auto *post = GetPostprocessors<is_last>(wa);

    permute_in_place(cpu_samples, indices);
    if (pre)
      permute_in_place(pre, indices);
    if (post)
      permute_in_place(post, indices);

    wa.CopyParamsToDevice(ctx.stream);

    SampleDesc *gpu_samples = wa.GetDeviceParam(cpu_samples);
    auto *gpu_pre = wa.GetDeviceParam(pre);
    auto *gpu_post = wa.GetDeviceParam(post);

    using pre_bank_t = std::remove_cv_t<std::remove_reference_t<decltype(*pre)>>;;
    using post_t = std::remove_cv_t<std::remove_reference_t<decltype(*post)>>;;
    using red_t = std::remove_reference_t<decltype(This().GetReduction())>;

    auto launch_params = [&](auto kernel, int nsamples, int shm_size, int max_block_size) {
      int preferred_block_size = max_block_size;
      int preferred_grid_size;  // unused
      CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(
        &preferred_grid_size,
        &preferred_block_size,
        kernel,
        shm_size,
        max_block_size));

      dim3 block(32, preferred_block_size / 32);
      int gridx = std::max(32, 512/nsamples);
      dim3 grid(gridx, nsamples);
      return std::make_pair(grid, block);
    };

    dim3 grid, block;
    int sample_offset = 0;

    // Degenerate
    if (num_degenerate) {
      std::tie(grid, block) = launch_params(
          ReduceMiddleDegenerateKernel<Acc, StageOut, StageIn, red_t, post_t>,
          num_degenerate, 0, 256);

      ReduceMiddleDegenerateKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_degenerate;
    }

    // None
    if (num_none) {
      std::tie(grid, block) = launch_params(
          ReduceMiddleNoneKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_none, 0, 256);

      ReduceMiddleNoneKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_none;
    }

    // MiddleSmall
    if (num_middle_small) {
      std::tie(grid, block) = launch_params(
          ReduceMiddleSmallKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_middle_small, 0, 256);

      ReduceMiddleSmallKernel<Acc><<<grid, block, 0, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_middle_small;
    }

    int shm_size = 0x8000;

    // MiddleLargeInnerSmall

    if (num_middle_large_inner_small) {
      std::tie(grid, block) = launch_params(
          ReduceMiddleLargeInnerSmallKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_middle_large_inner_small, shm_size, 256);

      ReduceMiddleLargeInnerSmallKernel<Acc><<<grid, block, shm_size, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);

      sample_offset += num_middle_large_inner_small;
    }

    // MiddleLargeInnerMedium

    if (num_middle_large_inner_medium) {
      std::tie(grid, block) = launch_params(
          ReduceMiddleLargeInnerMediumKernel<Acc, StageOut, StageIn, red_t, pre_bank_t, post_t>,
          num_middle_large_inner_medium, shm_size, 256);

      ReduceMiddleLargeInnerMediumKernel<Acc><<<grid, block, shm_size, ctx.stream>>>(
          gpu_samples + sample_offset,
          This().GetReduction(),
          gpu_pre  ? gpu_pre  + sample_offset : nullptr,
          gpu_post ? gpu_post + sample_offset : nullptr);
    }

    assert((sample_offset + num_middle_large_inner_medium) == num_samples);
    CUDA_CALL(cudaGetLastError());
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::Fold>) {
    assert(is_last);
    assert(!stage.shape.empty());
    WorkArea &wa = ctx.work_area;

    const StageIn *const *in = InputPtrs<StageIn>(ctx, stage);
    StageOut *out = reinterpret_cast<StageOut*>(ctx.output.data[0]);
    int N = stage.num_samples();
    for (int i = 0; i < N; i++) {
      assert(stage.shape[i].inner == stage.shape[0].inner);
      assert(stage.shape[i].outer == stage.shape[0].outer);
      assert(stage.shape[i].reduced_in == 1);
      assert(stage.shape[i].reduced_out == (i ? 0 : 1));
    }

    int64_t sample_size = stage.shape[0].input_elements();

    dim3 block(std::min<int64_t>(1024, sample_size));
    dim3 grid(std::min<int>(div_ceil(sample_size, 1024), 1024));

    auto *pre = GetPreprocessorBanks<is_first, 1>(wa, -1);
    auto post = GetPostprocessor<is_last>(0, true);

    wa.CopyParamsToDevice(ctx.stream);

    auto *gpu_in  = wa.GetDeviceParam(in);
    auto *gpu_pre = wa.GetDeviceParam(pre);

    ReduceSamplesKernel<Acc><<<grid, block, 0, ctx.stream>>>(
      out, gpu_in, sample_size, N, This().GetReduction(), gpu_pre, post);

    CUDA_CALL(cudaGetLastError());
  }

  template <typename StageOut, typename StageIn, bool is_first, bool is_last>
  void LaunchStage(Context &ctx, ReductionStage &stage,
                   ReductionKindTag<ReductionKind::None>) {
    assert(is_last);
    assert(!stage.shape.empty());
    WorkArea &wa = ctx.work_area;

    const StageIn *const *in = InputPtrs<StageIn>(ctx, stage);
    StageOut *const *out = OutputPtrs<StageOut>(ctx, stage);

    int N = stage.num_samples();
    int64_t *sample_sizes = wa.ParamBuffer<int64_t>(N);

    for (int i = 0; i < N; i++) {
      sample_sizes[i] = stage.shape[i].input_elements();
    }

    dim3 block(1024);
    dim3 grid(std::max(div_ceil(1024, N), 32), N);

    auto *pre = GetPreprocessorBanks<is_first, 1>(wa, -1);
    auto *post = GetPostprocessors<is_last>(wa);

    wa.CopyParamsToDevice(ctx.stream);

    auto *gpu_in              = wa.GetDeviceParam(in);
    auto *gpu_out             = wa.GetDeviceParam(out);
    const int64_t *gpu_sizes  = wa.GetDeviceParam(sample_sizes);
    auto *gpu_pre             = wa.GetDeviceParam(pre);
    auto *gpu_post            = wa.GetDeviceParam(post);

    ReduceNoneRawKernel<<<grid, block, 0, ctx.stream>>>(
      gpu_out, gpu_in, gpu_sizes, gpu_pre, gpu_post);

    CUDA_CALL(cudaGetLastError());
  }

  static constexpr int kMaxStaticDims = DynamicTensorShapeContainer::static_size;
  /// Input shape with merged dims
  TensorListShape<> in_shape_;
  /// Output shape with merged dims (reduced dims kept)
  TensorListShape<> out_shape_;
  /// Original axes adjusted to the positive range [0, ndim-1]
  SmallVector<int, kMaxStaticDims> axes_;
  /// Merged axes (without unit ones)
  SmallVector<int, kMaxStaticDims> simplified_axes_;
  /// Groups of dimensions merged by Simplify
  SmallVector<std::pair<int, int>, kMaxStaticDims> dim_groups_;

  bool reduce_batch_ = false;

  vector<ReductionStage> stages_;
  vector<int64_t> reduced_elements_;
  int64_t total_reduced_;

  TempBufferSizes buffer_sizes_;
};

template <typename Out, typename In>
struct DefaultSumAcc {
  using type = std::conditional_t<std::is_same<Out, double>::value, Out, float>;
};

template <typename Out>
struct DefaultSumAcc<Out, int8_t> {
  using type = int64_t;
};

template <typename Out>
struct DefaultSumAcc<Out, uint8_t> {
  using type = uint64_t;
};

template <typename Out>
struct DefaultSumAcc<Out, int16_t> {
  using type = int64_t;
};

template <typename Out>
struct DefaultSumAcc<Out, uint16_t> {
  using type = uint64_t;
};

template <typename Out>
struct DefaultSumAcc<Out, int32_t> {
  using type = std::conditional_t<std::is_floating_point<Out>::value, Out, int64_t>;
};

template <typename Out>
struct DefaultSumAcc<Out, uint32_t> {
  using type = std::conditional_t<std::is_floating_point<Out>::value, Out, uint64_t>;
};

template <typename Out>
struct DefaultSumAcc<Out, int64_t> {
  using type = std::conditional_t<std::is_floating_point<Out>::value, Out, int64_t>;
};

template <typename Out>
struct DefaultSumAcc<Out, uint64_t> {
  using type = std::conditional_t<std::is_floating_point<Out>::value, Out, uint64_t>;
};

template <typename Out, typename In>
using default_sum_acc_t = typename DefaultSumAcc<Out, In>::type;

template <typename Out, typename In>
class SumImplGPU : public ReduceImplGPU<Out, In, default_sum_acc_t<Out, In>, SumImplGPU<Out, In>> {
 public:
  reductions::sum GetReduction() const { return {}; }
};

template <typename Out, typename In>
class MinImplGPU : public ReduceImplGPU<Out, In, In, MinImplGPU<Out, In>> {
 public:
  reductions::min GetReduction() const { return {}; }
};

template <typename Out, typename In>
class MaxImplGPU : public ReduceImplGPU<Out, In, In, MaxImplGPU<Out, In>> {
 public:
  reductions::max GetReduction() const { return {}; }
};

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_GPU_IMPL_CUH_
