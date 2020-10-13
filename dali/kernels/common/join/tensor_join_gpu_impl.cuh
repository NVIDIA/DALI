// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/core/fast_div.h"
#include "dali/core/math_util.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace tensor_join {

template <typename Element>
struct InputDesc {
  // Offset of this tensor's slice in the otput tensor's slice
  uint64_t join_offset;

  // Distance between joined slices
  uint64_t outer_stride;

  const Element *__restrict__ data;
};

template <typename Element>
struct OutputDesc {
  Element *__restrict__ data;
  fast_div<uint64_t> outer_stride;
  float guess_tensor_mul;  // used for quickly estimating which tensor we've hit based on the offset
  bool is_uniform_join;    // if true, the inputs have the same shape along

  uint64_t total_size;
};

/**
 * @brief A guided binary search for a tensor at given offset
 *
 * The input tensors have their `join_offsets` fields so given the inner offset in the
 * output, we can find an input that corresponds to this offset.
 *
 * The output descriptor contains a factor which is used to estimate which tensor we've hit
 * without searching. This greatly improves the performance of this function when stacking
 * (since this guess succeeds unless there's a precision issue) - especially when stacking
 * many inputs. This guess can also prove useful when concatenating tensors of similar shape.
 */
template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float guess_tensor_mul,
                             const InputDesc<Element> *__restrict__ descs, int njoin,
                             std::integral_constant<int, -1> ) {
  int lo = 0, hi = njoin - 1;
  int m = min(floor_int((offset + 0.5f) * guess_tensor_mul), njoin - 1);
  do {  // binary search with initial guess
    if (offset < descs[m].join_offset)
      hi = m - 1;
    else if (offset >= descs[m].join_offset + descs[m].outer_stride)
      lo = m + 1;
    else
      break;  // Found! For stacking this will always work unless precision fails
    m = (lo + hi) >> 1;  // proceed to regular binary search
  } while (lo <= hi);
  return m;
}

/**
 * @brief A fast for trivial join
 */
template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float /*guess_tensor_mul*/,
                             const InputDesc<Element> *__restrict__ descs, int /*njoin*/,
                             std::integral_constant<int, 1> ) {
  return 0;
}

/**
 * @brief A fast path for joining 2 tensors
 */
template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float /*guess_tensor_mul*/,
                             const InputDesc<Element> *__restrict__ descs, int /*njoin*/,
                             std::integral_constant<int, 2> ) {
  return offset >= descs[1].join_offset ? 1 : 0;
}

/**
 * @brief A fast path for joining 3 tensors
 */
template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float /*guess_tensor_mul*/,
                             const InputDesc<Element> *__restrict__ descs, int /*njoin*/,
                             std::integral_constant<int, 3> ) {
  return offset >= descs[2].join_offset ? 2 : offset >= descs[1].join_offset ? 1 : 0;
}

/**
 * @brief A fast path for joining 4 tensors
 */
template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float /*guess_tensor_mul*/,
                             const InputDesc<Element> *__restrict__ descs, int /*njoin*/,
                             std::integral_constant<int, 4> ) {
  return offset >= descs[2].join_offset
        ? offset >= descs[3].join_offset ? 3 : 2
        : offset >= descs[1].join_offset ? 1 : 0;
}


/**
 * @brief A fast path for joining a small number of tensors
 */
template <typename Element, int static_njoin>
DALI_HOST_DEV int FindTensor(uint64_t offset, float guess_tensor_mul,
                             const InputDesc<Element> *__restrict__ descs, int njoin,
                             std::integral_constant<int, static_njoin> ) {
  const int mid = (static_njoin + 1) / 2;
  const int hi = static_njoin - mid;
  return offset >= descs[mid].join_offset
    ? FindTensor(offset, guess_tensor_mul, descs + mid, hi, std::integral_constant<int, hi>()) + mid
    : FindTensor(offset, guess_tensor_mul, descs, mid, std::integral_constant<int, mid>());
}


template <int static_njoin, typename Element>
__device__ void JoinTensors(OutputDesc<Element> out,
                            const InputDesc<Element> *__restrict__ in,
                            int njoin) {
  uint64_t block_size = static_cast<uint64_t>(blockDim.x);
  uint64_t out_offset = threadIdx.x + block_size * blockIdx.x;
  uint64_t step = block_size * gridDim.x;
  for (; out_offset < out.total_size; out_offset += step) {
    uint64_t join_offset;
    uint64_t outer = div_mod(join_offset, out_offset, out.outer_stride);

    int t = FindTensor(join_offset, out.guess_tensor_mul, in, njoin,
                        std::integral_constant<int, static_njoin>());

    ptrdiff_t offset_in_tensor = join_offset - in[t].join_offset;
    out.data[out_offset] = in[t].data[offset_in_tensor + outer * in[t].outer_stride];
  }
}

template <typename Element>
__global__ void JoinTensorsKernel(const OutputDesc<Element> *__restrict__ out,
                                  const InputDesc<Element> *__restrict__ in,
                                  int njoin) {
  int sample_idx = blockIdx.y;
  if (out[sample_idx].is_uniform_join) {
    // uniform join - it's cheaper to calculate use the multiplier to guess the tensor index
    VALUE_SWITCH(njoin, static_njoin, (1, 2, 3, 4),
      (JoinTensors<static_njoin>(out[sample_idx], in + sample_idx * njoin, njoin)),
      (JoinTensors<-1>(out[sample_idx], in + sample_idx * njoin, njoin)));
  } else {
    // non-uniform join - the guessing doesn't help much
    VALUE_SWITCH(njoin, static_njoin, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
      (JoinTensors<static_njoin>(out[sample_idx], in + sample_idx * njoin, njoin)),
      (JoinTensors<-1>(out[sample_idx], in + sample_idx * njoin, njoin)));
  }
}

/**
 * @brief Populates the input and output descriptors given the input and output tensor lists.
 *
 * @tparam ElementType tensor element
 * @tparam out_ndim   dimensionality of the output, typically DynamicDimensions
 * @tparam in_ndim    dimensionality of the inputs, typically DynamicDimensions
 *
 * @remarks
 * The descirptors are the same when concatenating and stacking, hence no `new_axis` parameter.
 */
template <typename ElementType, int out_ndim, int in_ndim>
void FillDescs(span<OutputDesc<ElementType>> output_descs,
               span<InputDesc<ElementType>> input_descs,
               const OutListGPU<ElementType, out_ndim> &output,
               span<const InListGPU<ElementType, in_ndim> *const> inputs,
               int axis) {
  int njoin = inputs.size();
  int N = output.num_samples();
  assert(static_cast<int>(output_descs.size()) == N);
  assert(static_cast<int>(input_descs.size()) == N * njoin);

  for (int i = 0; i < N; i++) {
    int64_t join_offset = 0;
    bool is_uniform_join = true;
    for (int t = 0; t < njoin; t++) {
      auto in_shape = inputs[t]->tensor_shape_span(i);
      auto &desc = input_descs[i*njoin+t];
      desc.data = inputs[t]->data[i];
      desc.outer_stride = volume(in_shape.begin() + axis, in_shape.end());
      desc.join_offset = join_offset;
      join_offset += desc.outer_stride;
      if (desc.outer_stride != input_descs[i*njoin].outer_stride)
        is_uniform_join = false;
    }

    auto out_shape = output.tensor_shape_span(i);
    auto &out_desc = output_descs[i];
    out_desc.data = output.data[i];
    auto join_size = volume(out_shape.begin() + axis, out_shape.end());
    out_desc.outer_stride = join_size;
    out_desc.total_size = volume(out_shape);
    out_desc.guess_tensor_mul = 1.0 * njoin / join_size;
    out_desc.is_uniform_join = is_uniform_join;
  }
}


}  // namespace tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
