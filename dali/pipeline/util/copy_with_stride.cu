// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>

#include <cuda_runtime.h>
#include <dali/core/dev_array.h>
#include <dali/core/geom/vec.h>
#include <dali/core/util.h>
#include "dali/pipeline/util/copy_with_stride.h"

namespace dali {

namespace strided_copy {

constexpr int MAX_DIMS = 15;

struct StridedCopyDesc {
  void *output;
  const void *input;
  int64_t in_strides[MAX_DIMS];
  int64_t out_strides[MAX_DIMS];
  int64_t size;
};

template <typename T, size_t num_elements, size_t alignment>
struct alignas(alignment) Vectorized {
  T payload[num_elements];  // NOLINT(runtime/arrays)

  DALI_DEVICE DALI_FORCEINLINE T &operator[](int idx) {
    return payload[idx];
  }
};

template <int NumBytes>
struct ElementTypeOfSize {};

template <>
struct ElementTypeOfSize<1> {
  using type = uint8_t;
};

template <>
struct ElementTypeOfSize<2> {
  using type = uint16_t;
};

template <>
struct ElementTypeOfSize<4> {
  using type = uint32_t;
};

template <>
struct ElementTypeOfSize<8> {
  using type = uint64_t;
};

template <int NumBytes, int VecLen = 4>
struct ElementType {
  static constexpr int vec_len = VecLen;
  using type = typename ElementTypeOfSize<NumBytes>::type;
  using vec_type = Vectorized<type, vec_len, vec_len * sizeof(type)>;
};


template <int MaxNDim_>
struct MismatchedNdim {
  DALI_DEVICE DALI_FORCEINLINE constexpr int operator()() {
    return MaxNDim_;
  }
};

template <>
struct MismatchedNdim<-1> {
  MismatchedNdim(int max_ndim) : max_ndim_{max_ndim} {}
  DALI_DEVICE DALI_FORCEINLINE int operator()() {
    return max_ndim_;
  }

 private:
  int max_ndim_;
};

/**
 * @brief Takes a flat output index, recomputes the output coordiantes based on the out_strides
 * and returns flat input index based on input strides.
 *
 */
template <typename MismatchedNdimT>
DALI_DEVICE DALI_FORCEINLINE int64_t GetInputIdx(MismatchedNdimT mismatched_ndim, int64_t out_idx,
                                                 const int64_t *in_strides,
                                                 const int64_t *out_strides) {
  int64_t in_idx = 0;
  int64_t elem_offset = out_idx;
#pragma unroll
  for (int dim = 0; dim < mismatched_ndim(); ++dim) {
    auto n = elem_offset / out_strides[dim];
    in_idx += n * in_strides[dim];
    elem_offset -= n * out_strides[dim];
  }
  return in_idx + elem_offset;
}

/**
 * @brief Copies element by element (in contrast to vector of elements) `num_padded_elements`
 * elements from the start and the `num_cropped_elements` elements from the end of the sample.
 *
 * Assumes that there is very few padded and cropped elements
 * (less than the ElementTypeDesc's vector type elements, typically 4).
 * In particular, only a single block will be active.
 */
template <typename ElementTypeDesc, typename MismatchedNdimT>
DALI_DEVICE DALI_FORCEINLINE void UnalignedCopy(const StridedCopyDesc &sample,
                                                MismatchedNdimT mismatched_ndim,
                                                const int num_padded_elements,
                                                const int num_cropped_elements) {
  using T = typename ElementTypeDesc::type;
  using VecT = typename ElementTypeDesc::vec_type;
  constexpr int vec_len = ElementTypeDesc::vec_len;
  if (blockIdx.x == 0 && (num_padded_elements > 0 || num_cropped_elements > 0)) {
    assert(2 * vec_len <= blockDim.x);
    assert(num_padded_elements < vec_len);
    assert(num_cropped_elements < vec_len);
    const T *__restrict__ input = static_cast<const T *>(sample.input);
    T *__restrict__ output = static_cast<T *>(sample.output);
    int padded_idx = threadIdx.x;
    int cropped_idx = threadIdx.x - vec_len;
    if (padded_idx < num_padded_elements) {
      auto in_idx =
          GetInputIdx(mismatched_ndim, threadIdx.x, sample.in_strides, sample.out_strides);
      output[threadIdx.x] = input[in_idx];
    } else if (0 <= cropped_idx && cropped_idx < num_cropped_elements) {
      int64_t idx = sample.size - num_cropped_elements + cropped_idx;
      auto in_idx = GetInputIdx(mismatched_ndim, idx, sample.in_strides, sample.out_strides);
      output[idx] = input[in_idx];
    }
  }
}

template <typename ElementTypeDesc>
DALI_HOST_DEV DALI_FORCEINLINE void SampleAlignmentInfo(int &num_padded_elements,
                                                        int &num_cropped_elements,
                                                        int64_t &aligned_down_size,
                                                        const StridedCopyDesc &sample) {
  using T = typename ElementTypeDesc::type;
  using VecT = typename ElementTypeDesc::vec_type;
  auto output_base_addr = reinterpret_cast<std::uintptr_t>(sample.output);
  auto aligned_output_addr = align_up(output_base_addr, alignof(VecT));
  num_padded_elements = (aligned_output_addr - output_base_addr) / sizeof(T);
  int64_t remaining_size = sample.size - num_padded_elements;
  aligned_down_size = align_down(remaining_size, ElementTypeDesc::vec_len);
  num_cropped_elements = remaining_size - aligned_down_size;
}

template <typename ElementTypeDesc>
bool IsAligned(const StridedCopyDesc &sample) {
  int num_padded_elements, num_cropped_elements;
  int64_t aligned_down_size;
  SampleAlignmentInfo<ElementTypeDesc>(num_padded_elements, num_cropped_elements, aligned_down_size,
                                       sample);
  return num_padded_elements == 0 && num_cropped_elements == 0;
}

/**
 * @brief Performs aligned copy.
 *
 * The input is read element-by-element but the output is stored in a vectorized type and
 * written into global memory with wider, vectorized writes.
 * This benefits performance by 1. vectorized writes and 2. utilization of
 * cache in the reads if the input happens to be mostly compact
 * (for example the strides are due to row-major image with padded rows).
 *
 * Note, the output address must be aligned to meet the vectorized writes requirements.
 * If it is not, the first `num_padded_elements` elements are skipped and the
 * (T*)(output) + num_padded_elements is assumed to be aligned.
 */
template <typename ElementTypeDesc, typename MismatchedNdimT>
DALI_DEVICE DALI_FORCEINLINE void AlignedCopy(const StridedCopyDesc &sample,
                                              MismatchedNdimT mismatched_ndim,
                                              const int num_padded_elements,
                                              const int64_t aligned_down_size) {
  using T = typename ElementTypeDesc::type;
  using VecT = typename ElementTypeDesc::vec_type;
  constexpr int vec_len = ElementTypeDesc::vec_len;
  const T *__restrict__ input = static_cast<const T *>(sample.input);
  VecT *__restrict__ output =
      reinterpret_cast<VecT *>(static_cast<T *>(sample.output) + num_padded_elements);
  for (int64_t flat_idx = vec_len * (blockIdx.x * blockDim.x + threadIdx.x);
       flat_idx < aligned_down_size; flat_idx += vec_len * blockDim.x * gridDim.x) {
    VecT out_vec;
#pragma unroll
    for (int i = 0; i < vec_len; i++) {
      auto in_idx = GetInputIdx(mismatched_ndim, flat_idx + num_padded_elements + i,
                                sample.in_strides, sample.out_strides);
      out_vec[i] = input[in_idx];
    }
    output[flat_idx / vec_len] = out_vec;
  }
}

/**
 * @brief Performs partially vectorized copy.
 *
 * The input is read element-by-element but the output is stored in a vectorized type and
 * written into global memory with wider, vectorized writes.
 * This benefits performance by 1. vectorized writes and 2. utilization of
 * cache in the reads if the input happens to be mostly compact
 * (for example the strides are due to row-major image with padded rows).
 *
 * If the output base address or size is not aligned with the vectorized type, the
 * begining and end of the sample is handled separately.
 */
template <typename ElementTypeDesc, bool IsOutputAligned = false, typename MismatchedNdimT>
__global__ void BatchedCopy(const StridedCopyDesc *sample_descs, MismatchedNdimT mismatched_ndim) {
  using T = typename ElementTypeDesc::type;
  using VecT = typename ElementTypeDesc::vec_type;
  static_assert(sizeof(VecT) == sizeof(T) * ElementTypeDesc::vec_len);
  const auto sample = sample_descs[blockIdx.y];
  int num_padded_elements, num_cropped_elements;
  int64_t aligned_down_size;
  if constexpr (!IsOutputAligned) {
    SampleAlignmentInfo<ElementTypeDesc>(num_padded_elements, num_cropped_elements,
                                         aligned_down_size, sample);
  } else {
    num_padded_elements = 0;
    num_cropped_elements = 0;
    aligned_down_size = sample.size;
  }
  UnalignedCopy<ElementTypeDesc>(sample, mismatched_ndim, num_padded_elements,
                                 num_cropped_elements);
  AlignedCopy<ElementTypeDesc>(sample, mismatched_ndim, num_padded_elements, aligned_down_size);
}

template <int ElementSize, typename MismatchedNdimT>
void CopyBatch(span<const StridedCopyDesc> sample_descs, MismatchedNdimT mismatched_ndim,
               cudaStream_t stream) {
  kernels::DynamicScratchpad scratchpad({}, stream);
  using T = ElementType<ElementSize>;
  constexpr unsigned int kMaxBlockSize = 1024u;
  static constexpr int kBlockSize = 128;
  const StridedCopyDesc *sample_descs_gpu;
  std::tie(sample_descs_gpu) = scratchpad.ToContiguousGPU(stream, sample_descs);
  int64_t max_vol = 0;
  bool has_aligned_output = true;
  for (auto &sample_desc : sample_descs) {
    max_vol = std::max(max_vol, sample_desc.size);
    has_aligned_output &= IsAligned<T>(sample_desc);
  }
  unsigned int blocks_num = div_ceil(max_vol, T::vec_len * kBlockSize);
  blocks_num = std::min(blocks_num, kMaxBlockSize);
  unsigned int num_samples = sample_descs.size();
  dim3 grid = {blocks_num, num_samples, 1};
  BOOL_SWITCH(has_aligned_output, HasAlignedOutput,
              (BatchedCopy<T, HasAlignedOutput>
               <<<grid, kBlockSize, 0, stream>>>(sample_descs_gpu, mismatched_ndim);));
}


void CopyBatch(span<const StridedCopyDesc> sample_descs, int max_mismatched_ndim, int element_size,
               cudaStream_t stream) {
  VALUE_SWITCH(element_size, ElementSize, (1, 2, 4, 8), (
    VALUE_SWITCH(max_mismatched_ndim, NDim, (0, 1, 2, 3, 4, 5), (
      CopyBatch<ElementSize>(sample_descs, MismatchedNdim<NDim>{}, stream);
    ), (
      CopyBatch<ElementSize>(sample_descs, MismatchedNdim<-1>(max_mismatched_ndim), stream);
    ));
  ), DALI_FAIL(make_string("Unsupported element size: ", element_size)););  // NOLINT
}

}  // namespace strided_copy

void ValidateBatch(int &element_size, int &ndim, std::vector<DLMTensorPtr> &dl_tensors,
                   int batch_size) {
  int num_bits;
  for (int i = 0; i < batch_size; i++) {
    auto &dlm_tensor_ptr = dl_tensors[i];
    auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
    int lanes = dl_tensor.dtype.lanes;
    DALI_ENFORCE(lanes == 1, make_string("DALI Tensors do not support types with the number of "
                                         "lanes other than 1, got tensor with `",
                                         lanes, "` lanes."));
    if (i == 0) {
      num_bits = dl_tensor.dtype.bits;
      ndim = dl_tensor.ndim;
    } else {
      DALI_ENFORCE(num_bits == dl_tensor.dtype.bits,
                   "All tensors in the DALI batch must have the same type.");
      int cur_ndim = dl_tensor.ndim;
      DALI_ENFORCE(
          ndim == cur_ndim,
          make_string("All tensors in the DALI batch must have the same number of extents. Got "
                      "tensors with `",
                      ndim, "` and `", cur_ndim, "` dims."));
    }
  }
  // Limitation based on DLToDALIType
  DALI_ENFORCE(num_bits == 8 || num_bits == 16 || num_bits == 32 || num_bits == 64,
               "Unsupported data type width. Currently DALI tensors support only types of 8, 16, "
               "32 or 64 bits");
  DALI_ENFORCE(0 <= ndim && ndim <= strided_copy::MAX_DIMS,
               make_string("DALI tensor must have between 0 and ", strided_copy::MAX_DIMS,
                           " dimensions, got tensor with `", ndim, "` dimensions."));
  element_size = num_bits / 8;
}


/**
 * @brief Copies batch of DlTensors (which may be strided) into a batch of DALI tensors (which are
 * dense/compact).
 *
 * For the input DlTensors that are not strided, we simply run the cudaMemcpyAsync. Otherwise, a
 * copy kernel is used. The copy kernel will go over the output DALI tensors linearly (the tensor is
 * compact/dese) and translate the flat output indicies into input indicies.
 *
 * The input batch is validated against some of DALI batch requirements, such as uniform data
 * type and dimensionality.
 *
 * @param output
 * @param dl_tensors
 * @param batch_size
 * @param stream
 */
void CopyDlTensorBatchGpu(TensorList<GPUBackend> &output, std::vector<DLMTensorPtr> &dl_tensors,
                          int batch_size, cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }
  int element_size, ndim;
  ValidateBatch(element_size, ndim, dl_tensors, batch_size);
  SmallVector<strided_copy::StridedCopyDesc, 128> sample_descs;
  const auto cuda_mem_copy = [&output, element_size, stream](int sample_idx,
                                                             const auto &dl_tensor) {
    void *out_data = output.raw_mutable_tensor(sample_idx);
    auto size = volume(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim) * element_size;
    CUDA_CALL(cudaMemcpyAsync(out_data, dl_tensor.data, size, cudaMemcpyDeviceToDevice, stream));
  };
  // If some innermost (the smallest in DALI tensor) strides match the strides of the incoming
  // DlPack tensor, we can stop the translation from output index to input index early. For that,
  // we need to keep track of how many outermost dimensions actually mismatch.
  int max_mismatched_ndim = 0;
  for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
    auto &dlm_tensor_ptr = dl_tensors[sample_idx];
    auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
    if (!dl_tensor.strides) {
      cuda_mem_copy(sample_idx, dl_tensor);
      continue;
    }
    strided_copy::StridedCopyDesc sample_desc;
    // compute input and the compact output strides first, if they match
    for (int d = 0; d < dl_tensor.ndim; d++) {
      sample_desc.in_strides[d] = dl_tensor.strides[d];
    }
    if (ndim > 0) {
      sample_desc.out_strides[ndim - 1] = 1;
      for (int d = ndim - 2; d >= 0; --d) {
        sample_desc.out_strides[d] = sample_desc.out_strides[d + 1] * dl_tensor.shape[d + 1];
      }
    }
    // if the strides match (and given that the out strides are compact/dense),
    // we can just go with cudamemcpy
    {
      int mismatched_ndim = ndim;
      for (int d = ndim - 1; d >= 0; d--) {
        if (sample_desc.in_strides[d] != sample_desc.out_strides[d]) {
          break;
        }
        mismatched_ndim--;
      }
      if (mismatched_ndim == 0) {
        cuda_mem_copy(sample_idx, dl_tensor);
        continue;
      }
      max_mismatched_ndim = std::max(max_mismatched_ndim, mismatched_ndim);
    }
    // otherwise, when the strides do not match, fill the rest of sample_desc
    // and add it to the vector with samples for the kernel
    sample_desc.output = output.raw_mutable_tensor(sample_idx);
    sample_desc.input = dl_tensor.data;
    sample_desc.size = volume(dl_tensor.shape, dl_tensor.shape + ndim);
    sample_descs.push_back(sample_desc);
  }
  if (sample_descs.size() > 0) {
    strided_copy::CopyBatch(make_cspan(sample_descs), max_mismatched_ndim, element_size, stream);
  }
}

}  // namespace dali
