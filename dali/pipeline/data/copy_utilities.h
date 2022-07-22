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

#ifndef DALI_PIPELINE_DATA_COPY_UTILITIES_H_
#define DALI_PIPELINE_DATA_COPY_UTILITIES_H_

#include "dali/pipeline/data/types.h"
#include "dali/core/access_order.h"

namespace dali {

namespace copy_impl {

/**
 * @defgroup copy_impl Helper code for copying batches
 * The functions used as scaffolding for the synchronization of order for source and destination
 * buffers and extract the pointers from contiguous and non-contiguous batches.
 *
 * The usage is expected to be as follows:
 * 1. SyncBefore
 * 2. Resize the destination buffer(s)
 * 3. SyncAfterResize
 * 4. Use one of the CopySamplewiseImpl - it can copy between batch and a single contiguous
 *    allocation, assuming both batches are correctly resized already
 * 5. SyncAfter
 *
 * @{
 */
/**
 * @brief Pick the order for Copy to be run on and synchronize
 *
 * The copy ordering can be:
 * - explict, as specified in `order`
 * - the one from `src_order`, if set
 * - the one from `dst_order`
 * @return copy_order - order on which we will do the copy
 */
AccessOrder SyncBefore(AccessOrder dst_order, AccessOrder src_order, AccessOrder order) {
  if (!order)
    order = src_order ? src_order : dst_order;

  // Wait on the order on which we will run the copy for the work to finish on the dst
  order.wait(dst_order);

  return order;
}


/**
 * @brief Wait for the reallocation to happen in the copy order, so we can actually proceed.
 */
void SyncAfterResize(AccessOrder dst_order, AccessOrder copy_order) {
  copy_order.wait(dst_order);
}


/**
 * @brief Wait for the copy to finish in the order of the dst buffer.
 */
void SyncAfter(AccessOrder dst_order, AccessOrder copy_order) {
  dst_order.wait(copy_order);
}

/**
 * @brief Copy between two non-contiguous batches
 * Assumes matching shapes and types
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename DstBatch,
          template <typename> typename SrcBatch>
void CopySamplewiseImpl(DstBatch<DstBackend> &dst, const SrcBatch<SrcBackend> &src,
                        const TypeInfo &type_info, AccessOrder order,
                        bool use_copy_kernel = false) {
  auto num_samples = src.num_samples();
  SmallVector<const void *, 256> srcs;
  srcs.reserve(num_samples);
  SmallVector<void *, 256> dsts;
  dsts.reserve(num_samples);
  SmallVector<Index, 256> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.emplace_back(dst.raw_mutable_tensor(i));
    srcs.emplace_back(src.raw_tensor(i));
    sizes.emplace_back(src.shape()[i].num_elements());
  }

  type_info.Copy<SrcBackend, DstBackend>(dsts.data(), srcs.data(), sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}


/**
 * @brief Copy to non-contiguous batch from contiguous source.
 * Assumes matching shapes and type.
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename DstBatch>
void CopySamplewiseImpl(DstBatch<DstBackend> &dst, const void *src, const TypeInfo &type_info,
                        AccessOrder order, bool use_copy_kernel = false) {
  auto num_samples = dst.num_samples();
  SmallVector<void *, 256> dsts;
  dsts.reserve(num_samples);
  SmallVector<Index, 256> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    dsts.emplace_back(dst.raw_mutable_tensor(i));
    sizes.emplace_back(dst.shape()[i].num_elements());
  }

  type_info.Copy<DstBackend, SrcBackend>(dsts.data(), src, sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}


/**
 * @brief Copy from non-contiguous batch to contiguous destination.
 * Assumes matching shapes and types.
 */
template <typename DstBackend, typename SrcBackend, template <typename> typename SrcBatch>
void CopySamplewiseImpl(void *dst, const SrcBatch<SrcBackend> &src, const TypeInfo &type_info,
                        AccessOrder order, bool use_copy_kernel = false) {
  auto num_samples = src.num_samples();
  SmallVector<const void *, 256> srcs;
  srcs.reserve(num_samples);
  SmallVector<Index, 256> sizes;
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    srcs.emplace_back(src.raw_tensor(i));
    sizes.emplace_back(src.shape()[i].num_elements());
  }

  type_info.Copy<DstBackend, SrcBackend>(dst, srcs.data(), sizes.data(), num_samples,
                                         order.stream(), use_copy_kernel);
}

/** @} */  // end of copy_impl

}  // namespace copy_impl

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_COPY_UTILITIES_H_
