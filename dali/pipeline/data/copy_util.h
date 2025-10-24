// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_COPY_UTIL_H_
#define DALI_PIPELINE_DATA_COPY_UTIL_H_

#include <cassert>
#include <utility>
#include "dali/core/access_order.h"
#include "dali/core/cuda_stream_pool.h"

namespace dali {
namespace copy_impl {

/** Selects the most appropriate copy order and device based on backends, orders and devices
 *
 * @param dst_order         The order (stream or host) associated with the destination buffer
 * @param dst_device_id     The device id of the destination
 * @param src_order         The order (stream or host) associated with the source buffer
 * @param src_device_id     The device id of the source
 * @param explicit_order    An explicit order passed by the user.
 * @param tmp_stream        [out] Lease of a temporary stream.
 *
 * H2H:
 * Copy cannot happen on a stream - the function just checks that explicit order wasn't a stream
 * and returns host order.
 *
 * D2D/H2D/D2H:
 * If explcit stream is given, try to use it. If `host` order is passed for a device copy,
 * a temporary stream will be used and stored in tmp_stream.
 * If explicit stream is not given, try to use the stream associated with the GPU argument.
 * If not set, use whichever order is a GPU stream. If neither is, use a temporary stream.
 */
template <typename DstBackend, typename SrcBackend>
std::pair<AccessOrder, int> GetCopyOrderAndDevice(
        AccessOrder dst_order, int dst_device_id,
        AccessOrder src_order, int src_device_id,
        AccessOrder explicit_order,
        CUDAStreamLease &tmp_stream) {
    constexpr bool is_host_to_host = std::is_same_v<DstBackend, CPUBackend> &&
                                     std::is_same_v<SrcBackend, CPUBackend>;
    if constexpr (is_host_to_host) {
        if (explicit_order.is_device())
            throw std::logic_error("Cannot issue a host-to-host copy on a device stream.");
        return { AccessOrder::host(), -1 };
    } else {
        if (explicit_order) {
            int dev = -1;
            if (std::is_same_v<DstBackend, GPUBackend>)
                dev = dst_device_id;
            else if (std::is_same_v<SrcBackend, GPUBackend>)
                dev = src_device_id;
            else
                dev = dst_device_id >= 0 ? dst_device_id : src_device_id;

            if (!is_host_to_host && !explicit_order.is_device()) {
                tmp_stream = CUDAStreamPool::instance().Get(dev);
                return { AccessOrder(tmp_stream.get()), dev };
            }

            return { explicit_order, dev };
        } else {
            int dev = -1;
            // use the device (and possibly stream) from any GPU argument
            if (std::is_same_v<DstBackend, GPUBackend>) {
                explicit_order = dst_order;
                dev = dst_device_id;
            } else {
                assert((std::is_same_v<SrcBackend, GPUBackend>));
                explicit_order = src_order;
                dev = src_device_id;
            }

            if (!explicit_order.is_device()) {
                tmp_stream = CUDAStreamPool::instance().Get(dev);
                explicit_order = AccessOrder(tmp_stream.get());
            }
            return { explicit_order, dev };
        }
    }
}

/** Waits in the copy order for src_order (to be ready) and dst_order (to avoid overwriting) */
inline void SyncBefore(AccessOrder dst_order, AccessOrder src_order, AccessOrder copy_order) {
    copy_order.wait(src_order);
    copy_order.wait(dst_order);
}

/** Wait in destination order for the copy_order. */
inline void SyncAfter(AccessOrder dst_order, AccessOrder copy_order) {
    dst_order.wait(copy_order);
}

}  // namespace copy_impl
}  // namespace dali


#endif  // DALI_PIPELINE_DATA_COPY_UTIL_H_
