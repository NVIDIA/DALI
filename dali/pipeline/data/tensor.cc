// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {
  /**
 * Loads the Tensor with data from the input Tensor.
 */
template <typename Backend>
template <typename InBackend>
void Tensor<Backend>::Copy(const Tensor<InBackend> &other, AccessOrder order) {
  constexpr bool is_host_to_host = std::is_same_v<Backend, CPUBackend> &&
                                   std::is_same_v<InBackend, CPUBackend>;
  auto src_order = other.order();
  auto dst_order = order_;
  if (!order) {
    if (is_host_to_host)
      order = AccessOrder::host();
    else  // use device order, if available; if not, use whichever (dst, src) is set
      order = dst_order.is_device()
              ? dst_order
              : src_order.is_device()
                ? src_order
                : dst_order ? dst_order : src_order;
  }
  int device_id = -1;
  if (std::is_same_v<Backend, GPUBackend>)
    device_id = this->device_id();
  else if (std::is_same_v<InBackend, GPUBackend>)
    device_id = other.device_id();
  else
    device_id = this->device_id() >= 0 ? this->device_id() : other.device_id();
  DeviceGuard dg(device_id);
  CUDAStreamLease lease;
  if (!is_host_to_host && !order.is_device()) {
    lease = CUDAStreamPool::instance().Get();
    order = lease.get();
  }
  DALI_ENFORCE(!is_host_to_host || !order.is_device(),
                "Cannot issue a host-to-host copy on a device stream.");
  this->Resize(other.shape(), other.type());
  order.wait(dst_order);  // wait for the destination to avoid overwriting while in use
  order.wait(other.order());  // wait for the source to avoid reading while not ready
  this->SetLayout(other.GetLayout());
  this->SetSourceInfo(other.GetSourceInfo());
  this->SetSkipSample(other.ShouldSkipSample());
  type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
      other.raw_data(), this->size(), order.stream());
  dst_order.wait(order);
}

template <typename Backend>
void Tensor<Backend>::Copy(
      const void *data,
      const TensorShape<> &shape,
      DALIDataType type,
      AccessOrder order) {
  this->Resize(shape, type);
  CUDAStreamLease lease;
  if (!order) {
    order = std::is_same_v<Backend, CPUBackend> ? AccessOrder::host() : order_;
    if (std::is_same_v<Backend, GPUBackend> && !order.is_device()) {
      lease = CUDAStreamPool::instance().Get();
      order = lease.get();
    }
  }

  order.wait(order_);
  type_.template Copy<Backend, CPUBackend>(this->raw_mutable_data(),
      data, this->size(), order.stream());
  order_.wait(order);
}


template class DLL_PUBLIC Tensor<CPUBackend>;
template class DLL_PUBLIC Tensor<GPUBackend>;

template void Tensor<CPUBackend>::Copy<CPUBackend>(const Tensor<CPUBackend> &, AccessOrder);
template void Tensor<CPUBackend>::Copy<GPUBackend>(const Tensor<GPUBackend> &, AccessOrder);
template void Tensor<GPUBackend>::Copy<CPUBackend>(const Tensor<CPUBackend> &, AccessOrder);
template void Tensor<GPUBackend>::Copy<GPUBackend>(const Tensor<GPUBackend> &, AccessOrder);

}  // namespace dali
