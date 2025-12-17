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
#include "dali/pipeline/data/copy_util.h"

namespace dali {
  /**
 * Loads the Tensor with data from the input Tensor.
 */
template <typename Backend>
template <typename SrcBackend>
void Tensor<Backend>::Copy(const Tensor<SrcBackend> &src, AccessOrder order) {
  CUDAStreamLease lease;
  auto [copy_order, device_id] = copy_impl::GetCopyOrderAndDevice<Backend, SrcBackend>(
    this->order(), this->device_id(), src.order(), src.device_id(), std::move(order), lease);
  // from now on, use `copy_order`, not `order`

  DeviceGuard dg(device_id);
  copy_impl::SyncBefore(this->order(), src.order(), copy_order);

  this->Resize(src.shape(), src.type());
  this->SetLayout(src.GetLayout());
  this->SetSourceInfo(src.GetSourceInfo());
  this->SetSkipSample(src.ShouldSkipSample());
  std::optional<int> dst_device_id, src_device_id;
  if (std::is_same_v<Backend, GPUBackend>)
    dst_device_id = this->device_id();
  if (std::is_same_v<SrcBackend, GPUBackend>)
    src_device_id = src.device_id();
  type_.template Copy<Backend, SrcBackend>(
      this->raw_mutable_data(),
      dst_device_id,
      src.raw_data(),
      src_device_id,
      this->size(),
      copy_order.stream());
  copy_impl::SyncAfter(this->order(), copy_order);
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

  std::optional<int> dst_dev_id, src_dev_id;
  if (std::is_same_v<Backend, GPUBackend>)
    dst_dev_id = this->device_id();

  type_.template Copy<Backend, CPUBackend>(
      this->raw_mutable_data(), dst_dev_id,
      data, std::nullopt,
      this->size(), order.stream());
  order_.wait(order);
}


template class DLL_PUBLIC Tensor<CPUBackend>;
template class DLL_PUBLIC Tensor<GPUBackend>;

template void Tensor<CPUBackend>::Copy<CPUBackend>(const Tensor<CPUBackend> &, AccessOrder);
template void Tensor<CPUBackend>::Copy<GPUBackend>(const Tensor<GPUBackend> &, AccessOrder);
template void Tensor<GPUBackend>::Copy<CPUBackend>(const Tensor<CPUBackend> &, AccessOrder);
template void Tensor<GPUBackend>::Copy<GPUBackend>(const Tensor<GPUBackend> &, AccessOrder);

}  // namespace dali
