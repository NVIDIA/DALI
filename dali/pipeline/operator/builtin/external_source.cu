// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <utility>
#include <memory>
#include <list>

#include "dali/pipeline/operator/builtin/external_source.h"

namespace dali {

template <>
struct ExternalSource<GPUBackend>::RecycleFunctor {
  RecycleFunctor() = default;
  RecycleFunctor(const RecycleFunctor &) {
    assert(!"Should never happen");
  }
  RecycleFunctor(RecycleFunctor &&) = default;
  RecycleFunctor& operator=(const RecycleFunctor&) = default;
  RecycleFunctor& operator=(RecycleFunctor&&) = default;
  ~RecycleFunctor() = default;


  RecycleFunctor(ExternalSource<GPUBackend> *owner, std::list<uptr_cuda_event_type> event,
                 std::list<uptr_tl_type> ptr, std::list<uptr_cuda_event_type> internal_copy_to_gpu)
          : owner(owner), event(std::move(event)), copy_to_gpu(std::move(internal_copy_to_gpu)),
            ptr(std::move(ptr)) {}

  ExternalSource<GPUBackend> *owner;
  std::list<uptr_cuda_event_type> event, copy_to_gpu;
  std::list<uptr_tl_type> ptr;
  void operator()() {
    owner->RecycleBuffer(ptr, &event, &copy_to_gpu);
  }
};

template<>
void ExternalSource<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> cuda_event, internal_copy_to_storage;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    internal_copy_to_storage = copy_to_storage_events_.PopFront();
    cuda_event = cuda_events_.GetEmpty();
  }

  auto &output = ws.Output<GPUBackend>(0);
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  CUDA_CALL(cudaStreamWaitEvent(stream_used, *internal_copy_to_storage.front(), 0));
  output.Copy(*(tensor_list_elm.front()), stream_used);
  // record an event so Recycle can synchronize on it
  cudaEventRecord(*cuda_event.front(), stream_used);
  sync_worker_.DoWork(RecycleFunctor{this, std::move(cuda_event), std::move(tensor_list_elm),
                                     std::move(internal_copy_to_storage)});
}

DALI_REGISTER_OPERATOR(_ExternalSource, ExternalSource<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
