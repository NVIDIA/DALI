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
    std::list<uptr_cuda_event_type> *event_ptr = nullptr;
    std::list<uptr_cuda_event_type> *copy_to_gpu_ptr = nullptr;
    if (event.size()) event_ptr = &event;
    if (copy_to_gpu.size()) copy_to_gpu_ptr = &copy_to_gpu;

    owner->RecycleBuffer(ptr, event_ptr, copy_to_gpu_ptr);
  }
};

template<>
void ExternalSource<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> cuda_event, internal_copy_to_storage;
  ExternalSourceState state_info;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_info = state_.front();
    state_.pop_front();
    // even with no_copy we may have copied from TensorVector to TensorList and we
    // need to sync with that
    if (!no_copy_ || !tensor_list_elm.front()->shares_data()) {
      internal_copy_to_storage = copy_to_storage_events_.PopFront();
      if (!no_copy_) {
        cuda_event = cuda_events_.GetEmpty();
      }
    }
  }

  auto &output = ws.Output<GPUBackend>(0);
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  if (!no_copy_ || state_info.copied_shared_data) {
    CUDA_CALL(cudaStreamWaitEvent(stream_used, *internal_copy_to_storage.front(), 0));
  }

  if (!no_copy_) {
    output.Copy(*(tensor_list_elm.front()), stream_used);
     // record an event so Recycle can synchronize on it
    cudaEventRecord(*cuda_event.front(), stream_used);
    sync_worker_.DoWork(RecycleFunctor{this, std::move(cuda_event), std::move(tensor_list_elm),
                                       std::move(internal_copy_to_storage)});
  } else if (state_info.copied_shared_data) {
    // make a shared pointer which will recycle buffer upon destruction. So when pipeline
    // no longer needs that buffer we can return it to the pool
    void *ptr = tensor_list_elm.front()->raw_mutable_data();
    int device_id = tensor_list_elm.front()->device_id();

    auto tmp_capacity = tensor_list_elm.front()->capacity();
    auto tmp_shape = tensor_list_elm.front()->shape();
    auto tmp_type = tensor_list_elm.front()->type();
    RecycleFunctor funct{this, std::move(cuda_event), std::move(tensor_list_elm),
                            std::move(internal_copy_to_storage)};
    auto tmp_shr_ptr = shared_ptr<void>(ptr, [functor = std::move(funct)] (void*) mutable {  // NOLINT (*)
                                              functor();
                                              });

    output.ShareData(tmp_shr_ptr, tmp_capacity, tmp_shape, tmp_type);
    output.set_device_id(device_id);
  } else {
    output.ShareData(tensor_list_elm.front().get());
    // empty tensor_list_elm
    tensor_list_elm.front()->Reset();
    // recycle right away as tensor_list_elm is only sharing data
    RecycleBuffer(tensor_list_elm);
  }
}

DALI_REGISTER_OPERATOR(_ExternalSource, ExternalSource<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
