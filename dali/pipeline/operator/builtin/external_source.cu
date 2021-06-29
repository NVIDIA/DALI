// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template<>
void ExternalSource<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> internal_copy_to_storage;
  ExternalSourceState state_info;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_info = state_.front();
    state_.pop_front();
    // even with no_copy we may have copied from TensorVector to TensorList and we
    // need to sync with that
    if (!state_info.no_copy || state_info.copied_shared_data) {
      internal_copy_to_storage = copy_to_storage_events_.PopFront();
    }
  }

  auto &output = ws.Output<GPUBackend>(0);
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  if (!state_info.no_copy || state_info.copied_shared_data) {
    CUDA_CALL(cudaStreamWaitEvent(stream_used, *internal_copy_to_storage.front(), 0));
  }

  std::swap(output, *tensor_list_elm.front());

  if (!state_info.no_copy || state_info.copied_shared_data) {
    RecycleBuffer(tensor_list_elm, &internal_copy_to_storage);
  } else {
    RecycleBuffer(tensor_list_elm);
  }
}

DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
