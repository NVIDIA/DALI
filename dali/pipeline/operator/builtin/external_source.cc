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

#include "dali/pipeline/operator/builtin/external_source.h"

namespace dali {

template<>
void ExternalSource<CPUBackend>::RunImpl(HostWorkspace &ws) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> internal_copy_to_storage;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    cv_.wait(busy_lock, [&data = tl_data_, &blocking = blocking_] {
        return !(data.IsEmpty() && blocking);
      });
    if (!blocking_ && tl_data_.IsEmpty()) {
      DALI_FAIL("No data was provided to the ExternalSource. Make sure to feed it properly.");
    }
    tensor_list_elm = tl_data_.PopFront();
    internal_copy_to_storage = copy_to_storage_events_.PopFront();
  }
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  CUDA_CALL(cudaStreamWaitEvent(stream_used, *internal_copy_to_storage.front(), 0));
  auto &thread_pool = ws.GetThreadPool();
  for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
    thread_pool.DoWorkWithID([&ws, data_idx, &tensor_list_elm]
                             (int tid) {
      Tensor<CPUBackend> &output = ws.Output<CPUBackend>(0, data_idx);
      // HostWorkspace doesn't have any stream
      cudaStream_t stream = 0;
      output.Copy(*(tensor_list_elm.front()), data_idx, stream);
    });
  }
  thread_pool.WaitForWork();
  RecycleBuffer(tensor_list_elm, nullptr, &internal_copy_to_storage);
}

DALI_REGISTER_OPERATOR(_ExternalSource, ExternalSource<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);

DALI_SCHEMA(_ExternalSource)
  .DocStr(R"code("This is a backend for `ExternalSource` operator. Refer to the proper documentation
  for details.)code")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("blocking",
      R"code(Whether external source should block until data is available or just
fail when it is not)code", false)
  .MakeInternal();

DALI_SCHEMA(ExternalSource)
  .DocStr(R"code(Allows externally provided data to be passed as an input to the pipeline,
see :meth:`nvidia.dali.ops.ExternalSource.__init__`,
see :meth:`nvidia.dali.fn.external_source`,
:meth:`nvidia.dali.pipeline.Pipeline.feed_input` and
:meth:`nvidia.dali.pipeline.Pipeline.iter_setup`.
Currently this operator is not supported in TensorFlow. It is worth noting that fed inputs
should match the number of dimensions expected by the next operator in the pipeline
(e.g. HWC will expect 3-dimensional tensors
where the last dimension represents the different channels).)code")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("blocking",
      R"code(Whether external source should block until data is available or just
fail when it is not)code", false);

}  // namespace dali
