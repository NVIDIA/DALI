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

#include <functional>
#include "dali/pipeline/operator/builtin/external_source.h"

namespace dali {

template<>
void ExternalSource<CPUBackend>::RunImpl(HostWorkspace &ws) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> internal_copy_to_storage;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    internal_copy_to_storage = copy_to_storage_events_.PopFront();
  }
  auto &thread_pool = ws.GetThreadPool();

  // sort by the work size
  sample_ids_.clear();
  sample_ids_.reserve(batch_size_);
  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
    sample_ids_.emplace_back(volume(tensor_list_elm.front()->tensor_shape(sample_id)), sample_id);
  }
  std::sort(sample_ids_.begin(), sample_ids_.end(), std::greater<VolumeSampleIdPair>());

  for (const auto &sample : sample_ids_) {
    auto data_idx = sample.second;
    thread_pool.DoWorkWithID([&ws, data_idx, &tensor_list_elm]
                             (int tid) {
      Tensor<CPUBackend> &output = ws.Output<CPUBackend>(0, data_idx);
      // HostWorkspace doesn't have any stream
      cudaStream_t stream = 0;
      output.Copy(*(tensor_list_elm.front()), data_idx, stream);
    });
  }
  thread_pool.WaitForWork();
  // as we copy element by element and the output is continuous we need to set layout for the whole
  // output not each element(view)
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(tensor_list_elm.front()->GetLayout());
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
