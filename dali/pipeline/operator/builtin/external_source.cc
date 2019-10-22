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
  bool is_tl_data;
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_vt_type> vector_tensor_elm;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    cv_.wait(busy_lock, [&data = data_in_tl_]{return !data.empty();});
    is_tl_data = data_in_tl_.front();
    data_in_tl_.pop_front();
    if (is_tl_data) {
        DALI_ENFORCE(!tl_data_.IsEmpty(), "ExternalSource is empty. Need to feed data first.");
        tensor_list_elm = tl_data_.PopFront();
        DALI_ENFORCE(OperatorBase::batch_size_ ==
                     static_cast<int>(tensor_list_elm.front()->ntensor()),
          "Data list provided to ExternalSource needs to have batch_size = " +
          std::to_string(batch_size_) + " length, found " +
          std::to_string(static_cast<int>(tensor_list_elm.front()->ntensor())) + " samples.");
    } else {
        DALI_ENFORCE(!t_data_.IsEmpty(), "ExternalSource is empty. Need to feed data first.");
        vector_tensor_elm = t_data_.PopFront();
        DALI_ENFORCE(OperatorBase::batch_size_ ==
                     static_cast<int>(vector_tensor_elm.front()->size()),
          "Data list provided to ExternalSource needs to have batch_size length = " +
          std::to_string(batch_size_) + " length, found " +
          std::to_string(static_cast<int>(vector_tensor_elm.front()->size())) + " samples.");
    }
  }
  auto &thread_pool = ws.GetThreadPool();
  for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
    thread_pool.DoWorkWithID([&ws, data_idx, is_tl_data, &tensor_list_elm, &vector_tensor_elm]
                             (int tid) {
      Tensor<CPUBackend> &output = ws.Output<CPUBackend>(0, data_idx);
      // HostWorkspace doesn't have any stream
      cudaStream_t stream = 0;
      if (is_tl_data) {
        output.Copy(*(tensor_list_elm.front()), data_idx, stream);
      } else {
        auto &data = (*(vector_tensor_elm.front()))[data_idx];
        output.Copy(data, stream);
      }
    });
  }
  thread_pool.WaitForWork();
  if (is_tl_data) {
    RecycleBuffer(tensor_list_elm);
  } else {
    RecycleBuffer(vector_tensor_elm);
  }
}

DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);

DALI_SCHEMA(ExternalSource)
  .DocStr(R"code(Allows externally provided data to be passed as an input to the pipeline,
see :meth:`nvidia.dali.pipeline.Pipeline.feed_input` and
:meth:`nvidia.dali.pipeline.Pipeline.iter_setup`. Currently this operator is not
supported in TensorFlow. It is worth noting that fed inputs should match the number of dimensions
expected by the next operator in the pipeline (e.g. NHWC will expect 3-dimensional tensors
where the last dimension represents the different channels).)code")
  .NumInput(0)
  .NumOutput(1);

}  // namespace dali
