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
  std::list<uptr_tv_type> tensor_vector_elm;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_vector_elm = tv_data_.PopFront();
    state_.pop_front();
  }
  auto &output = ws.template OutputRef<CPUBackend>(0);
  // if the output is pinned and input not it needs to be copied
  if (output.is_pinned() && !tensor_vector_elm.front()->is_pinned()) {
    auto &thread_pool = ws.GetThreadPool();
    const auto &shapes = tensor_vector_elm.front()->shape();
    output.Resize(shapes, tensor_vector_elm.front()->type());

    for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
      thread_pool.AddWork([&ws, sample_id, &tensor_vector_elm] (int tid) {
        Tensor<CPUBackend> &output_tensor = ws.Output<CPUBackend>(0, sample_id);
        // HostWorkspace doesn't have any stream
        cudaStream_t stream = 0;
        output_tensor.Copy((*tensor_vector_elm.front())[sample_id], stream);
      }, shapes.tensor_size(sample_id));
    }
    thread_pool.RunAll();
    // as we copy element by element and the output is contiguous we need to set layout
    // for the whole output not each element(view)
    auto &output = ws.template OutputRef<CPUBackend>(0);
    output.SetLayout(tensor_vector_elm.front()->GetLayout());
  } else {
    // swap output with tensor_vector_elm content
    std::swap(output, *tensor_vector_elm.front());
  }
  RecycleBuffer(tensor_vector_elm);
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
fail when it is not)code", true)
  .AddOptionalArg("no_copy",
      R"code(Whether DALI should copy the buffer when feed_input is called
If True, DALI passes the user memory directly to the Pipeline, instead of copying.
It is the user's responsibility to keep the buffer alive and unmodified
until it is consumed by the pipeline.

The buffer can be modified or freed again after the relevant iteration output has been consumed.
Effectively, it happens after ``prefetch_queue_depth`` or ``cpu_queue_depth * gpu_queue_depth``
(when they are not equal) iterations following the``feed_input`` call.

Provided memory must match the specified `device` parameter of the operator.
For CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
For GPU to not do any copies the provided buffer must be contiguous. If user provides a list
of separate Tensors there will be an additional internal copy made.)code", false)
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
fail when it is not)code", false)
  .AddOptionalArg("no_copy",
      R"code(Whether DALI should copy the buffer when feed_input is called
If True, DALI passes the user memory directly to the Pipeline, instead of copying.
It is the user's responsibility to keep the buffer alive and unmodified
until it is consumed by the pipeline.

The buffer can be modified or freed again after the relevant iteration output has been consumed.
Effectively, it happens after ``prefetch_queue_depth`` or ``cpu_queue_depth * gpu_queue_depth``
(when they are not equal) iterations following the``feed_input`` call.

Provided memory must match the specified `device` parameter of the operator.
For CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
For GPU to not do any copies the provided buffer must be contiguous. If user provides a list
of separate Tensors there will be an additional internal copy made.)code", false);

}  // namespace dali
