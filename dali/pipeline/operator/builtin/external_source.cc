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

#include "dali/pipeline/operator/builtin/external_source.h"
#include <functional>

namespace dali {

template <>
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
    auto curr_batch_size = shapes.num_samples();
    output.Resize(shapes, tensor_vector_elm.front()->type());

    for (int sample_id = 0; sample_id < curr_batch_size; ++sample_id) {
      thread_pool.AddWork(
          [&ws, sample_id, &tensor_vector_elm](int tid) {
            Tensor<CPUBackend> &output_tensor = ws.Output<CPUBackend>(0, sample_id);
            // HostWorkspace doesn't have any stream
            cudaStream_t stream = 0;
            output_tensor.Copy((*tensor_vector_elm.front())[sample_id], stream);
          },
          shapes.tensor_size(sample_id));
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


DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);


// This schema is partially internal. We want it to be listed int the supported_ops,
// but it is explicitly not loaded by the Op Factory. Instead the Python wrapper classes
// access it directly.
// C++ operators should access this operator directly as well.
DALI_SCHEMA(ExternalSource)
  .DocStr(R"code(Allows externally provided data to be passed as an input to the pipeline.

  This is a backend for `ExternalSource` operator. For Python functionality, refer to
  nvidia.dali.fn.external_source operator documentation.

  This operator can be used with C and C++ APIs by either directly specyfing it with OpSpec
  or by the Pipeline::AddExternalInput method.)code")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("blocking",
      R"code(Whether external source should block until data is available or just
fail when it is not)code", true)
  .AddOptionalArg("no_copy",
      R"code(Determines whether DALI should copy the buffer when feed_input is called.

If set to True, DALI passes the user's memory directly to the pipeline, instead of copying it.
It is the user's responsibility to keep the buffer alive and unmodified until it is
consumed by the pipeline.

The buffer can be modified or freed again after the outputs of the relevant iterations
have been consumed. Effectively, it happens after ``prefetch_queue_depth`` or
``cpu_queue_depth * gpu_queue_depth`` (when they are not equal) iterations following
the``feed_input`` call.

The memory location must match the specified ``device`` parameter of the operator.
For the CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
For the GPU, to avoid extra copy, the provided buffer must be contiguous. If you provide a list
of separate Tensors, there will be an additional copy made internally, consuming both memory
and bandwidth.)code", false);

}  // namespace dali
