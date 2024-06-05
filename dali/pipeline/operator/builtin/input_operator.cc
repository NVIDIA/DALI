// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/input_operator.h"

namespace dali {


DALI_SCHEMA(InputOperatorBase)
                .DocStr(R"doc(
A base for any operator that forwards in-memory data to DALI pipeline.)doc")
                .NumInput(0)
                .NumOutput(0)
                .AddOptionalArg("blocking", R"code(
**Advanced** If ``True``, this operator will block until the data is available
(e.g. by calling ``feed_input``).
If ``False``, the operator will raise an error, if the data is not available.
)code", false)
                .AddOptionalArg("no_copy", R"code(
Determines whether DALI should copy the buffer when ``feed_input`` is called.

If set to True, DALI passes the user's memory directly to the pipeline, instead of copying it.
It is the user's responsibility to keep the buffer alive and unmodified until it is
consumed by the pipeline.

The buffer can be modified or freed again after the outputs of the relevant iterations
have been consumed. Effectively, it happens after ``prefetch_queue_depth`` or
``cpu_queue_depth * gpu_queue_depth`` (when they are not equal) iterations following
the ``feed_input`` call.

The memory location must match the specified ``device`` parameter of the operator.
For the CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
For the GPU, to avoid extra copy, the provided buffer must be contiguous. If you provide a list
of separate Tensors, there will be an additional copy made internally, consuming both memory
and bandwidth.
)code", false);


template<>
void InputOperator<CPUBackend>::ForwardCurrentData(TensorList<CPUBackend> &target,
                                                   std::optional<std::string> &target_data_id,
                                                   ThreadPool &thread_pool) {
  std::list<uptr_tl_type> tensor_list_elm;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_.pop_front();
  }
  target_data_id = std::move(tensor_list_elm.front().data_id);
  tensor_list_elm.front().data_id = std::nullopt;
  // if the output is pinned and input not it needs to be copied
  if (target.is_pinned() && !tensor_list_elm.front()->is_pinned()) {
    const auto &shapes = tensor_list_elm.front()->shape();
    auto curr_batch_size = shapes.num_samples();
    target.Resize(shapes, tensor_list_elm.front()->type());

    // as we copy element by element and the output is contiguous we need to set layout
    // for the whole output not each element(view)
    target.SetLayout(tensor_list_elm.front()->GetLayout());

    for (int sample_id = 0; sample_id < curr_batch_size; ++sample_id) {
      thread_pool.AddWork(
              [&target, sample_id, &tensor_list_elm](int tid) {
                  target.CopySample(sample_id, *tensor_list_elm.front(), sample_id,
                                    AccessOrder::host());
              },
              shapes.tensor_size(sample_id));
    }
    thread_pool.RunAll();
  } else {
    // swap output with tensor_list_elm content
    std::swap(target, *tensor_list_elm.front());
  }
  RecycleBuffer(tensor_list_elm);
}


template<>
void InputOperator<GPUBackend>::ForwardCurrentData(TensorList<GPUBackend> &target,
                                                   std::optional<std::string> &target_data_id,
                                                   cudaStream_t stream) {
  std::list<uptr_tl_type> tensor_list_elm;
  std::list<uptr_cuda_event_type> internal_copy_to_storage;
  InputSourceState state_info;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_info = state_.front();
    state_.pop_front();
    // even with no_copy we may have copied from TensorList to TensorList and we
    // need to sync with that
    if (!state_info.no_copy || state_info.copied_shared_data) {
      internal_copy_to_storage = copy_to_storage_events_.PopFront();
    }
  }

  if (!state_info.no_copy || state_info.copied_shared_data) {
    CUDA_CALL(cudaStreamWaitEvent(stream, *internal_copy_to_storage.front(), 0));
  }
  target_data_id = std::move(tensor_list_elm.front().data_id);
  tensor_list_elm.front().data_id = std::nullopt;

  std::swap(target, *tensor_list_elm.front());
  target.set_order(stream, false);
  tensor_list_elm.front()->set_order(internal_copy_order_);

  if (!state_info.no_copy || state_info.copied_shared_data) {
    RecycleBuffer(tensor_list_elm, &internal_copy_to_storage);
  } else {
    RecycleBuffer(tensor_list_elm);
  }
}


template<>
void InputOperator<MixedBackend>::ForwardCurrentData(TensorList<GPUBackend> &target,
                                                     std::optional<std::string> &target_data_id,
                                                     cudaStream_t stream) {
  std::list<uptr_tl_type> tensor_list_elm;
  InputSourceState state_info;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_info = state_.front();
    state_.pop_front();
  }
  target_data_id = std::move(tensor_list_elm.front().data_id);
  tensor_list_elm.front().data_id = std::nullopt;

  target.Copy(*tensor_list_elm.front(), stream);

  tensor_list_elm.front()->set_order(internal_copy_order_);

  RecycleBuffer(tensor_list_elm);
}


template<>
void InputOperator<MixedBackend>::ForwardCurrentData(TensorList<CPUBackend> &target,
                                                     std::optional<std::string> &target_data_id,
                                                     ThreadPool &thread_pool) {
  std::list<uptr_tl_type> tensor_list_elm;
  {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    tensor_list_elm = tl_data_.PopFront();
    state_.pop_front();
  }
  target_data_id = std::move(tensor_list_elm.front().data_id);
  tensor_list_elm.front().data_id = std::nullopt;
  // if the output is pinned and input not it needs to be copied
  if (target.is_pinned() && !tensor_list_elm.front()->is_pinned()) {
    const auto &shapes = tensor_list_elm.front()->shape();
    auto curr_batch_size = shapes.num_samples();
    target.Resize(shapes, tensor_list_elm.front()->type());

    // as we copy element by element and the output is contiguous we need to set layout
    // for the whole output not each element(view)
    target.SetLayout(tensor_list_elm.front()->GetLayout());

    for (int sample_id = 0; sample_id < curr_batch_size; ++sample_id) {
      thread_pool.AddWork(
              [&target, sample_id, &tensor_list_elm](int tid) {
                  target.CopySample(sample_id, *tensor_list_elm.front(), sample_id,
                                    AccessOrder::host());
              },
              shapes.tensor_size(sample_id));
    }
    thread_pool.RunAll();
  } else {
    // swap output with tensor_list_elm content
    std::swap(target, *tensor_list_elm.front());
  }
  RecycleBuffer(tensor_list_elm);
}


}  // namespace dali
