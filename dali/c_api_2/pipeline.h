// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_PIPELINE_H_
#define DALI_C_API_2_PIPELINE_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/pipeline/pipeline.h"
#include "dali/c_api_2/pipeline_outputs.h"

struct _DALIPipeline {
 protected:
  _DALIPipeline() = default;
  ~_DALIPipeline() = default;
};

namespace dali::c_api {

#define GET_OR_DEFAULT(params_struct, param_name, default_value) \
  (params_struct.param_name##_present ? params_struct.param_name : default_value)

class PipelineWrapper : public _DALIPipeline {
 public:
  explicit PipelineWrapper(const daliPipelineParams_t &params) {
    bool pipelined = true, async = true, dynamic = false, set_affinity = false;
    if (params.exec_type_present) {
      pipelined = params.exec_type & DALI_EXEC_IS_PIPELINED;
      async = params.exec_type & DALI_EXEC_IS_ASYNC;
      dynamic = params.exec_type & DALI_EXEC_IS_DYNAMIC;
    }
    if (params.exec_flags_present) {
      set_affinity = params.exec_flags & DALI_EXEC_FLAGS_SET_AFFINITY;
    }
    pipeline_ = std::make_unique<Pipeline>(
      GET_OR_DEFAULT(params, max_batch_size, -1),
      GET_OR_DEFAULT(params, num_threads, -1),
      GET_OR_DEFAULT(params, device_id, -1),
      GET_OR_DEFAULT(params, seed, -1_i64),
      pipelined,
      GET_OR_DEFAULT(params, prefetch_queue_depth, 2),
      async,
      dynamic,
      0,
      set_affinity);
  }

  PipelineWrapper(const void *serialized, size_t length, const daliPipelineParams_t &params) {
    bool pipelined = true, async = true, dynamic = false, set_affinity = false;
    if (params.exec_type_present) {
      pipelined = params.exec_type & DALI_EXEC_IS_PIPELINED;
      async = params.exec_type & DALI_EXEC_IS_ASYNC;
      dynamic = params.exec_type & DALI_EXEC_IS_DYNAMIC;
    }
    if (params.exec_flags_present) {
      set_affinity = params.exec_flags & DALI_EXEC_FLAGS_SET_AFFINITY;
    }
    pipeline_ = std::make_unique<Pipeline>(
      std::string(static_cast<const char *>(serialized), length),
      GET_OR_DEFAULT(params, max_batch_size, -1),
      GET_OR_DEFAULT(params, num_threads, -1),
      GET_OR_DEFAULT(params, device_id, -1),
      pipelined,
      GET_OR_DEFAULT(params, prefetch_queue_depth, 2),
      async,
      dynamic,
      0,
      set_affinity,
      GET_OR_DEFAULT(params, seed, -1_i64));
  }

  std::unique_ptr<PipelineOutputs> PopOutputs(AccessOrder order = AccessOrder::host()) {
    return std::make_unique<PipelineOutputs>(pipeline_.get(), order);
  }

  void Build() {
    pipeline_->Build();
  }

  void Run() {
    pipeline_->Run();
  }

  void Prefetch() {
    pipeline_->Prefetch();
  }

  int GetFeedCount(std::string_view input_name) {
    return pipeline_->InputFeedCount(input_name);
  }

  void FeedInput(
      std::string_view input_name,
      const ITensorList *input_data,
      std::optional<std::string_view> data_id,
      daliFeedInputFlags_t options,
      std::optional<cudaStream_t> stream) {
    assert(input_data);
    if (input_data->GetBufferPlacement().device_type == DALI_STORAGE_CPU) {
      FeedInputImpl(
        input_name,
        *input_data->Unwrap<CPUBackend>(),
        data_id,
        options,
        stream);
    } else if (input_data->GetBufferPlacement().device_type == DALI_STORAGE_GPU) {
      FeedInputImpl(
        input_name,
        *input_data->Unwrap<GPUBackend>(),
        data_id,
        options,
        stream);
    } else {
      assert(!"Impossible device type encountered.");
    }
  }

  template <typename Backend>
  void FeedInputImpl(
        std::string_view input_name,
        const TensorList<Backend> &tl,
        std::optional<std::string_view> data_id,
        daliFeedInputFlags_t options,
        std::optional<cudaStream_t> stream) {
    InputOperatorNoCopyMode copy_mode = InputOperatorNoCopyMode::DEFAULT;

    if (options & DALI_FEED_INPUT_FORCE_COPY) {
      if (options & DALI_FEED_INPUT_NO_COPY)
        throw std::invalid_argument("DALI_FEED_INPUT_FORCE_COPY and DALI_FEED_INPUT_NO_COPY"
                                    " must notbe used together.");
      copy_mode = InputOperatorNoCopyMode::FORCE_COPY;
    } else if (options & DALI_FEED_INPUT_NO_COPY) {
      copy_mode = InputOperatorNoCopyMode::FORCE_NO_COPY;
    }

    pipeline_->SetExternalInput(
      std::string(input_name),  // TODO(michalz): switch setting input to string_view
      tl,
      stream.has_value() ? AccessOrder(*stream) : tl.order(),
      options & DALI_FEED_INPUT_SYNC,
      options & DALI_FEED_INPUT_USE_COPY_KERNEL,
      copy_mode,
      data_id ? std::optional<std::string>(std::in_place, *data_id) : std::nullopt);
  }

 private:
  std::unique_ptr<Pipeline> pipeline_;
};

PipelineWrapper *ToPointer(daliPipeline_h handle);

}  // namespace dali::c_api


#endif  // DALI_C_API_2_PIPELINE_H_
