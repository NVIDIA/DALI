// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_PIPELINED_EXECUTOR_H_
#define DALI_PIPELINE_EXECUTOR_PIPELINED_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/executor/executor.h"

namespace dali {

/**
 * @brief In addition to the functionality provided by Executor, 
 * the PipelinedExecutor enables pipelined execution by queueing
 * the outputs of each stage (that aren't pipeline outputs - these
 * are already queued by the Executor), and increasing the queue
 * depth to 3. Because we have more, and deeper queues, this
 * executor requires more memory than the normal Executor, but can
 * see large performance benefits from pipelining the cpu, mixed,
 * and gpu portions of the graph.
 */
class DLL_PUBLIC PipelinedExecutor : public Executor {
 public:
  DLL_PUBLIC inline PipelinedExecutor(int batch_size, int num_thread,
      int device_id, size_t bytes_per_sample_hint,
      bool set_affinity = false, int max_num_stream = -1, int prefetch_queue_depth = 2) :
    Executor(batch_size, num_thread, device_id, bytes_per_sample_hint,
        set_affinity, max_num_stream, prefetch_queue_depth) {
  }

  DLL_PUBLIC ~PipelinedExecutor() override = default;

  DLL_PUBLIC void Build(OpGraph *graph, vector<string> output_names) override;

  DISABLE_COPY_MOVE_ASSIGN(PipelinedExecutor);

 protected:
  void SetupStageOutputsForGraph();

  void SetStageOutputsForIter(int queue_idx, WorkspaceBlob *wsb);

  template <typename Backend>
  class TensorVectorPool {
   public:
    inline TensorVectorPool(int size, int batch_size, size_t bytes_hint) {
      tvs_.resize(size);
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
          tvs_[i].push_back(std::make_shared<Tensor<Backend>>());
          tvs_[i].back()->Resize({(Index)bytes_hint});
          tvs_[i].back()->set_pinned(false);
        }
      }
    }

    inline vector<shared_ptr<Tensor<Backend>>> Get(int idx) {
      return tvs_[idx];
    }
   private:
    vector<vector<shared_ptr<Tensor<Backend>>>> tvs_;
  };

  template <typename Backend>
  class TensorPool {
   public:
    inline TensorPool(int size, int, size_t bytes_hint) {
      for (int i = 0; i < size; ++i) {
        ts_.push_back(std::make_shared<Tensor<Backend>>());
        ts_.back()->Resize({(Index)bytes_hint});
      }
    }

    inline shared_ptr<Tensor<Backend>> Get(int idx) {
      return ts_[idx];
    }
   private:
    vector<shared_ptr<Tensor<Backend>>> ts_;
  };

  // Note: Pipelining the cpu, mixed, and gpu execution
  // can be viewed as prefetching each stage w.r.t. the
  // other stages. Thus, we need to queue the outputs of
  // each stage to avoid overwriting data that could still
  // be in use. To do this, we find all outputs of the
  // cpu & mixed stages of the pipeline that aren't
  // outputs requested by the user and setup `queue_depth`
  // extra buffers that we will rotate between. Note that
  // we do not worry about CPU outputs of the mixed
  // stage, as these will only be created as outputs
  // requested by the user.
  vector<TensorPool<CPUBackend>> support_stage_outputs_;
  vector<TensorVectorPool<CPUBackend>> cpu_stage_outputs_;
  vector<TensorListPool<CPUBackend>> mixed_stage_cpu_outputs_;
  vector<TensorListPool<GPUBackend>> mixed_stage_gpu_outputs_;
  vector<OutputInfo> support_stage_output_info_;
  vector<OutputInfo> cpu_stage_output_info_;
  vector<OutputInfo> mixed_stage_cpu_output_info_;
  vector<OutputInfo> mixed_stage_gpu_output_info_;

  USE_EXECUTOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_PIPELINED_EXECUTOR_H_
