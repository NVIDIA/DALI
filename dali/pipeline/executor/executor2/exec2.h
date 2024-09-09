// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_

#include <memory>
#include <string>
#include <vector>
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/executor/executor.h"

namespace dali {
namespace exec2 {

/** Determines the output queue size for the operators in different places in the graph.
 *
 * An operator produces its output and stores it in a (virtual) queue. The size of the queue
 * determines, how many times an operator can be run ahead of its consumers.
 *
 * A queue size of 1 means there's no queue - the operator has just one output buffer and it
 * must be consumed before the next iteration can start. A queue size of 2 means that the operator
 * can be be run twice before the result of the first iteration is consumed, and so on.
 *
 * Depending on where in the graph an operator is, it may, or may not, have a queue longer than 1.
 * To maintain meaningful prefetching, at least the producers of the pipeline outputs must have a
 * queue (OutputOnly). The opposite is the FullyBuffered policy, where all operators can accumulate
 * multiple results. BackendChange policy maintains a queue only for operators which have consumers
 * with a different backend (e.g. a CPU operator's results are consumed by a Mixed operator).
 */
enum class QueueDepthPolicy : int {
  FullyBuffered,  //< All operators maintain a queue
  BackendChange,  //< Only operators followed by one with a different backend have a queue
  OutputOnly,     //< Only the pipeline output has multiple buffers
  Legacy = BackendChange,
};

/** Determines which operators can run in parallel. */
enum class OperatorConcurrency : int {
  None,      //< at no time can mutliple operators run
  Backend,   //< operators from different backends can execute in parallel
  Full,      //< independent operators can run in parallel
};

/** Determines how CUDA streams are assigned to the operators. */
enum class StreamPolicy : int {
  Single,       //< There's just one stream that's used by all operators
  PerBackend,   //< Operators are scheduled on a stream specific to their backend (mixed or GPU)
  PerOperator   //< Independent operators are executed on separate streams.

  // TODO(michalz): Check if this is legal with existing operator implementations - likely not
  // PerIteration, //< Streams are cycled on a per-iteration basis
};

class DLL_PUBLIC Executor2 : public ExecutorBase {
 public:
  struct Config {
    /** Device identifier */
    std::optional<int> device;
    /** The number of threads used for running operators Run function
     *
     * TODO(michalz): Consider unification of the threading engines.
     */
    int operator_threads = 0;
    /** The number of threads in the thread pool passed to the operators */
    int thread_pool_threads = 0;
    /** Whether the thread pool should set thread affinity with NVML */
    bool set_affinity = false;
    /** The number of pending results CPU operators produce */
    int cpu_queue_depth = 2;
    /** The number of pending results GPU (and mixed) operators produce */
    int gpu_queue_depth = 2;
    /** Maximum batch size */
    int max_batch_size = 1;
    /** If true, checkpoints are generated */
    bool checkpointing = false;
    /** If true, pipeline outputs are returned on a stream (no sync with host) */
    bool async_output = false;

    QueueDepthPolicy queue_policy = QueueDepthPolicy::Legacy;
    OperatorConcurrency concurrency = OperatorConcurrency::Backend;
    StreamPolicy stream_policy = StreamPolicy::PerBackend;
  };

  explicit Executor2(const Config &config);
  ~Executor2() override;

  void Build(const graph::OpGraph &graph) override;

  void Build(OpGraph *graph, std::vector<std::string> output_names) override {
    throw std::logic_error("This function is maintained in the interface for legacy tests only.");
  }

  void Init() override;
  void Run() override;
  void Prefetch() override;

  void Outputs(Workspace *ws) override;
  void ShareOutputs(Workspace *ws) override;
  void ReleaseOutputs() override;
  void EnableMemoryStats(bool enable_memory_stats = false) override;
  void EnableCheckpointing(bool checkpointing = false) override;
  ExecutorMetaMap GetExecutorMeta() override;
  void Shutdown() override;
  Checkpoint& GetCurrentCheckpoint() override;
  void RestoreStateFromCheckpoint(const Checkpoint &cpt) override;
  int InputFeedCount(std::string_view input_name) override;
  OperatorBase *GetOperator(std::string_view name) override;

 protected:
  bool HasConditionals() const override {
    throw std::logic_error("This function is maintained in the interface for legacy tests only.");
  }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace exec2
}  // namespace dali


#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_
