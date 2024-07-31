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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

#include "dali/pipeline/executor/executor2/workspace_cache.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

#include "dali/core/exec/tasking.h"

namespace dali {
namespace graph {
class OpGraph;
struct OpNode;
}  // namespace graph
namespace exec2 {

struct PipelineOutput {
  PipelineOutput(PipelineOutput &&) = default;
  PipelineOutput(const PipelineOutput &) {
    throw std::logic_error("This object is not copyable, but std::any needs it at compile time.");
  }
  PipelineOutput(const Workspace &ws, CUDAEvent event, std::optional<int> device)
  : workspace(ws), event(std::move(event)), device(device) {}

  ~PipelineOutput() {
    if (event)
      CUDAEventPool::instance().Put(std::move(event), device.value());
  }
  /** The payload */
  Workspace workspace;
  /** Owns the event used by the workspace */
  CUDAEvent event;
  /** The ordinal of the device used by the workspace */
  std::optional<int> device;
};


class ExecNode;
class Iteration;

/** An edge between execution nodes.
 *
 * An exec edge is an edge between execution nodes. It's not equivalent to a DataNode, which has
 * no counterpart in ExecGraph.
 */
struct ExecEdge {
  ExecNode *producer = nullptr;
  ExecNode *consumer = nullptr;
  /** The index of the output in OpSpec. It doesn't need to match the index producer->outputs. */
  int producer_output_idx = 0;
  /** The index of the input in OpSpec. It matches the edge's index in consumer->inputs. */
  int consumer_input_idx = 0;
  StorageDevice device = {};

  constexpr bool operator==(const ExecEdge &other) const {
    return producer == other.producer &&
           consumer == other.consumer &&
           producer_output_idx == other.producer_output_idx &&
           consumer_input_idx == other.consumer_input_idx &&
           device == other.device;
  }

  constexpr bool operator!=(const ExecEdge &other) const {
    return !(*this == other);
  }
};

/** A tag type for constructing output ExecNode */
struct PipelineOutputTag {};

struct ExecOutputDesc {
  SmallVector<ExecEdge *, 4> consumers;

  StorageDevice device = StorageDevice::CPU;
  bool pinned = false;

  bool parallel_consumers = true;
};

/** An execution node.
 *
 * An execution node corresponds to an operator node or an output node in the pipeline
 * definition graph (dali::graph::OpGraph).
 * It contains all (static) information and runtime environment required by the operator.
 * It also contains links to outgoing and incoming ExecEdges.
 * ExecNode owns the instance of the operator and manages the lifecycle of the tasks.
 *
 * NOTE: The output node is logically placed outside of DALI pipeline and represents the consumer
 *       of DALI outputs. As such, it has inputs rather than outputs.
 */
class DLL_PUBLIC ExecNode {
 public:
  ExecNode() = default;
  explicit ExecNode(std::unique_ptr<OperatorBase> op, const graph::OpNode *def = nullptr);
  explicit ExecNode(PipelineOutputTag) : is_pipeline_output(true) {}

  /** Inputs of the operator.
   *
   * The inputs must appear in the same order as they're defined in the operator's OpSpec.
   */
  SmallVector<ExecEdge *, 8> inputs;

  /** Outputs of the operator.
   *
   * The outputs must appear in the same order as they're defined in the operator's OpSpec.
   * The order of consumer edges in each output is not important.
   */
  SmallVector<ExecOutputDesc, 4> outputs;

  /** A semaphore limiting the cuncurrency of the operator.
   *
   * Apart from data dependency and the implicit dependency on the previous iteration,
   * an operator might by subject to limits in concurrency. This semaphore may be used
   * to implement such limitations, preventing operators that don't have a data dependency
   * from being executed concurrently.
   */
  std::shared_ptr<tasking::Semaphore> concurrency;

  /** Limits the number of output buffers for the operator.
   *
   * If the graph is scheduled multiple times ahead, the operator would produce multiple results,
   * potentially consuming a lot of memory. The output_queue_limit limits the number of pending
   * outputs. Once all consumers are done with previous iterations, the semaphore is released
   * and the task may proceed with the next iteration.
   */
  std::shared_ptr<tasking::Semaphore> output_queue_limit;

  /** The instance of the operator (or null for output node) */
  std::unique_ptr<OperatorBase> op;

  /** The task from the previous iteration - kept in order to maintain execution order */
  tasking::SharedTask prev;

  /** The task from the current iteration */
  tasking::SharedTask main_task;

  /** The task that releases the output_queue_limit semaphore.
   *
   * IMPORTANT: Release_outputs is NOT required for correctness. It's sole function is limiting
   *            the number of pending outputs from the operator.
   *
   * This is an auxiliary task which is scheduled after all direct successors of main_task.
   *
   * op A --- (output 0) ----- op B --------- ...
   *    \                         /
   *     ------- (output 1) -----(- op C --- ...
   *                              \   |
   *                               \  |
   *                                \  \
   *                                 \_release outptus (A)
   */
  tasking::SharedTask release_outputs;

  /** Data-independent execution environment (thread pool, stream, etc). */
  ExecEnv env = {};

  /** Places a workspace in a workspace cache. */
  void PutWorkspace(CachedWorkspace ws);

  /** Obtains a worskpace from a workspace cache or, if not found, creates a new one. */
  CachedWorkspace GetWorkspace(std::shared_ptr<IterationData> iter_data,
                               WorkspaceParams params);

  /** The instance name of the operator. */
  const std::string instance_name;

  /** The backend on which the operator runs. */
  OpType backend = OpType::CPU;

  /** Whether the node is the very output of the pipeline. There's only one such node. */
  bool is_pipeline_output = false;

  /** Whether the operator in the node is a batch size provider. */
  bool is_batch_size_provider = false;

  /** Visit marker for graph algorithms. */
  mutable bool visited = false;

 private:
  /** Creates a workspace at the pipeline's output.
   *
   * Output node's inputs are converted to Pipeline's outputs. The output workspace doesn't have
   * any inputs.
   */
  CachedWorkspace CreateOutputWorkspace();
  /** Creates a workspace suitable for running the operator `op`. */
  CachedWorkspace CreateOpWorkspace();

  WorkspaceCache workspace_cache_;

  /** Moves to a new iteration. */
  void NextIter() {
    prev = std::move(main_task);
    release_outputs.reset();
  }

  friend class ExecGraph;

  /** Creates the main task.
   *
   * The graph is built in multiple phases. First, the main tasks are created for all operators.
   * Then, data dependencies are added.
   * Finally, auxiliary tasks (e.g. release_outputs) are added. Creating all tasks for a node
   * cannot be don in one go because release_outputs succeeds the main tasks of this node's
   * consumers.
   */
  void CreateMainTask(std::shared_ptr<IterationData> iter, const WorkspaceParams &params);
  /** Subscribes to the results of the precending tasks. */
  void AddDataDeps();
  /** Creates auxiliary tasks and applies concurrency constraints. */
  void CreateAuxTasks();
  std::optional<tasking::TaskFuture> Launch(tasking::Scheduler &sched);
};

/** The execution graph */
class DLL_PUBLIC ExecGraph {
 public:
  std::list<ExecNode> &Nodes() {
    return nodes_;
  }

  std::list<ExecEdge> &Edges() {
    return edges_;
  }

  const std::list<ExecNode> &Nodes() const {
    return nodes_;
  }

  const std::list<ExecEdge> &Edges() const {
    return edges_;
  }

  ExecNode *Node(std::string_view name) {
    auto it = name2node_.find(name);
    if (it == name2node_.end())
      return nullptr;
    return it->second;
  }

  const ExecNode *Node(std::string_view name) const {
    auto it = name2node_.find(name);
    if (it == name2node_.end())
      return nullptr;
    return it->second;
  }

  template <typename... Args>
  ExecNode *AddNode(Args &&...args) {
    Invalidate();
    ExecNode *node = &nodes_.emplace_back(std::forward<Args>(args)...);
    if (!node->instance_name.empty()) {
      if (!name2node_.emplace(node->instance_name, node).second) {
        nodes_.pop_back();
        throw std::invalid_argument(
            make_string("Duplicate node name: \"", node->instance_name, "\""));
      }
    }
    return node;
  }

  ExecNode *AddOutputNode() {
    Invalidate();
    ExecNode *node = &nodes_.emplace_back(PipelineOutputTag());
    if (!node->instance_name.empty()) {
      if (!name2node_.emplace(node->instance_name, node).second) {
        nodes_.pop_back();
        throw std::invalid_argument(
            make_string("Duplicate node name: \"", node->instance_name, "\""));
      }
    }
    return node;
  }

  ExecEdge *Link(ExecNode *producer, int out_idx, ExecNode *consumer, int in_idx) {
    Invalidate();
    auto &edge = edges_.emplace_back();
    edge.producer = producer;
    edge.producer_output_idx = out_idx;
    edge.consumer = consumer;
    edge.consumer_input_idx = in_idx;

    if (producer) {
      producer->outputs.resize(std::max<size_t>(producer->outputs.size(), out_idx + 1));
      producer->outputs[out_idx].consumers.push_back(&edge);
    }
    if (consumer) {
      consumer->inputs.resize(std::max<size_t>(consumer->inputs.size(), in_idx + 1));
      consumer->inputs[in_idx] = &edge;
    }
    return &edge;
  }

  void Invalidate() {
    sorted_ = false;
    validated_ = false;
    analyzed_ = false;
  }

  /** Prepares the run-time resources necessary to execute an interation */
  void PrepareIteration(const std::shared_ptr<IterationData> &iter_data,
                        const WorkspaceParams &params);

  /** Executes the recently prepared iteration */
  tasking::TaskFuture Launch(tasking::Scheduler &sched);

  /** Populates the graph based on a pipeline definiton graph. */
  void Lower(const graph::OpGraph &def);

 private:
  /** Sorts the graph topologically. */
  void Sort();
  /** Runs various analyses on the graph. */
  void Analyze();
  /** A bugcheck for graph inconsitency. It throws upon detecting misconneted nodes. */
  void Validate();

  class Analyzer;
  class SortHelper;

  std::list<ExecNode> nodes_;
  std::list<ExecEdge> edges_;
  std::unordered_map<std::string_view, ExecNode *> name2node_;

  bool sorted_ = false;
  bool validated_ = false;
  bool analyzed_ = false;

  /** Creates a task that goes over batch sizes providers and establishes the batch size. */
  tasking::SharedTask InferBatchSize(const std::shared_ptr<IterationData> &iter_data,
                                     int max_batch_size);
  tasking::SharedTask infer_batch_size_task_;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_

