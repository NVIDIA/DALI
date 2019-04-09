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

#ifndef DALI_PIPELINE_PIPELINE_H_
#define DALI_PIPELINE_PIPELINE_H_

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operators/util/external_source.h"
#include "dali/pipeline/graph/op_graph.h"


namespace dali {

/**
 * @brief Organizes and executes the set of operators chosen by the user.
 *
 * Pipelines are composed of cpu and gpu operators. When adding ops the
 * the pipeline, users can specify the 'device' argument as either 'cpu'
 * or 'gpu' to control where the op is executed. The only constraints on
 * the graphs of ops that users can define is that all cpu ops must
 * precede all gpu ops, and that cpu ops can only output cpu data, and
 * gpu ops can only output gpu data.
 *
 * When adding an op, the user specifies a name for its outputs, as well
 * as the device they will exist on (cpu or gpu). If a GPU op requests
 * a gpu version of a cpu tensor that has been produced earlier in the
 * graph, the pipeline will insert the needed operations to transfer
 * the data to the gpu.
 */
class DLL_PUBLIC Pipeline {
 public:
  /**
   * @brief Creates a pipeline that will produce batches of size `batch_size`,
   * using `num_threads` worker threads on gpu `device_id`.
   *
   * GPU memory and pinned memory allocations cause implicit synchronization of
   * the device, resulting in very slow startup times as dali buffer sizes
   * stabilize. To avoid this slowdown, we optionally take in an estimated size
   * of each image that will be processed in bytes. This hint is used to
   * pre-size buffers, potentially avoiding slow startup if the hint is close
   * to the true amount of memory that will be needed by the largest image to
   * be processed.
   *
   * @param batch_size the size of the batch that should be produced.
   * @param num_threads the number of threads to use in the prefetch stage.
   * @param device_id id of the GPU to operate on.
   * @param whether to allocate the necessary buffers to pipeline execution
   * between the cpu and gpu portions of the graph. See PipelinedExecutor.
   * @param whether to use extra host-threads to enable asynchronous issue
   * of cpu and gpu work. See AsyncExecutor/AsyncPipelinedExecutor.
   * @param bytes_per_sample_hint Estimated size of each sample to be processed.
   * Defaults to 0.
   * @param set_affinity indicates whether thread affinity should be
   * configured in the thread pool. Defaults to 'false'.
   * @param max_num_stream set an upper limit on the number of cudaStreams
   * that can be allocated by the pipeline.
   * @param prefetch_queue_depth sets the length of the executor internal pipeline
   */
  DLL_PUBLIC inline Pipeline(int batch_size, int num_threads, int device_id, int64_t seed = -1,
                             bool pipelined_execution = true, int prefetch_queue_depth = 2,
                             bool async_execution = true, size_t bytes_per_sample_hint = 0,
                             bool set_affinity = false, int max_num_stream = -1,
                             int default_cuda_stream_priority = 0)
      : built_(false), separated_execution_{false} {
    Init(batch_size, num_threads, device_id, seed, pipelined_execution, separated_execution_,
         async_execution, bytes_per_sample_hint, set_affinity, max_num_stream,
         default_cuda_stream_priority, QueueSizes{prefetch_queue_depth});
  }

  DLL_PUBLIC Pipeline(const string &serialized_pipe, int batch_size = -1, int num_threads = -1,
                      int device_id = -1, bool pipelined_execution = true,
                      int prefetch_queue_depth = 2, bool async_execution = true,
                      size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                      int max_num_stream = -1, int default_cuda_stream_priority = 0);

  DLL_PUBLIC ~Pipeline() = default;

  /**
   * @brief Creates a placeholder for an external input with the given name
   */
  DLL_PUBLIC inline void AddExternalInput(const string &name) {
    DALI_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    // Verify that this name is unique and record it
    auto it = edge_names_.find(name);
    DALI_ENFORCE(it == edge_names_.end(), "External input name '" +
        name + "' conflicts with existing intermediate result name");
    EdgeMeta meta;
    meta.has_cpu = true;
    meta.has_gpu = false;
    meta.has_contiguous = false;
    meta.is_support = false;
    DALI_ENFORCE(edge_names_.insert({name, meta}).second,
        "ExternalInput name insertion failure.");

    // Create a spec for an ExternalInput op and add it to our graph
    OpSpec spec =
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddOutput(name, "cpu");
    PrepareOpSpec(&spec);
    graph_.AddOp(spec, "__ExternalInput_" + name);
    external_inputs_.push_back(name);
  }

  /**
   * @brief Helper function for the SetExternalInput.
   */
  template <typename T>
  inline void SetExternalInputHelper(const string &name,
      const T &tl) {
    if (!graph_.TensorExists(name + "_cpu")) {
      // Trying to set data for non existing node is a noop
      return;
    }
    OpNodeId node_id = graph_.TensorSourceID(name + "_cpu");
    DALI_ENFORCE(graph_.NodeType(node_id) == OpType::CPU,
        "Internal error setting external input data.");

    auto &node = graph_.Node(node_id);
    auto *op_ptr = &node.InstantiateOperator();
    ExternalSource<CPUBackend> *source =
      dynamic_cast<ExternalSource<CPUBackend>*>(op_ptr);
    DALI_ENFORCE(source != nullptr, "Input name '" +
        name + "' is not marked as an external input.");
    source->SetDataSource(tl);
  }

  /**
   * @brief Sets the external input with the input name to the
   * input data.
   */
  DLL_PUBLIC inline void SetExternalInput(const string &name,
      const TensorList<CPUBackend> &tl) {
    SetExternalInputHelper(name, tl);
  }

  /**
   * @brief Sets the external input with the input name to the
   * input data.
   */
  DLL_PUBLIC inline void SetExternalInput(const string &name,
      const vector<Tensor<CPUBackend>> &tl) {
    SetExternalInputHelper(name, tl);
  }

  /**
   * @brief Adds an Operator with the input specification to the pipeline. The
   * 'device' argument in the OpSpec determines whether the CPU or GPU version
   * of the named operator will be added to the pipeline
   */
  DLL_PUBLIC void AddOperator(OpSpec spec, const std::string& inst_name = "<no name>");

  /**
   * @brief Returns the graph node with Operator
   * with a given name
   */
  DLL_PUBLIC OpNode * GetOperatorNode(const std::string& name);

  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution. The input
   * vector specifies the name and device of the desired outputs of the pipeline.
   */
  DLL_PUBLIC void Build(vector<std::pair<string, string>> output_names);

  /**
   * @brief Build a pipeline from deserialized output (name, device) pairs
   */
  DLL_PUBLIC void Build() {
    Build(this->output_names_);
  }

  /**
   * @brief Set execution characteristics for this Pipeline
   *
   * @param pipelined_execution Use pipelined execution
   * @param separated_execution Use separated queues
   * @param async_execution Use worker threads for RunX() functions
   */
  DLL_PUBLIC void SetExecutionTypes(bool pipelined_execution = true,
                                    bool separated_execution = false, bool async_execution = true) {
    DALI_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed - cannot change execution type.");
    pipelined_execution_ = pipelined_execution;
    separated_execution_ = separated_execution;
    async_execution_ = async_execution;
  }


  /**
   * @brief Set queue sizes for Pipeline using Separated Queues
   *
   * Must be called before Build()
   *
   * @param cpu_size
   * @param gpu_size
   */
  DLL_PUBLIC void SetQueueSizes(int cpu_size, int gpu_size) {
    DALI_ENFORCE(!built_,
                 "Alterations to the pipeline after "
                 "\"Build()\" has been called are not allowed - cannot set queue sizes.");
    DALI_ENFORCE(separated_execution_ || (cpu_size == gpu_size),
                 "Setting different queue sizes for non-separated execution is not allowed");
    DALI_ENFORCE(cpu_size > 0 && gpu_size > 0, "Only positive queue sizes allowed");
    prefetch_queue_depth_ = QueueSizes(cpu_size, gpu_size);
  }

  /*
   * @brief Set name output_names of the pipeline. Used to update the graph without
   * running the executor.
   */
  void SetOutputNames(vector<std::pair<string, string>> output_names);

  /**
   * @brief Run the cpu portion of the pipeline.
   */
  DLL_PUBLIC void RunCPU();

  /**
   * @brief Run the gpu portion of the pipeline.
   */
  DLL_PUBLIC void RunGPU();

  /**
   * @brief Sets completion callback which is called when GPU work is done
   * It blocks next GPU iteration so it is up to the developer to schedule
   * long lasting work in some thread and just fire the work from this CB
   */
  DLL_PUBLIC void SetCompletionCallback(ExecutorBase::ExecutorCallback cb);

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * Previously returned buffers are released.
   * This method blocks until the next batch is complete. RunCPU, RunMixed and RunGPU
   * must be called prior to calling this or this method will result in
   * deadlock.
   */
  DLL_PUBLIC void Outputs(DeviceWorkspace *ws);

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * To release previously returned buffers ReleaseOutputs need to be called.
   * This method blocks until the next batch is complete. RunCPU, RunMixed and RunGPU
   * must be called prior to calling this or this method will result in
   * deadlock.
   */
  DLL_PUBLIC void ShareOutputs(DeviceWorkspace *ws);

  /**
   * @brief Release buffers returned by the Output call
   * This method is meant for cases where buffers are coppied out
   * or consumed in any other way, so it is possible to set them free
   * before next Outputs call
   */
  DLL_PUBLIC void ReleaseOutputs();

  /**
   * @brief serializes the pipe to a protobuf
   */
  DLL_PUBLIC string SerializeToProtobuf() const;

  /**
   * @brief Save graph in DOT direct graph format
   * in filename.
   */
  DLL_PUBLIC void SaveGraphToDotFile(const std::string filename);

  /**
   * @brief Returns the batch size that will be produced by the pipeline.
   */
  DLL_PUBLIC inline int batch_size() const { return batch_size_; }

  /**
   * @brief Returns the map of (node name, node's epoch size)
   * for all nodes that return a valid epoch size
   */
  DLL_PUBLIC std::map<std::string, Index> EpochSize();

  /**
   * @brief Returns the number of threads used by the pipeline.
   */
  DLL_PUBLIC inline int num_threads() const { return num_threads_; }

  /**
   * @brief Returns the GPU device number used by the pipeline
   */
  DLL_PUBLIC inline int device_id() const { return device_id_; }

  // For testing
  template <typename T>
  friend class PipelineTest;

  DLL_PUBLIC DISABLE_COPY_MOVE_ASSIGN(Pipeline);

 private:
  /**
   * @brief Initializes the Pipeline internal state
   */
  void Init(int batch_size, int num_threads, int device_id, int64_t seed, bool pipelined_execution,
            bool separated_execution, bool async_execution, size_t bytes_per_sample_hint,
            bool set_affinity, int max_num_stream, int default_cuda_stream_priority,
            QueueSizes prefetch_queue_depth = QueueSizes{2}) {
    this->batch_size_ = batch_size;
    this->num_threads_ = num_threads;
    this->device_id_ = device_id;
    this->original_seed_ = seed;
    this->pipelined_execution_ = pipelined_execution;
    this->separated_execution_ = separated_execution;
    this->async_execution_ = async_execution;
    this->bytes_per_sample_hint_ = bytes_per_sample_hint;
    this->set_affinity_ = set_affinity;
    this->max_num_stream_ = max_num_stream;
    this->default_cuda_stream_priority_ = default_cuda_stream_priority;
    this->prefetch_queue_depth_ = prefetch_queue_depth;
    DALI_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0");

    int lowest_cuda_stream_priority, highest_cuda_stream_priority;
    CUDA_CALL(cudaDeviceGetStreamPriorityRange(&lowest_cuda_stream_priority,
                                               &highest_cuda_stream_priority));
    const auto min_priority_value =
        std::min(lowest_cuda_stream_priority, highest_cuda_stream_priority);
    const auto max_priority_value =
        std::max(lowest_cuda_stream_priority, highest_cuda_stream_priority);
    DALI_ENFORCE(
        default_cuda_stream_priority >= min_priority_value &&
        default_cuda_stream_priority <= max_priority_value,
        "Provided default cuda stream priority `" + std::to_string(default_cuda_stream_priority) +
        "` is outside the priority range [" + std::to_string(min_priority_value) + ", " +
        std::to_string(max_priority_value) + "], with lowest priority being `" +
        std::to_string(lowest_cuda_stream_priority) + "` and highest priority being `" +
        std::to_string(highest_cuda_stream_priority) + "`");

    seed_.resize(MAX_SEEDS);
    current_seed_ = 0;
    if (seed < 0) {
      using Clock = std::chrono::high_resolution_clock;
      seed = Clock::now().time_since_epoch().count();
    }
    std::seed_seq ss{seed};
    ss.generate(seed_.begin(), seed_.end());
  }

  using EdgeMeta = struct {
    bool has_cpu, has_gpu, has_contiguous, is_support;
  };

  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & ~7) + 8;
    }
    return base_ptr_offset;
  }

  void SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
      int input_idx, OpSpec *spec);

  void SetupGPUInput(std::map<string, EdgeMeta>::iterator it);

  inline EdgeMeta NewEdge(string device) {
    EdgeMeta edge;
    edge.has_cpu = false;
    edge.has_gpu = false;
    edge.has_contiguous = false;
    edge.is_support = false;
    if (device == "cpu") {
      edge.has_cpu = true;
    } else if (device == "gpu") {
      edge.has_gpu = true;
    } else if (device == "mixed") {
      edge.has_gpu = true;
      edge.has_contiguous = true;
    } else if (device == "support") {
      edge.has_cpu = true;
      edge.is_support = true;
      edge.has_contiguous = true;
    } else {
      DALI_FAIL("Invalid device argument \"" + device + "\". "
          "Valid options are \"cpu\", \"gpu\", \"mixed\" or \"support\"");
    }
    return edge;
  }

  // Helper to add pipeline meta-data
  void PrepareOpSpec(OpSpec *spec);

  void PropagateMemoryHint(OpNode &node);

  // Helper for nvJPEGDecoder split_stages special handling
  inline void AddSplitNvJPEGDecoder(OpSpec &spec, const std::string &inst_name);

  const int MAX_SEEDS = 1024;

  bool built_;
  int batch_size_, num_threads_, device_id_;
  bool pipelined_execution_;
  bool separated_execution_;
  bool async_execution_;
  size_t bytes_per_sample_hint_;
  int set_affinity_;
  int max_num_stream_;
  int default_cuda_stream_priority_;
  QueueSizes prefetch_queue_depth_;

  std::vector<int64_t> seed_;
  int original_seed_;
  size_t current_seed_;

  OpGraph graph_;
  std::unique_ptr<ExecutorBase> executor_;
  std::map<string, EdgeMeta> edge_names_;

  // store a list of all OpSpec and external inputs
  // added, in order to recreate the pipeline in a
  // serialized form
  vector<string> external_inputs_;
  vector<std::pair<string, OpSpec>> op_specs_;
  vector<bool> op_specs_to_serialize_;
  vector<std::pair<string, string>> output_names_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_H_
