// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/pipeline_output_desc.h"
#include "dali/pipeline/operator/builtin/external_source.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"


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
   * @param max_batch_size the maximum size of the batch that can be produced.
   * @param num_threads the number of threads to use in the prefetch stage.
   * @param device_id id of the GPU to operate on.
   * @param seed used for random number generation. Leaving the default value
   * for this parameter results in random seed
   * @param pipelined_execution whether to allocate the necessary buffers for pipeline execution
   * between the cpu and gpu portions of the graph. See PipelinedExecutor.
   * @param prefetch_queue_depth sets the length of the executor internal pipeline
   * @param async_execution whether to use extra host-threads to enable asynchronous execution
   * of cpu and gpu work. See AsyncExecutor/AsyncPipelinedExecutor.
   * @param bytes_per_sample_hint Estimated size of each sample to be processed.
   * Defaults to 0.
   * @param set_affinity indicates whether thread affinity should be
   * configured in the thread pool. Defaults to 'false'.
   * @param max_num_stream set an upper limit on the number of cudaStreams
   * that can be allocated by the pipeline.
   * @param default_cuda_stream_priority  CUDA stream priority used by DALI.
   * See `cudaStreamCreateWithPriority` in CUDA documentation
   */
  DLL_PUBLIC Pipeline(int max_batch_size, int num_threads, int device_id, int64_t seed = -1,
                      bool pipelined_execution = true, int prefetch_queue_depth = 2,
                      bool async_execution = true, size_t bytes_per_sample_hint = 0,
                      bool set_affinity = false, int max_num_stream = -1,
                      int default_cuda_stream_priority = 0);

  DLL_PUBLIC Pipeline(const string &serialized_pipe, int max_batch_size = -1, int num_threads = -1,
                      int device_id = -1, bool pipelined_execution = true,
                      int prefetch_queue_depth = 2, bool async_execution = true,
                      size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                      int max_num_stream = -1, int default_cuda_stream_priority = 0,
                      int64_t seed = -1);

  virtual DLL_PUBLIC ~Pipeline();

  /**
   * @brief Creates a placeholder for an External Source operator with the given name
   * (and output of given name).
   *
   * Equivalent to inserting ExternalSource with output of given name and specified
   * device placemnt.
   */
  DLL_PUBLIC int AddExternalInput(const string &name,
                                         const string &device = "cpu",
                                         DALIDataType dtype = DALI_NO_TYPE,
                                         int ndim = -1,
                                         const TensorLayout &layout = "") {
    auto spec = OpSpec("ExternalSource")
                      .AddArg("name", name)
                      .AddArg("device", device)
                      .AddOutput(name, device);
    if (!layout.empty()) spec.AddArg("layout", layout);
    if (ndim >= 0) spec.AddArg("ndim", ndim);
    if (dtype != DALI_NO_TYPE) spec.AddArg("dtype", dtype);
    return AddOperator(spec, name);
  }


  template<typename TensorListBackend, typename OperatorBackend>
  void SetDataSourceHelper(
          const string &name, const TensorList<TensorListBackend> &tl,
          std::optional<std::string> data_id, OperatorBase *op_ptr, AccessOrder order = {},
          InputOperatorSettingMode in_op_setting_mode = {},
          bool is_refeeding = false) {
    if (is_refeeding ||
        !repeat_last_.SetLast<OperatorBackend>(name, tl, data_id, order, in_op_setting_mode)) {
      auto *source = dynamic_cast<InputOperator<OperatorBackend> *>(op_ptr);
      DALI_ENFORCE(source != nullptr,
                  "Input name '" + name + "' is not marked as an InputOperator.");
      source->template SetDataSource<TensorListBackend>(tl, order, in_op_setting_mode,
                                                        std::move(data_id));
    }
  }


  /**
   * @brief Helper function for the SetExternalInput.
   * @tparam Backend CPUBackend or GPUBackend
   * @param name name of the input
   * @param tl data
   * @param order synchronization order (CUDA stream or host)
   * @param ext_src_setting_mode Options passed to the External Source describing the behaviour
   *                        of setting the data.
   */
  template<typename Backend>
  void SetExternalInputHelper(const string &name, const TensorList<Backend> &tl,
                              std::optional<std::string> data_id, AccessOrder order = {},
                              InputOperatorSettingMode ext_src_setting_mode = {},
                              bool is_refeeding = false) {
    auto *node = GetInputOperatorNode(name);
    OperatorBase *op_ptr = node->op.get();

    switch (node->op_type) {
      case OpType::CPU:
        SetDataSourceHelper<Backend, CPUBackend>(name, tl, std::move(data_id), op_ptr, order,
                                                 ext_src_setting_mode, is_refeeding);
        break;
      case OpType::MIXED:
        SetDataSourceHelper<Backend, MixedBackend>(name, tl, std::move(data_id), op_ptr, order,
                                                   ext_src_setting_mode, is_refeeding);
        break;
      case OpType::GPU:
        SetDataSourceHelper<Backend, GPUBackend>(name, tl, std::move(data_id), op_ptr, order,
                                                 ext_src_setting_mode, is_refeeding);
        break;
      default:
        assert(false);  // This shouldn't happen.
    }
  }


  /**
   * @brief Sets the external input with the input name to the input data
   * @tparam Backend
   * @param name name of the input
   * @param tl data
   * @param order synchronization order (CUDA stream or host)
   * @param sync If SetExternalInputHelper should be blocking - waits until provided data is copied
   *             to the internal buffer
   * @param no_copy_mode Select whether to use the parameter defined in the External Source or
   *                     override the mode of operation forcing the copy or no-copy
   */
  template<typename Backend>
  DLL_PUBLIC void
  SetExternalInput(const string &name, const TensorList<Backend> &tl, AccessOrder order = {},
                   bool sync = false, bool use_copy_kernel = false,
                   InputOperatorNoCopyMode no_copy_mode = InputOperatorNoCopyMode::DEFAULT,
                   std::optional<std::string> data_id = std::nullopt) {
    InputOperatorSettingMode mode{sync, use_copy_kernel, no_copy_mode};
    // if SetLast succeeds, the data will be forcibly _shared_ (zero copy) upon Refeed
    SetExternalInputHelper(name, tl, std::move(data_id), order, mode, false);
  }


  /**
   * @brief  Adds an Operator with the input specification to the pipeline. The
   * 'device' argument in the OpSpec determines whether the CPU or GPU version
   * of the named operator will be added to the pipeline
   *
   * @param spec
   * @param inst_name
   * @param logical_id Allows to group operator that are supposed to have synchronized state
   * wrt randomness. Operators sharing the logical_id will have the same seed assigned.
   *
   * @return logical_id of added operator, so it can be used for further calls
   */
  DLL_PUBLIC int AddOperator(const OpSpec &spec, const std::string& inst_name, int logical_id);

  /**
   * @brief Adds an Operator with the input specification to the pipeline. It will be assigned
   * a separate logical_id based on internal state of the pipeline.
   */
  DLL_PUBLIC int AddOperator(const OpSpec &spec, const std::string& inst_name);

  /**
   * @brief Adds an unnamed Operator with the input specification to the pipeline.
   */
  DLL_PUBLIC int AddOperator(const OpSpec &spec, int logical_id);

  /**
   * @brief Adds an unnamed Operator with the input specification to the pipeline.  It will be
   * assigned a separate logical_id based on internal state of the pipeline.
   */
  DLL_PUBLIC int AddOperator(const OpSpec &spec);

  /**
   * @brief Returns true if there exists operator with given logical_id
   */
  DLL_PUBLIC bool IsLogicalIdUsed(int logical_id) const;

  /**
   * @brief Returns the graph node with Operator
   * with a given name
   */
  DLL_PUBLIC OpNode * GetOperatorNode(const std::string& name);

  /**
   * @brief Rreturns an input graph node with a given name
   */
  DLL_PUBLIC const OpNode *GetInputOperatorNode(const std::string &name);

  /** @{ */
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution. The input
   * vector specifies the name and device of the desired outputs of the pipeline.
   */
  DLL_PUBLIC void Build(const std::vector<std::pair<string, string>>& output_names);
  DLL_PUBLIC void Build(std::vector<PipelineOutputDesc> output_descs);
  /** @} */

  /**
   * @brief Build a pipeline from deserialized output (name, device) pairs
   */
  DLL_PUBLIC void Build();

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
   * @brief Set if the DALI pipeline should gather executor statistics of the operator ouput sizes
   *
   * @param enable_memory_stats If statistics should be gathered
   * Useful for `bytes_per_sample_hint` operator parameter.
   */
  DLL_PUBLIC void EnableExecutorMemoryStats(bool enable_memory_stats = true) {
    enable_memory_stats_ = enable_memory_stats;
    if (executor_) {
      executor_->EnableMemoryStats(enable_memory_stats_);
    }
  }

  /**
   * @brief Set if the DALI pipeline should create checkpoints between the epochs
   *
   * @param enable_memory_stats If checkpoints should be created
   */
  DLL_PUBLIC void EnableCheckpointing(bool checkpointing = true) {
    checkpointing_ = checkpointing;
    if (executor_) {
      executor_->EnableCheckpointing(checkpointing_);
    }
  }

  /**
   * @brief Returns a serialized Checkpoint
   *
   * @param external_ctx_cpt Additional information from python side to be included
   */
  DLL_PUBLIC string SerializedCheckpoint(const ExternalContextCheckpoint &external_ctx_cpt) const {
    auto cpt = GetCheckpoint();
    cpt.external_ctx_cpt_ = external_ctx_cpt;
    return cpt.SerializeToProtobuf(graph_);
  }

  /**
   * @brief Returns an unserialized Checkpoint
  */
  DLL_PUBLIC Checkpoint GetCheckpoint() const {
    DALI_ENFORCE(executor_, "Pipeline must be built before it can produce a checkpoint. ");
    DALI_ENFORCE(checkpointing_,
                 "Cannot save the checkpoint. The `enable_checkpointing` was not "
                 "specified when creating the pipeline");
    auto &cpt = executor_->GetCurrentCheckpoint();
    // Make sure the checkpoint is accessible on host
    cpt.SetOrder(AccessOrder::host());
    return cpt;
  }

  /**
   * @brief Restores pipeline state from a serialized Checkpoint
   *
   * Should be called before building.
   *
   * @return Extra context which was passed to SerializedCheckpoint when creating the checkpoint
  */
  DLL_PUBLIC ExternalContextCheckpoint RestoreFromSerializedCheckpoint(
      const std::string &serialized_checkpoint) {
    DALI_ENFORCE(checkpointing_,
                 "Cannot restore checkpoint. The `enable_checkpointing` was not "
                 "specified when creating the pipeline");
    Checkpoint cpt;
    cpt.DeserializeFromProtobuf(graph_, serialized_checkpoint);
    RestoreFromCheckpoint(cpt);
    return cpt.external_ctx_cpt_;
  }

  /**
   * @brief Restores pipeline state from an unserialized Checkpoint
   *
   * Should be called before building.
  */
  DLL_PUBLIC void RestoreFromCheckpoint(const Checkpoint &cpt) {
    executor_->RestoreStateFromCheckpoint(cpt);
  }

  /**
   * @brief Obtains the executor statistics
   */
  DLL_PUBLIC ExecutorMetaMap GetExecutorMeta() {
    if (executor_) {
      return executor_->GetExecutorMeta();
    } else {
      return {};
    }
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

  /** @{ */
  /**
   * @brief Set descriptors of the outputs of the pipeline. Used to update the graph without
   * running the executor and for pipeline serialization.
   */
  void SetOutputDescs(std::vector<PipelineOutputDesc> output_descs);
  /**
   * Convenience overload. Set only the name and device of an output, since the dtype and ndim
   * are not always necessary. This function can't reset the output descriptors. If they were already
   * set, the function will fail.
   */
  void SetOutputDescs(const vector<std::pair<string /* name */, string /* device */>> &out_names);
  /** @} */

  /**
   * @brief Run the cpu portion of the pipeline.
   */
  DLL_PUBLIC void RunCPU();

  /**
   * @brief Run the gpu portion of the pipeline.
   */
  DLL_PUBLIC void RunGPU();

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * Previously returned buffers are released.
   * This method blocks until the next batch is complete. RunCPU, RunMixed and RunGPU
   * must be called prior to calling this or this method will result in
   * deadlock.
   */
  DLL_PUBLIC void Outputs(Workspace *ws);

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * To release previously returned buffers ReleaseOutputs need to be called.
   * This method blocks until the next batch is complete. RunCPU, RunMixed and RunGPU
   * must be called prior to calling this or this method will result in
   * deadlock.
   */
  DLL_PUBLIC void ShareOutputs(Workspace *ws);

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
  DLL_PUBLIC void SaveGraphToDotFile(const std::string &filename, bool show_tensors = false,
                                     bool show_ids = false, bool use_colors = false);

  /** @{ */
  /**
   * @brief Returns the maximum batch size that can be processed by the Pipeline
   */
  DLL_PUBLIC inline int batch_size() const { return max_batch_size_; }
  DLL_PUBLIC inline int max_batch_size() const { return max_batch_size_; }
  /** @} */

  /**
   * @brief Returns the map of (node name, reader meta) for all nodes that return a valid meta
   */
  DLL_PUBLIC std::map<std::string, ReaderMeta> GetReaderMeta();

  /**
   * @brief Returns the reader meta for a node with given name
   */
  DLL_PUBLIC ReaderMeta GetReaderMeta(std::string name);

  /**
   * @brief Get the data layout required by the external input with a given name.
   */
  DLL_PUBLIC const TensorLayout &GetInputLayout(const std::string &name);

  /**
   * @brief Get the required number of dimensions for the external input with a given name.
   */
  DLL_PUBLIC int GetInputNdim(const std::string &name);

  /**
   * @brief Get the data type required by the external input with a given name.
   */
  DLL_PUBLIC DALIDataType GetInputDtype(const std::string &name);

  /**
   * @brief Returns the number of threads used by the pipeline.
   */
  DLL_PUBLIC inline int num_threads() const { return num_threads_; }

  /**
   * @brief Returns the GPU device number used by the pipeline
   */
  DLL_PUBLIC inline int device_id() const {
    return device_id_;
  }

  /**
   * @brief Returns number of external inputs.
   */
  DLL_PUBLIC int num_inputs() const;

  /**
   * @brief Returns number of outputs.
   */
  DLL_PUBLIC int num_outputs() const;

  /**
   * @brief Return the name of the nth external input in the pipeline in lexicographic order.
   */
  DLL_PUBLIC const std::string &input_name(int n) const;

  /**
   * @brief Returns a string describing the name of the output specified by given id.
   */
  DLL_PUBLIC const std::string &output_name(int id) const;

  /**
   * @brief Returns a string describing the device type backing the output specified by given id.
   */
  DLL_PUBLIC const std::string &output_device(int id) const;

  /**
   * @brief Returns data type of the output specified by given id.
   */
  DLL_PUBLIC DALIDataType output_dtype(int id) const;

  /**
   * @brief Returns number of dimensions in the output specified by given id.
   */
  DLL_PUBLIC int output_ndim(int id) const;

  /**
   * @brief Returns output descriptors for all outputs.
   */
  DLL_PUBLIC std::vector<PipelineOutputDesc> output_descs() const;

  /**
   * Checks, if a provided pipeline can be deserialized, according to the Pipeline protobuf
   * definition.
   *
   * @param serialized_pipeline
   * @return True, if the pipeline is serializable. False otherwise.
   */
  static bool IsDeserializable(const std::string &serialized_pipeline);

  /**
   * @brief Shutdown the executor
   */
  DLL_PUBLIC void Shutdown();

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
            QueueSizes prefetch_queue_depth = QueueSizes{2});

  using EdgeMeta = struct {
    bool has_cpu;
    bool has_gpu;
    bool has_contiguous;
    // MakeContiguous was added after this node to be used as output on specified device:
    bool has_make_contiguous_cpu;
    bool has_make_contiguous_gpu;
  };

  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & ~7) + 8;
    }
    return base_ptr_offset;
  }

  void SetupCPUInput(std::map<string, EdgeMeta>::iterator it, int input_idx, OpSpec *spec);

  void SetupGPUInput(std::map<string, EdgeMeta>::iterator it);

  inline EdgeMeta NewEdge(const std::string &device) {
    EdgeMeta edge;
    edge.has_cpu = false;
    edge.has_gpu = false;
    edge.has_contiguous = false;
    edge.has_make_contiguous_cpu = false;
    edge.has_make_contiguous_gpu = false;
    if (device == "cpu") {
      edge.has_cpu = true;
    } else if (device == "gpu") {
      edge.has_gpu = true;
    } else if (device == "mixed") {
      edge.has_gpu = true;
      edge.has_contiguous = true;
    } else {
      DALI_FAIL("Invalid device argument \"" + device + "\". "
          "Valid options are \"cpu\", \"gpu\" or \"mixed\".");
    }
    return edge;
  }

  // Helper to add pipeline meta-data
  void PrepareOpSpec(OpSpec *spec, int logical_id);

  void PropagateMemoryHint(OpNode &node);

  inline void AddToOpSpecs(const std::string &inst_name, const OpSpec &spec, int logical_id);

  int GetNextLogicalId();
  int GetNextInternalLogicalId();

  /**
   * Validate, that the outputs from the Pipeline match the criteria.
   * @return True, if the outputs passed the validation test.
   */
  bool ValidateOutputs(const Workspace &ws) const;

  /**
   * @brief Prepare the OpSpec and generate operator name and output name for a specified
   * MakeContiguous node.
   *
   * Note that inserting mixed MakeContiguous for cpu -> gpu transfer has special rules regarding
   * output naming.
   *
   * @param meta the output edge - that is edge from the operator to tensor that we need to
   * insert MakeContiguous after
   * @param input_name Name of the input Tensor node to the MakeContiguous
   * @param input_dev Device placement of the input Tensor node
   * @param device Device of the requested MakeContiguous node.
   * @param output_dev Placement of the requested output data from the MakeContiguous.
   * For given MakeContiguous device, we have following possible outputs:
   *  * "mixed" -> "cpu", "gpu"
   *  * "gpu" -> "gpu"
   * @return std::tuple<OpSpec, string, string> Operator OpSpec, Operator Name, Output Name
   */
  std::tuple<OpSpec, std::string, std::string> PrepareMakeContiguousNode(
      EdgeMeta &meta, const std::string &input_name, const std::string &input_dev,
      const std::string &device, const std::string &output_dev);

  /**
   * @brief Add new MakeContiguous node (if one does not exist yet) for the requested output Edge
   *
   * @param meta the output edge - that is edge from the operator to tensor that we need to
   * insert MakeContiguous after
   * @param input_name Name of the input Tensor node to the MakeContiguous
   * @param input_dev Device placement of the input Tensor node
   * @param device Device of the requested MakeContiguous node.
   * @param output_dev Placement of the requested output data from the MakeContiguous.
   * For given MakeContiguous device, we have following possible outputs:
   *  * "mixed" -> "cpu", "gpu"
   *  * "gpu" -> "gpu"
   * @return The name of the output of the MakeContiguous node that replaces the requested output.
   */
  std::string AddMakeContiguousNode(EdgeMeta &meta, const std::string &input_name,
                                    const std::string &input_dev, const std::string &device,
                                    const std::string &output_dev);

  /**
   * Traverses the Operator graph and collects all operators that are Input Operators.
   */
  void DiscoverInputOperators();

  const int MAX_SEEDS = 1024;

  bool built_;
  int max_batch_size_, num_threads_, device_id_;
  bool pipelined_execution_;
  bool separated_execution_;
  bool async_execution_;
  size_t bytes_per_sample_hint_;
  int set_affinity_;
  int max_num_stream_;
  int default_cuda_stream_priority_;
  int next_logical_id_ = 0;
  int next_internal_logical_id_ = -1;
  QueueSizes prefetch_queue_depth_;
  bool enable_memory_stats_ = false;
  bool checkpointing_ = false;

  std::vector<int64_t> seed_;
  int original_seed_;
  size_t current_seed_;

  std::unique_ptr<ExecutorBase> executor_;
  OpGraph graph_;
  std::map<string, EdgeMeta> edge_names_;

  struct OpDefinition {
    std::string instance_name;
    OpSpec spec;
    int logical_id;
  };

  std::vector<OpDefinition> op_specs_;
  std::vector<OpDefinition> op_specs_for_serialization_;

  std::vector<PipelineOutputDesc> output_descs_;

  // Mapping between logical id and index in op_specs_
  std::map<int, std::vector<size_t>> logical_ids_;
  std::map<int, int64_t> logical_id_to_seed_;

  // input operators are sorted by names
  std::map<std::string, const OpNode*> input_operators_;

  /**
   * @brief Handles repeating recent inputs for ExternalSource nodes with repeat_last flag on
   *
   * ExternalSource nodes can specify a repeat_last flag which works by re-submitting the most
   * recently fed input in case where no new data was fed between calls to Pipeline::Run.
   *
   * This class maintains a list of such nodes, stores the most recently fed input and re-submits
   * it if no new data was fed.
   */
  struct RepeatLastInputs {
    void FindNodes(const OpGraph &graph);

    template <typename OperatorBackend, typename DataBackend>
    bool SetLast(const std::string &name, const TensorList<DataBackend> &data,
                 const std::optional<std::string> &data_id,
                 AccessOrder order,
                 InputOperatorSettingMode ext_src_setting_mode) {
      auto &nodes = GetNodes<OperatorBackend>();
      auto it = nodes.find(name);
      if (it == nodes.end())
        return false;

      auto &node = it->second;
      auto &inp = dynamic_cast<InputOperator<OperatorBackend>&>(*node.op_node->op);

      auto do_copy = [&]() {
        node.last_input.Reset();
        node.last_input.set_order(order);
        node.last_input.Copy(data, order, ext_src_setting_mode.use_copy_kernel);
        if (ext_src_setting_mode.sync)
          AccessOrder::host().wait(order);
      };

      if constexpr (std::is_same_v<OperatorBackend, DataBackend>) {
        if (inp.WouldCopy(ext_src_setting_mode.no_copy_mode)) {
          do_copy();
        } else {
          node.last_input.ShareData(data);
        }
      } else {
        do_copy();
      }
      node.data_id = data_id;

      return true;
    }

    template <typename Backend>
    void Refeed(Pipeline &owner);

    template <typename Backend>
    struct RepeatLastInput {
      const OpNode *op_node = nullptr;
      using InputBackend = std::conditional_t<std::is_same_v<Backend, MixedBackend>,
                                             CPUBackend, Backend>;
      TensorList<InputBackend> last_input;
      std::optional<std::string> data_id;
    };

    std::map<std::string, RepeatLastInput<CPUBackend>> cpu_nodes_;
    std::map<std::string, RepeatLastInput<GPUBackend>> gpu_nodes_;
    std::map<std::string, RepeatLastInput<MixedBackend>> mixed_nodes_;

    template <typename Backend>
    std::map<std::string, RepeatLastInput<Backend>> &GetNodes() {
      if constexpr (std::is_same_v<Backend, CPUBackend>)
        return cpu_nodes_;
      else if constexpr (std::is_same_v<Backend, GPUBackend>)
        return gpu_nodes_;
      else
        return mixed_nodes_;
    }
  };

  RepeatLastInputs repeat_last_;
};

template <typename Backend>
void Pipeline::RepeatLastInputs::Refeed(Pipeline &owner) {
  auto &nodes = GetNodes<Backend>();
  for (auto &[name, node] : nodes) {
    owner.SetExternalInputHelper(name, node.last_input, node.data_id, node.last_input.order(),
      InputOperatorSettingMode{false, false, InputOperatorNoCopyMode::FORCE_NO_COPY},
      true);
  }
}

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_H_
