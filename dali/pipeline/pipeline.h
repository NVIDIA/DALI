// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <exception>
#include <functional>
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
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/pipeline_output_desc.h"
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"
#include "dali/pipeline/pipeline_params.h"

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
   * @warning This constructor is deprecated. Use Pipeline(const PipelineParams &params) instead.
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
   * @param dynamic_execution whether to use the Executor2, enabling GPU->CPU transfers
   * and dynamic allocation of memory.
   * @param bytes_per_sample_hint Estimated size of each sample to be processed.
   * Defaults to 0. Ignored when dynamic_execution is true.
   * @param set_affinity indicates whether thread affinity should be
   * configured in the thread pool. Defaults to 'false'.
   */
  DLL_PUBLIC Pipeline(int max_batch_size, int num_threads, int device_id, int64_t seed = -1,
                      bool pipelined_execution = true, int prefetch_queue_depth = 2,
                      bool async_execution = true, bool dynamic_execution = false,
                      size_t bytes_per_sample_hint = 0, bool set_affinity = false);

  /**
   * @warning This constructor is deprecated. Use
   *          Pipeline(const string &serialized_pipe, const PipelineParams &param_override) instead.
   */
  DLL_PUBLIC Pipeline(const string &serialized_pipe,
                      int max_batch_size = -1, int num_threads = -1, int device_id = -1,
                      bool pipelined_execution = true, int prefetch_queue_depth = 2,
                      bool async_execution = true, bool dynamic_execution = false,
                      size_t bytes_per_sample_hint = 0, bool set_affinity = false,
                      int64_t seed = -1);

  /** Constructs a pipeline with parameters specified in the PipelineParams structure. */
  DLL_PUBLIC Pipeline(const PipelineParams &params);

  /** Constructs a pipeline from a serialized pipeline, optionally overriding the parameters. */
  DLL_PUBLIC Pipeline(const string &serialized_pipe, const PipelineParams &param_override);

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
                      .AddOutput(name, ParseStorageDevice(device));
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
    if (node == nullptr)
      throw invalid_key(make_string("Could not find an input operator with name \"", name, "\""));
    OperatorBase *op_ptr = executor_->GetOperator(name);
    assert(op_ptr != nullptr);

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
   * @param copy_mode Select whether to use the parameter defined in the External Source or
   *                  override the mode of operation forcing the copy or no-copy
   */
  template<typename Backend>
  DLL_PUBLIC void
  SetExternalInput(const string &name, const TensorList<Backend> &tl, AccessOrder order = {},
                   bool sync = false, bool use_copy_kernel = false,
                   InputOperatorCopyMode copy_mode = InputOperatorCopyMode::DEFAULT,
                   std::optional<std::string> data_id = std::nullopt) {
    InputOperatorSettingMode mode{sync, use_copy_kernel, copy_mode};
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
  DLL_PUBLIC int AddOperator(const OpSpec &spec, std::string_view inst_name, int logical_id);

  /**
   * @brief Adds an Operator with the input specification to the pipeline. It will be assigned
   * a separate logical_id based on internal state of the pipeline.
   */
  DLL_PUBLIC int AddOperator(const OpSpec &spec, std::string_view inst_name);

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
   * @brief Returns the graph node describing an operator with the given name.
   */
  DLL_PUBLIC graph::OpNode *GetOperatorNode(std::string_view instance_name);

  /**
   * @brief Returns an operator instance with given name or nullptr, if not found.
   *
   * NOTE: Some operators may be dropped or replaced during graph pruning and optimization.
   */
  DLL_PUBLIC OperatorBase *GetOperator(std::string_view instance_name);

  /**
   * @brief Returns an input graph node with a given name
   */
  DLL_PUBLIC const graph::OpNode *GetInputOperatorNode(std::string_view name);

  /**
   * @brief Get input operatos as a name-to-node mapping.
   *
   */
  DLL_PUBLIC const auto &GetInputOperators() const & {
    DALI_ENFORCE(built_);
    return input_operators_;
  }

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
   * @brief Set if the DALI pipeline should create checkpoints between the epochs
   *
   * @param enable_memory_stats If checkpoints should be created
   */
  DLL_PUBLIC void EnableCheckpointing(bool checkpointing = true) {
    params_.enable_checkpointing = checkpointing;
    if (executor_) {
      executor_->EnableCheckpointing(checkpointing);
    }
  }

  /**
   * @brief Returns an unserialized Checkpoint
  */
  DLL_PUBLIC Checkpoint GetCheckpoint() const {
    DALI_ENFORCE(executor_, "Pipeline must be built before it can produce a checkpoint. ");
    DALI_ENFORCE(checkpointing_enabled(),
                 "Cannot save the checkpoint. The `enable_checkpointing` was not "
                 "specified when creating the pipeline");
    auto &cpt = executor_->GetCurrentCheckpoint();
    // Make sure the checkpoint is accessible on host
    cpt.SetOrder(AccessOrder::host());
    return cpt;
  }

  DLL_PUBLIC string SerializeCheckpoint(const Checkpoint &cpt) const {
    return cpt.SerializeToProtobuf(*executor_);
  }

  /**
   * @brief Returns a serialized Checkpoint
   *
   * @param external_ctx_cpt Additional information from python side to be included
   */
  DLL_PUBLIC string
  GetSerializedCheckpoint(const ExternalContextCheckpoint &external_ctx_cpt) const {
    auto cpt = GetCheckpoint();
    cpt.external_ctx_cpt_ = external_ctx_cpt;
    return cpt.SerializeToProtobuf(*executor_);
  }

  /**
   * @brief Reconstitutes a checkpoint from a serialzied representation.
   */
  DLL_PUBLIC Checkpoint DeserializeCheckpoint(std::string_view serialized_checkpoint) const {
    Checkpoint cpt;
    cpt.DeserializeFromProtobuf(*executor_, serialized_checkpoint);
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
      std::string_view serialized_checkpoint) {
    DALI_ENFORCE(checkpointing_enabled(),
                 "Cannot restore checkpoint. The `enable_checkpointing` was not "
                 "specified when creating the pipeline");
    Checkpoint cpt = DeserializeCheckpoint(serialized_checkpoint);
    RestoreFromCheckpoint(cpt);
    return cpt.external_ctx_cpt_;
  }

  /**
   * @brief Restores pipeline state from an unserialized Checkpoint
   *
   * Should be called before building.
  */
  DLL_PUBLIC void RestoreFromCheckpoint(const Checkpoint &cpt) {
    DALI_ENFORCE(checkpointing_enabled(),
                 "Cannot restore checkpoint. The `enable_checkpointing` was not "
                 "specified when creating the pipeline");
    DALI_ENFORCE(executor_, "The pipeline must be built before restoring its state.");
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

  DLL_PUBLIC QueueSizes GetQueueSizes() const {
    return *params_.prefetch_queue_depths;
  }

  /**
   * @brief Returns the parameters with which the pipeline was created
   */
  DLL_PUBLIC const PipelineParams &GetParams() const & {
    return params_;
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
   * @brief Run the pipeline
   */
  DLL_PUBLIC void Run();

  /**
   * @brief Fills the prefetch queues
   *
   * Runs a prefetching function in the executor so that internal and output queues are full.
   * Note that it requires populating the external sources InputFeedCount(name) times.
   */
  DLL_PUBLIC void Prefetch();

  /**
   * @brief Calculates how many times a given input must be populated before the pipeline can be run
   *
   * @param input_name The name of the input, as specified in the input operator.
   * @return The number of times that feed_input needs to be called.
   */
  DLL_PUBLIC int InputFeedCount(std::string_view input_name);

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * Previously returned buffers are released.
   * This method blocks until the next batch is complete. Run must be called prior to calling this
   * method or it will result in a deadlock.
   */
  DLL_PUBLIC void Outputs(Workspace *ws);

  /**
   * @brief Fills the input device workspace with the output of the pipeline.
   * To release previously returned buffers ReleaseOutputs need to be called.
   * This method blocks until the next batch is complete. Run must be called prior to calling this
   * method or it will result in a deadlock.
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
                                     bool use_colors = false);

  /** @{ */
  /**
   * @brief Returns the maximum batch size that can be processed by the Pipeline
   */
  DLL_PUBLIC inline int batch_size() const { return max_batch_size(); }
  DLL_PUBLIC inline int max_batch_size() const { return params_.max_batch_size.value_or(-1); }
  /** @} */

  /**
   * @brief Returns the map of (node name, reader meta) for all nodes that return a valid meta
   */
  DLL_PUBLIC std::map<std::string_view, ReaderMeta, std::less<>> GetReaderMeta();

  /**
   * @brief Returns the reader meta for a node with given name
   */
  DLL_PUBLIC ReaderMeta GetReaderMeta(std::string_view name);

  /**
   * @brief Get the data layout required by the external input with a given name.
   */
  DLL_PUBLIC const TensorLayout &GetInputLayout(std::string_view name);

  /**
   * @brief Get the required number of dimensions for the external input with a given name.
   */
  DLL_PUBLIC int GetInputNdim(std::string_view name);

  /**
   * @brief Get the data type required by the external input with a given name.
   */
  DLL_PUBLIC DALIDataType GetInputDtype(std::string_view name);

  /**
   * @brief Returns the number of threads used by the pipeline.
   */
  DLL_PUBLIC inline int num_threads() const { return params_.num_threads.value_or(-1); }

  /**
   * @brief Returns the GPU device number used by the pipeline
   */
  DLL_PUBLIC inline int device_id() const {
    return params_.device_id.value_or(CPU_ONLY_DEVICE_ID);
  }

  /**
   * @brief Returns whether the pipeline requires a CUDA-capable GPU to run.
   */
  DLL_PUBLIC inline bool requires_gpu() const {
    return requires_gpu_;
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
  DLL_PUBLIC StorageDevice output_device(int id) const;

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
  DLL_PUBLIC const std::vector<PipelineOutputDesc> &output_descs() const &;

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
  /**  Initializes the Pipeline internal state */
  void Init(const PipelineParams &params);
  /** Validate the Pipeline parameters */
  static void Validate(const PipelineParams &params);

  struct EdgeMeta {
    bool has_cpu;
    bool has_gpu;
    // Whether the given backend is guaranteed to have contiguous storage
    bool has_contiguous_cpu;
    bool has_contiguous_gpu;
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

  /**
   * @brief See Pipeline::AddOperator for details.
   *
   * Does the internal processing allowing the errors to be processed once.
   * Assumes that Build() has not been called.
   */
  int AddOperatorImpl(const OpSpec &spec, std::string_view inst_name, int logical_id);

  void ToCPU(std::map<string, EdgeMeta>::iterator it);
  void ToGPU(std::map<string, EdgeMeta>::iterator it);

  inline EdgeMeta NewEdge(StorageDevice device) {
    EdgeMeta edge{};
    if (device == StorageDevice::CPU) {
      edge.has_cpu = true;
    } else if (device == StorageDevice::GPU) {
      edge.has_gpu = true;
    } else {
      assert(!"Unreachable code");
    }
    return edge;
  }

  // Helper to add pipeline meta-data
  void PrepareOpSpec(OpSpec *spec, int logical_id);

  void PropagateMemoryHint(graph::OpNode &node);

  inline void AddToOpSpecs(std::string_view inst_name, const OpSpec &spec, int logical_id);

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
      EdgeMeta &meta,
      std::string_view input_name,
      StorageDevice input_dev,
      std::string_view device,
      StorageDevice output_dev);

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
  std::string AddMakeContiguousNode(EdgeMeta &meta,
                                    std::string_view input_name,
                                    StorageDevice input_dev,
                                    std::string_view device,
                                    StorageDevice output_dev);

  /**
   * Traverses the Operator graph and collects all operators that are Input Operators.
   *
   * NOTE: This function must not throw - if it does, the pipeline may be left in an
   *       inconsistent state!
   */
  void DiscoverInputOperators() noexcept;

  /**
   * @brief Process exception that was thrown when executing DALI. Executor already provided context
   * for operator if possible.
   */
  void ProcessException(std::exception_ptr eptr);

  const int MAX_SEEDS = 1024;

  static DLL_PUBLIC PipelineParams DefaultParams();

  bool built_ = false;
  PipelineParams params_ = DefaultParams();

  inline bool pipelined_execution() const {
    return Test(*params_.executor_type, ExecutorType::PipelinedFlag);
  }
  inline bool async_execution() const {
    return Test(*params_.executor_type, ExecutorType::AsyncFlag);
  }
  inline bool dynamic_execution() const {
    return Test(*params_.executor_type, ExecutorType::DynamicFlag);
  }
  inline bool separated_execution() const {
    return Test(*params_.executor_type, ExecutorType::SeparatedFlag);
  }

  inline bool checkpointing_enabled() const { return params_.enable_checkpointing.value_or(false); }
  inline bool memory_stats_enabled() const { return params_.enable_memory_stats.value_or(false); }
  inline size_t bytes_per_sample_hint() const { return params_.bytes_per_sample_hint.value_or(0); }

  int next_logical_id_ = 0;
  int next_internal_logical_id_ = -1;

  std::vector<int64_t> seed_;
  int64_t original_seed_ = -1;
  size_t current_seed_ = 0;
  bool requires_gpu_ = false;

  std::unique_ptr<ExecutorBase> executor_;
  graph::OpGraph graph_;
  graph::OpGraph::Builder graph_builder_;
  std::map<string, EdgeMeta> edge_names_;

  struct OpDefinition {
    std::string instance_name;
    OpSpec spec;
    int logical_id;
  };

  std::vector<OpDefinition> op_specs_;
  std::vector<OpDefinition> op_specs_for_serialization_;
  std::set<std::string, std::less<>> instance_names_;

  std::vector<PipelineOutputDesc> output_descs_;

  // Mapping between logical id and index in op_specs_
  std::map<int, std::vector<size_t>> logical_ids_;
  std::map<int, int64_t> logical_id_to_seed_;

  // input operators are sorted by names
  std::map<std::string, const graph::OpNode*, std::less<>> input_operators_;

  /**
   * @brief Handles repeating recent inputs for ExternalSource nodes with repeat_last flag on
   *
   * ExternalSource nodes can specify a repeat_last flag which works by re-submitting the most
   * recently fed input in case where no new data was fed between calls to Pipeline::Run.
   *
   * This class maintains a list of such nodes, stores the most recently fed input and re-submits
   * it if no new data was fed.
   */
  class RepeatLastInputs {
   public:
    void FindNodes(const graph::OpGraph &graph, ExecutorBase &exec);

    template <typename OperatorBackend, typename DataBackend>
    bool SetLast(std::string_view name, const TensorList<DataBackend> &data,
                 const std::optional<std::string> &data_id,
                 AccessOrder order,
                 InputOperatorSettingMode ext_src_setting_mode) {
      auto &nodes = GetNodes<OperatorBackend>();
      auto it = nodes.find(name);
      if (it == nodes.end())
        return false;

      auto &node = it->second;
      auto &inp = dynamic_cast<InputOperator<OperatorBackend>&>(*node.op);

      auto do_copy = [&]() {
        node.last_input.Reset();
        if constexpr (std::is_same_v<OperatorBackend, GPUBackend>) {
          if (!order.is_device())
            order = set_last_stream_.get();
        }
        node.last_input.set_order(order);
        node.last_input.Copy(data, order, ext_src_setting_mode.use_copy_kernel);
        if (ext_src_setting_mode.sync)
          AccessOrder::host().wait(order);
      };

      if constexpr (std::is_same_v<OperatorBackend, DataBackend>) {
        if (inp.WouldCopy(ext_src_setting_mode.copy_mode)) {
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

    /**
     * @brief Feeds the recently set inputs to the inputs that have `repeat_last` property
     *
     * @param owner       The pipeline
     * @param fill_queue  If true, the inputs are fed `InputFeedCount(name)` times;
     *                    otherwise they're fed once.
     */
    void Refeed(Pipeline &owner, bool fill_queue = false);

   private:
    template <typename Backend>
    void Refeed(Pipeline &owner, bool fill_queue);

    template <typename Backend>
    struct RepeatLastInput {
      const graph::OpNode *op_node = nullptr;
      Operator<Backend> *op = nullptr;

      using InputBackend = std::conditional_t<std::is_same_v<Backend, MixedBackend>,
                                             CPUBackend, Backend>;
      TensorList<InputBackend> last_input;
      std::optional<std::string> data_id;
    };

    bool empty() const {
      return cpu_nodes_.empty() && gpu_nodes_.empty() && mixed_nodes_.empty();
    }

    std::map<std::string_view, RepeatLastInput<CPUBackend>, std::less<>> cpu_nodes_;
    std::map<std::string_view, RepeatLastInput<GPUBackend>, std::less<>> gpu_nodes_;
    std::map<std::string_view, RepeatLastInput<MixedBackend>, std::less<>> mixed_nodes_;

    template <typename Backend>
    std::map<std::string_view, RepeatLastInput<Backend>, std::less<>> &GetNodes() {
      if constexpr (std::is_same_v<Backend, CPUBackend>)
        return cpu_nodes_;
      else if constexpr (std::is_same_v<Backend, GPUBackend>)
        return gpu_nodes_;
      else
        return mixed_nodes_;
    }
    CUDAStreamLease set_last_stream_;
  };

  RepeatLastInputs repeat_last_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_PIPELINE_H_
