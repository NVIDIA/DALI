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

#include "dali/pipeline/pipeline.h"

#include <google/protobuf/message.h>
#include <google/protobuf/io/coded_stream.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>

#include "dali/core/device_guard.h"
#include "dali/core/mm/default_resources.h"
#include "dali/pipeline/dali.pb.h"
#include "dali/pipeline/executor/executor_factory.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/error_reporting.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/graph/graph2dot.h"
#include "dali/pipeline/graph/cse.h"
#include "dali/pipeline/operator/builtin/input_operator.h"

namespace dali {

namespace {

bool DeserializePipeline(const std::string &serialized_pipeline,
                         dali_proto::PipelineDef &deserialized_pipeline) {
  //  Reading Protobuf file has a limitation of 64 MB
  //  Following instructions will increase the
  //  limits to support the given pipeline
  google::protobuf::io::CodedInputStream coded_input(
      reinterpret_cast<const uint8_t *>(serialized_pipeline.c_str()), serialized_pipeline.size());
  coded_input.SetTotalBytesLimit(serialized_pipeline.size());
  return deserialized_pipeline.ParseFromCodedStream(&coded_input);
}


void DeserializeOpSpec(const dali_proto::OpDef &def, OpSpec *spec) {
  std::string name = def.name();

  // Due to the fact that External Source existed in DALI as two entities we need to have a place
  // where we merge it back into one. "ExternalSource" special handling for serialization was
  // removed so we can merge back _ExternalSource into it.
  // We need to rename the spec that we construct at some point to not serialize it back
  // with the doubled operator.
  if (name == "_ExternalSource") {
    name = "ExternalSource";
  }

  spec->SetSchema(name);

  // Extract all the arguments with correct types
  for (auto &arg : def.args()) {
    auto name = arg.name();
    const DaliProtoPriv arg_wrap(&arg);

    spec->AddInitializedArg(name, DeserializeProtobuf(arg_wrap));
  }

  for (int i = 0; i < def.input_size(); ++i) {
    if (!def.input(i).is_argument_input()) {
      spec->AddInput(def.input(i).name(), ParseStorageDevice(def.input(i).device()));
    }
  }

  for (int i = 0; i < def.input_size(); ++i) {
    if (def.input(i).is_argument_input()) {
      spec->AddArgumentInput(def.input(i).arg_name(), def.input(i).name());
    }
  }

  for (int i = 0; i < def.output_size(); ++i) {
    spec->AddOutput(def.output(i).name(), ParseStorageDevice(def.output(i).device()));
  }
}

void InitializeMemoryResources() {
  (void)mm::GetDefaultResource<mm::memory_kind::host>();
}

bool IsGraphOptimizationEnabled() {
  static const bool enabled = []() {
    if (const char *env = getenv("DALI_OPTIMIZE_GRAPH"))
      return atoi(env) != 0;
    else  // enabled by default
      return true;
  }();
  return enabled;
}

bool IsCSEEnabled() {
  static const bool enabled = []() {
    if (!IsGraphOptimizationEnabled())
      return false;
    if (const char *env = getenv("DALI_ENABLE_CSE"))
      return atoi(env) != 0;
    else  // enabled by default
      return true;
  }();
  return enabled;
}

}  // namespace

Pipeline::Pipeline(int max_batch_size, int num_threads, int device_id, int64_t seed,
                   bool pipelined_execution, int prefetch_queue_depth,
                   bool async_execution, bool dynamic_execution, size_t bytes_per_sample_hint,
                   bool set_affinity)
: Pipeline(PipelineParams{
    max_batch_size != -1 ? std::optional<int>(max_batch_size) : std::nullopt,
    num_threads != -1 ? std::optional<int>(num_threads) : std::nullopt,
    device_id >= 0 ? std::optional<int>(device_id) : std::nullopt,
    seed != -1 ? std::optional<int64_t>(seed) : std::nullopt,
    MakeExecutorType(pipelined_execution, async_execution, false, dynamic_execution),
    set_affinity ? ExecutorFlags::SetAffinity : ExecutorFlags::None,
    QueueSizes(prefetch_queue_depth),
    std::nullopt,
    std::nullopt,
    bytes_per_sample_hint}) {}

Pipeline::Pipeline(const string &serialized_pipeline,
                   int batch_size, int num_threads, int device_id,
                   bool pipelined_execution, int prefetch_queue_depth,
                   bool async_execution, bool dynamic_execution,
                   size_t bytes_per_sample_hint, bool set_affinity,
                   int64_t seed)
: Pipeline(serialized_pipeline, PipelineParams{
    batch_size != -1 ? std::optional<int>(batch_size) : std::nullopt,
    num_threads != -1 ? std::optional<int>(num_threads) : std::nullopt,
    device_id >= 0 ? std::optional<int>(device_id) : std::nullopt,
    seed != -1 ? std::optional<int64_t>(seed) : std::nullopt,
    MakeExecutorType(pipelined_execution, async_execution, false, dynamic_execution),
    set_affinity ? ExecutorFlags::SetAffinity : ExecutorFlags::None,
    QueueSizes(prefetch_queue_depth),
    std::nullopt,
    std::nullopt,
    bytes_per_sample_hint})
{}

Pipeline::Pipeline(const PipelineParams &params) {
  InitializeMemoryResources();
  Init(params);
}

Pipeline::Pipeline(const string &serialized_pipe, const PipelineParams &params) {
  InitializeMemoryResources();
  dali_proto::PipelineDef def;
  DALI_ENFORCE(DeserializePipeline(serialized_pipe, def), "Error parsing serialized pipeline.");

    auto to_optional = [](auto value) {
      return value >= 0 ? std::optional<decltype(value)>(value) : std::nullopt;
    };

    // First, set parameters from the serialized pipeline
    params_.max_batch_size = to_optional(def.batch_size());

    if (def.has_num_threads())
      params_.num_threads = to_optional(def.num_threads());

    if (def.has_device_id())
      params_.device_id = to_optional(def.device_id());

    if (def.has_seed())
      params_.seed = to_optional(def.seed());

    if (def.has_executor_type())
      params_.executor_type = static_cast<ExecutorType>(def.executor_type());

    if (def.has_executor_flags())
      params_.executor_flags = static_cast<ExecutorFlags>(def.executor_flags());

    if (def.has_prefetch_queue_depth_cpu() && def.has_prefetch_queue_depth_gpu())
      params_.prefetch_queue_depths = QueueSizes(def.prefetch_queue_depth_cpu(),
                                                 def.prefetch_queue_depth_gpu());
    else if (def.has_prefetch_queue_depth_cpu())
      params_.prefetch_queue_depths = QueueSizes(def.prefetch_queue_depth_cpu(),
                                                 def.prefetch_queue_depth_cpu());
    else if (def.has_prefetch_queue_depth_gpu())
      params_.prefetch_queue_depths = QueueSizes(def.prefetch_queue_depth_gpu(),
                                                 def.prefetch_queue_depth_gpu());

    if (def.has_enable_checkpointing())
      params_.enable_checkpointing = def.enable_checkpointing();

    // Then, update with any provided parameters
    params_.Update(params);

    Init(params_);

    // from serialized pipeline, construct new pipeline
    // All external inputs
    for (auto& ex : def.external_inputs()) {
      this->AddExternalInput(ex);
    }
    // all operators
    for (auto& op_def : def.op()) {
      OpSpec spec;
      dali::DeserializeOpSpec(op_def, &spec);

      this->AddOperator(spec, op_def.inst_name(),
                        op_def.logical_id() == -1 ? GetNextLogicalId() : op_def.logical_id());
    }
    // output names
    for (auto &output : def.pipe_outputs()) {
      this->output_descs_.emplace_back(output.name(), output.device(),
                                       static_cast<DALIDataType>(output.dtype()),
                                       output.ndim(),
                                       output.layout());
    }

    // checkpointing
    if (checkpointing_enabled()) {
      this->EnableCheckpointing();
    }
}

Pipeline::~Pipeline() {
  Shutdown();
  graph_ = {};
  executor_ = {};
  repeat_last_ = {};
}

PipelineParams Pipeline::DefaultParams() {
  PipelineParams params{};
  params.executor_type = ExecutorType::AsyncPipelined;
  params.executor_flags = ExecutorFlags::ConcurrencyBackend;
  params.prefetch_queue_depths = QueueSizes{2};
  params.enable_checkpointing = false;
  params.enable_memory_stats = false;
  params.bytes_per_sample_hint = 0;
  return params;
}

void Pipeline::Validate(const PipelineParams &params) {
  if (params.device_id.has_value()) {
    int ndev = 0;
    CUDA_CALL(cudaGetDeviceCount(&ndev));
    if (*params.device_id < 0 || *params.device_id >= ndev) {
      if (ndev == 0) {
        throw std::runtime_error("The pipeline requires a CUDA device, but none are available.");
      } else {
        throw std::invalid_argument(make_string("Invalid device id: ", *params.device_id, ". "
                                                "Valid device ids are 0..", ndev - 1 , "."));
      }
    }
  }

  if (!params.max_batch_size.has_value())
    throw std::invalid_argument("The batch size must be set.");
  if (*params.max_batch_size <= 0)
    throw std::invalid_argument(make_string("Invalid batch size: ", *params.max_batch_size, ". "
                                            "Please set a positive integer."));

  if (!params.num_threads.has_value())
    throw std::invalid_argument("The number of threads must be set.");
  if (*params.num_threads <= 0)
    throw std::invalid_argument(make_string("Invalid number of threads: ", *params.num_threads, ". "
                                            "Please set a positive integer."));

  if (!params.prefetch_queue_depths.has_value())
    throw std::logic_error("Internal error: prefetch queue depths not set.");

  if (params.prefetch_queue_depths->cpu_size <= 0 ||
      params.prefetch_queue_depths->gpu_size <= 0) {
    throw std::invalid_argument("The prefetch queue depth must be greater than 0.");
  }
  if (params.prefetch_queue_depths->cpu_size != params.prefetch_queue_depths->gpu_size) {
    if (params.executor_type.has_value() &&
        !Test(*params.executor_type, ExecutorType::SeparatedFlag)) {
      throw std::invalid_argument("The prefetch queue depth must be the same for CPU and GPU "
                                  "if separated execution is not used.");
    }
  }
}

void Pipeline::Init(const PipelineParams &params) {
    DALI_ENFORCE(!params.device_id.has_value() || cuInitChecked(),
                "You are trying to create a GPU DALI pipeline, while CUDA is not available. "
                "Please install CUDA or set `device_id = None` in Pipeline constructor. "
                "If running inside Docker container, you may need to use  `--gpus` option.");

    int device_id = params.device_id.value_or(CPU_ONLY_DEVICE_ID);
    DeviceGuard g(device_id);
    params_.Update(params);
    Validate(params_);
    /*this->max_batch_size_ = max_batch_size;
    this->num_threads_ = num_threads;
    this->device_id_ = device_id;
    using Clock = std::chrono::high_resolution_clock;
    this->original_seed_ = seed < 0 ? Clock::now().time_since_epoch().count() : seed;
    this->pipelined_execution_ = pipelined_execution;
    this->async_execution_ = async_execution;
    this->dynamic_execution_ = dynamic_execution;
    this->bytes_per_sample_hint_ = bytes_per_sample_hint;
    this->set_affinity_ = set_affinity;
    this->prefetch_queue_depth_ = prefetch_queue_depth;
    this->separated_execution_ = (prefetch_queue_depth.cpu_size != prefetch_queue_depth.gpu_size);
    DALI_ENFORCE(max_batch_size_ > 0, "Max batch size must be greater than 0");*/

    using Clock = std::chrono::high_resolution_clock;
    original_seed_ = params.seed.value_or(Clock::now().time_since_epoch().count());

    seed_.resize(MAX_SEEDS);
    current_seed_ = 0;
    std::seed_seq ss{this->original_seed_};
    ss.generate(seed_.begin(), seed_.end());
  }

static bool has_prefix(const std::string &operator_name, const std::string& prefix) {
  if (operator_name.size() < prefix.size()) return false;
  return std::equal(operator_name.begin(), operator_name.begin() + prefix.size(),
                    prefix.begin());
}

int Pipeline::AddOperator(const OpSpec &spec, std::string_view inst_name) {
  return AddOperator(spec, inst_name, GetNextLogicalId());
}

int Pipeline::AddOperator(const OpSpec &spec, int logical_id) {
  std::string name = make_string("__", spec.SchemaName(), "_", logical_id);
  if (instance_names_.count(name)) {
    int group_size = 0;
    auto it = logical_ids_.find(logical_id);
    if (it != logical_ids_.end())
      group_size = it->second.size();
    for (int aux_id = group_size; instance_names_.count(name) > 0; aux_id++)
      name = make_string("__", spec.SchemaName(), "_", logical_id, "_", aux_id);
  }
  return AddOperator(spec, name, logical_id);
}

int Pipeline::AddOperator(const OpSpec &spec) {
  return AddOperator(spec, GetNextLogicalId());
}

int Pipeline::AddOperator(const OpSpec &const_spec, std::string_view inst_name, int logical_id) {
  DALI_ENFORCE(!built_,
               "Alterations to the pipeline after \"Build()\" has been called are not allowed");

  // Validate op device
  auto device = const_spec.GetArgument<std::string>("device");
  // Doesn't make sense to state the wrong device in the error message, so we use the operator name
  // directly instead of the error context message..
  DALI_ENFORCE(device == "cpu" || device == "gpu" || device == "mixed" || device == "support",
               make_string("Invalid device argument \"", device, "\" for operator `",
                           GetOpDisplayName(const_spec, true),
                           "`. Valid options are \"cpu\", \"gpu\" or \"mixed\""));

  int result = -1;
  try {
    result = AddOperatorImpl(const_spec, inst_name, logical_id);
  } catch (...) {
    PropagateError({std::current_exception(), GetErrorContextMessage(const_spec), ""});
  }
  return result;
}

int Pipeline::AddOperatorImpl(const OpSpec &const_spec,
                              std::string_view inst_name,
                              int logical_id) {
  assert(!built_ && "Already checked by AddOperator()");

  DALI_ENFORCE(0 <= logical_id,
               "Logical id of the node must be positive, got " + std::to_string(logical_id) + ".");

  DALI_ENFORCE(instance_names_.insert(std::string(inst_name)).second,
               make_string("Duplicate operator instance name: \"", inst_name, "\"."));

  if (logical_id > next_logical_id_) {
    next_logical_id_ = logical_id + 1;
  }

  // we modify spec so copy it
  OpSpec spec = const_spec;
  auto operator_name = GetOpDisplayName(spec, true);

  auto device = const_spec.GetArgument<std::string>("device");
  if (spec.GetSchema().IsNoPrune())
    spec.SetArg("preserve", true);

  if (spec.SchemaName() == "ExternalSource")
    spec.SetArg("preserve_name", true);  // ExternalSource must not be collapsed in CSE

  // Take a copy of the passed OpSpec for serialization purposes before any modification
  this->op_specs_for_serialization_.push_back({std::string(inst_name), spec, logical_id});

  DALI_ENFORCE(device != "gpu" || device_id() != CPU_ONLY_DEVICE_ID,
               "Cannot add a GPU operator. Pipeline 'device_id' should not be equal to "
               "`CPU_ONLY_DEVICE_ID`.");

  if (device == "support") {
    auto warning_context = GetErrorContextMessage(spec, "Warning");
    DALI_WARN(warning_context,
              " \"support\" device is deprecated; use \"cpu\" or leave blank instead.");
    device = "cpu";
    spec.SetArg("device", "cpu");
  }

  DeviceGuard g(device_id());

  // Verify the regular inputs to the op
  for (int i = 0; i < spec.NumInput(); ++i) {
    if (spec.IsArgumentInput(i)) {
      // We only look at regular inputs here
      continue;
    }
    string input_name = spec.InputName(i);
    auto input_device = spec.InputDevice(i);
    auto it = edge_names_.find(input_name);

    DALI_ENFORCE(it != edge_names_.end(),
                 make_string("Data node \"", input_name, "\" requested as ", FormatInput(spec, i),
                             " to operator \"", inst_name, "\" is not known to the pipeline."));

    DALI_ENFORCE(device_id() != CPU_ONLY_DEVICE_ID || device == "cpu",
                  "Cannot add a Mixed operator with a GPU output, 'device_id' "
                  "should not be `CPU_ONLY_DEVICE_ID`.");

    if (input_device == StorageDevice::GPU) {
      ToGPU(it);
    } else {
      ToCPU(it);
    }
  }

  // Verify the argument inputs to the op
  for (const auto &[arg_name, input_idx] : spec.ArgumentInputs()) {
    std::string input_name = spec.InputName(input_idx);
    auto it = edge_names_.find(input_name);

    // TODO(michalz): This is a bit ugly, but it provides a nice and generic error message.
    //                In the long run, we should probably allow GPU named inputs and then
    //                the check should be done based on the schema, without a universal ToCPU.
    if (dynamic_execution())
      ToCPU(it);

    DALI_ENFORCE(
        it != edge_names_.end(),
        make_string("Data node \"", input_name, "\" requested as ", FormatArgument(spec, arg_name),
                    " to operator \"", inst_name, "\" is not known to the pipeline."));

    if (!it->second.has_cpu) {
      assert(it->second.has_gpu);
      DALI_FAIL(make_string("Error while specifying ", FormatArgument(spec, arg_name),
                            ". Named argument inputs to operators must be CPU data nodes. "
                            "However, a GPU data node was provided."));
    }
  }

  // Verify and record the outputs of the op
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string output_name = spec.OutputName(i);
    auto output_device = spec.OutputDevice(i);

    auto it = edge_names_.find(output_name);
    DALI_ENFORCE(
        it == edge_names_.end(),
        make_string("Error while specifying ", FormatOutput(spec, i), ". Output name \"",
                    output_name, "\" conflicts with an existing intermediate result name."));

    if (device == "cpu") {
      DALI_ENFORCE(output_device == StorageDevice::CPU,
                   make_string("Error while specifying ", FormatOutput(spec, i),
                               ". Only CPU operators can produce CPU outputs."));
    }
    // The edge describes that the named output of this operator produces the CPU or GPU data,
    // the former for "cpu" ops, the latter for "mixed" and "gpu".
    EdgeMeta meta = NewEdge(output_device);

    DALI_ENFORCE(edge_names_.insert({output_name, meta}).second,
                 make_string("Error while specifying ", FormatOutput(spec, i), "node name: \"",
                             output_name, "\". Output name insertion failure."));
  }

  // store updated spec
  AddToOpSpecs(inst_name, spec, logical_id);
  return logical_id;
}

bool Pipeline::IsLogicalIdUsed(int logical_id) const {
  return logical_ids_.find(logical_id) != logical_ids_.end();
}

void Pipeline::AddToOpSpecs(std::string_view inst_name, const OpSpec &spec, int logical_id) {
  auto& logical_group = logical_ids_[logical_id];
  if (logical_group.size() > 0) {
    const auto &group_name = op_specs_[logical_group.front()].spec.SchemaName();
    DALI_ENFORCE(
        group_name == spec.SchemaName(),
        "Different Operator types cannot be grouped with the same logical id. Tried to group `" +
            GetOpDisplayName(spec, true) + "` using logical_id=" + std::to_string(logical_id) +
            " which is already assigned to " + group_name + ".");
    const OpSchema &schema = SchemaRegistry::GetSchema(spec.SchemaName());
  }
  op_specs_.push_back({std::string(inst_name), spec, logical_id});
  logical_ids_[logical_id].push_back(op_specs_.size() - 1);
}

inline int GetMemoryHint(const OpSpec &spec, int output_index) {
  if (!spec.HasArgument("bytes_per_sample_hint"))
    return 0;
  std::vector<int> hints;
  GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", spec.NumOutput());

  DALI_ENFORCE(output_index < static_cast<int>(hints.size()),
               "Output index out of range: " + std::to_string(output_index));
  return hints[output_index];
}

inline void SetMemoryHint(OpSpec &spec, int output_index, int value) {
  std::vector<int> hints;
  int no = spec.NumOutput();

  DALI_ENFORCE(output_index < no, "Output index out of range: " +
    std::to_string(output_index) + " >= " + std::to_string(no));

  GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", no);
  hints[output_index] = value;
  spec.SetArg("bytes_per_sample_hint", hints);
}

void Pipeline::PropagateMemoryHint(graph::OpNode &node) {
  for (int inp_idx = 0; inp_idx < node.spec.NumRegularInput(); inp_idx++) {
    const auto &input_name = node.spec.Input(inp_idx);
    const auto &data_node = graph_.GetData(input_name);
    if (auto *src = data_node->producer.op) {
      int hint = GetMemoryHint(src->spec, data_node->producer.idx);
      if (hint) {
        // inp_idx == out_idx for MakeContiguous
        SetMemoryHint(node.spec, inp_idx, hint);
      }
    }
  }
}

void Pipeline::Build() {
  Build(this->output_descs_);
}

void Pipeline::Build(const vector<std::pair<string, string>>& output_names) {
  std::vector<PipelineOutputDesc> output_descs = {output_names.begin(), output_names.end()};
  this->Build(output_descs);
}

void Pipeline::Build(std::vector<PipelineOutputDesc> output_descs) {
  DeviceGuard d(device_id());
  SetOutputDescs(std::move(output_descs));
  DALI_ENFORCE(!built_, "\"Build()\" can only be called once.");
  auto num_outputs = output_descs_.size();
  DALI_ENFORCE(num_outputs > 0,
               make_string("User specified incorrect number of outputs (", num_outputs, ")."));

  executor_ =
      GetExecutor(*params_.executor_type, *params_.executor_flags,
                  max_batch_size(), num_threads(), device_id(), bytes_per_sample_hint(),
                  GetQueueSizes());
  executor_->EnableMemoryStats(memory_stats_enabled());
  executor_->EnableCheckpointing(checkpointing_enabled());
  executor_->Init();

  // Validate the output tensors names
  vector<string> outputs;
  for (const auto &out_desc : output_descs_) {
    string name = out_desc.name;
    auto device = out_desc.device;
    auto it = edge_names_.find(name);
    DALI_ENFORCE(it != edge_names_.end(), "Requested output name '" +
        name + "' is not known to the pipeline.");

    if (device == StorageDevice::CPU) {
      if (!it->second.has_cpu)
        ToCPU(it);

      if (!it->second.has_contiguous_cpu) {
        // Add a make contiguous op to produce this output - we need pipeline outputs to be dense.
        auto output_name = AddMakeContiguousNode(
            it->second,
            name,
            StorageDevice::CPU,
            "cpu",
            StorageDevice::CPU);
        outputs.push_back(output_name);
      } else {
        outputs.push_back(it->first + "_cpu");
      }

    } else {
      assert(device == StorageDevice::GPU);
      DALI_ENFORCE(device_id() != CPU_ONLY_DEVICE_ID,
                   make_string(
                     "Cannot move the data node ", name, " to the GPU "
                     "in a CPU-only pipeline. The 'device_id' parameter "
                     "is set to `CPU_ONLY_DEVICE_ID`. Set 'device_id' "
                     "to a valid GPU identifier to enable GPU features "
                     "in the pipeline."));
      if (!it->second.has_gpu)
        ToGPU(it);

      if (!it->second.has_contiguous_gpu) {
        auto output_name = AddMakeContiguousNode(
            it->second,
            name,
            StorageDevice::GPU,
            "gpu",
            StorageDevice::GPU);
        outputs.push_back(output_name);
      } else {
        outputs.push_back(it->first + "_gpu");
      }
    }
  }

  // Creating the graph

  for (auto &name_op_spec : op_specs_) {
    const string &inst_name = name_op_spec.instance_name;
    OpSpec op_spec = name_op_spec.spec;
    try {
      PrepareOpSpec(&op_spec, name_op_spec.logical_id);
      graph_builder_.Add(inst_name, op_spec);
    } catch (...) {
      PropagateError({std::current_exception(),
                      "Critical error when building pipeline:\n" + GetErrorContextMessage(op_spec),
                      "\nCurrent pipeline object is no longer valid."});
    }
  }

  for (auto &out : outputs)
    graph_builder_.AddOutput(out);

  graph_ = std::move(graph_builder_).GetGraph(true);

  for (auto &node : graph_.OpNodes()) {
    if (node.spec.SchemaName() == "MakeContiguous") {
      PropagateMemoryHint(node);
    }
  }

  // Graph optimization goes here
  if (IsCSEEnabled())
    graph::EliminateCommonSubgraphs(graph_);

  // Load the final graph into the executor
  executor_->Build(graph_);
  // CAUTION: Do not insert anything that can throw between `executor_->Build()` and
  //          `DiscoverInputOperators`.
  DiscoverInputOperators();

  repeat_last_.FindNodes(graph_, *executor_);

  built_ = true;
}


void Pipeline::SetOutputDescs(const vector<std::pair<string, string>> &output_names) {
  DALI_ENFORCE(
      output_descs_.empty(),
      "Resetting output_descs_ providing only name and device is forbidden. If you want to reset "
      "the output_descs_, please use the `SetOutputDescs(vector<PipelineOutputDesc>)`.");
  output_descs_ = {output_names.begin(), output_names.end()};
}

void Pipeline::SetOutputDescs(std::vector<PipelineOutputDesc> output_descs) {
  DALI_ENFORCE(output_descs_.empty() || output_descs_ == output_descs,
               make_string("Once set, changing values of `output_descs_` is forbidden. "
                           "`SetOutputDescs` may be invoked multiple times, with the argument "
                           "equal to existing `output_descs_`.\nReceived:\n", output_descs));
  output_descs_ = std::move(output_descs);
}

void Pipeline::Run() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  repeat_last_.Refeed(*this);
  executor_->Run();
}

void Pipeline::Prefetch() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  repeat_last_.Refeed(*this, true);
  executor_->Prefetch();
}

bool Pipeline::ValidateOutputs(const Workspace &ws) const {
  DALI_ENFORCE(ws.NumOutput() == static_cast<int>(output_descs_.size()),
               make_string("Number of outputs does not match. Expected: ", output_descs_.size(),
                           ". Received: ", ws.NumOutput(), "."));
  for (int i = 0; i < ws.NumOutput(); i++) {
    DALI_ENFORCE(output_descs_[i].ndim == -1 || ws.GetOutputDim(i) == output_descs_[i].ndim,
                 make_string("Number of dimensions in the output_idx=", i,
                             " does not match. Expected: ", output_descs_[i].ndim,
                             ". Received: ", ws.GetOutputDim(i), "."));
    DALI_ENFORCE(
        output_descs_[i].dtype == DALI_NO_TYPE || ws.GetOutputDataType(i) == output_descs_[i].dtype,
        make_string("Data type in the output_idx=", i, " does not match. Expected: ",
                    output_descs_[i].dtype, ". Received: ", ws.GetOutputDataType(i), "."));
    DALI_ENFORCE(
        output_descs_[i].layout.empty() || ws.GetOutputLayout(i) == output_descs_[i].layout,
        make_string("Layout in the output_idx=", i, " does not match. Expected: ",
                    output_descs_[i].layout, ". Received: ", ws.GetOutputLayout(i), "."));
  }
  return true;
}

void Pipeline::Outputs(Workspace *ws) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to executing the pipeline.");
  try {
    executor_->Outputs(ws);
  } catch (...) {
    ProcessException(std::current_exception());
  }

  ValidateOutputs(*ws);
}

void Pipeline::ShareOutputs(Workspace *ws) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to executing the pipeline.");
  try {
    executor_->ShareOutputs(ws);
  } catch (...) {
    ProcessException(std::current_exception());
  }

  ValidateOutputs(*ws);
}

void Pipeline::ReleaseOutputs() {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to executing the pipeline.");
  try {
    executor_->ReleaseOutputs();
  } catch (...) {
    ProcessException(std::current_exception());
  }
}

void Pipeline::ToCPU(std::map<string, EdgeMeta>::iterator it) {
  // Insert a D2H copy, if needed
  if (it->second.has_cpu)
    return;
  OpSpec copy_to_host_spec =
    OpSpec("Copy")
    .AddArg("device", "cpu")
    .AddInput(it->first, StorageDevice::GPU)
    .AddOutput(it->first, StorageDevice::CPU);
  // don't put it into op_specs_for_serialization_, only op_specs_
  AddToOpSpecs("__Copy_GpuToCpu_" + it->first, copy_to_host_spec, GetNextInternalLogicalId());
  it->second.has_cpu = true;
  it->second.has_contiguous_cpu = true;  // the result is always contiguous
}

void Pipeline::ToGPU(std::map<string, EdgeMeta>::iterator it) {
  // Insert a H2D copy, if needed
  if (it->second.has_gpu)
    return;
  OpSpec copy_to_dev_spec =
    OpSpec("MakeContiguous")
    .AddArg("device", "mixed")
    .AddInput(it->first, StorageDevice::CPU)
    .AddOutput(it->first, StorageDevice::GPU);
  // don't put it into op_specs_for_serialization_, only op_specs_
  AddToOpSpecs("__Copy_CpuToGpu_" + it->first, copy_to_dev_spec, GetNextInternalLogicalId());
  it->second.has_gpu = true;
  it->second.has_contiguous_gpu = true;  // the result is always contiguous
}

void Pipeline::PrepareOpSpec(OpSpec *spec, int logical_id) {
  spec->AddArg("max_batch_size", max_batch_size())
    .AddArg("num_threads", num_threads())
    .AddArg("device_id", device_id())
    .AddArg("checkpointing", checkpointing_enabled());
  string dev = spec->GetArgument<string>("device");
  if (dev == "cpu" || dev == "mixed")
    spec->AddArg("cpu_prefetch_queue_depth", GetQueueSizes().cpu_size);
  if (dev == "gpu" || dev == "mixed")
    spec->AddArg("gpu_prefetch_queue_depth", GetQueueSizes().gpu_size);

  if (spec->GetSchemaOrDefault().HasRandomSeedArg()) {
    if (spec->ArgumentDefined("seed")) {
      logical_id_to_seed_[logical_id] = spec->GetArgument<int64_t>("seed");
    } else {
      if (logical_id_to_seed_.find(logical_id) == logical_id_to_seed_.end())
        logical_id_to_seed_[logical_id] = seed_[current_seed_];
      spec->AddArg("seed", logical_id_to_seed_[logical_id]);
      current_seed_ = (current_seed_+1) % MAX_SEEDS;
    }
  }
}

/**
 * @brief Helper method that serialized OpSpec
 * decouples spec class from dali_proto
 */
void SerializeToProtobuf(dali_proto::OpDef *op, const string &inst_name, const OpSpec &spec,
                         int logical_id) {
  op->set_name(spec.SchemaName());
  op->set_inst_name(inst_name);
  op->set_logical_id(logical_id);

  for (int i = 0; i < spec.NumInput(); ++i) {
    dali_proto::InputOutput *in = op->add_input();
    in->set_name(spec.InputName(i));
    in->set_device(to_string(spec.InputDevice(i)));
    if (spec.IsArgumentInput(i)) {
        in->set_arg_name(spec.ArgumentInputName(i));
    }
    in->set_is_argument_input(spec.IsArgumentInput(i));
  }

  for (int i = 0; i < spec.NumOutput(); ++i) {
    dali_proto::InputOutput *out = op->add_output();
    out->set_name(spec.OutputName(i));
    out->set_device(to_string(spec.OutputDevice(i)));
    out->set_is_argument_input(false);
  }

  for (auto& a : spec.Arguments()) {
    // filter out args that need to be dealt with on
    // loading a serialized pipeline
    auto name = a->get_name();
    if (name == "max_batch_size" ||
        name == "num_threads" ||
        name == "bytes_per_sample_hint") {
      continue;
    }

    dali_proto::Argument *arg = op->add_args();
    DaliProtoPriv arg_wrap(arg);

    a->SerializeToProtobuf(&arg_wrap);
  }
}

string Pipeline::SerializeToProtobuf() const {
  dali_proto::PipelineDef pipe;
  pipe.set_num_threads(this->num_threads());
  pipe.set_batch_size(this->max_batch_size());
  pipe.set_device_id(this->device_id());
  pipe.set_seed(this->original_seed_);
  pipe.set_enable_checkpointing(this->checkpointing_enabled());
  pipe.set_bytes_per_sample_hint(this->bytes_per_sample_hint());
  pipe.set_prefetch_queue_depth_cpu(this->GetQueueSizes().cpu_size);
  pipe.set_prefetch_queue_depth_gpu(this->GetQueueSizes().gpu_size);
  pipe.set_executor_type(static_cast<int32_t>(*params_.executor_type));
  pipe.set_executor_flags(static_cast<int32_t>(*params_.executor_flags));

  // loop over ops, create messages and append
  for (size_t i = 0; i < this->op_specs_for_serialization_.size(); ++i) {
    dali_proto::OpDef *op_def = pipe.add_op();

    const auto& p = this->op_specs_for_serialization_[i];
    const OpSpec& spec = p.spec;

    DALI_ENFORCE(spec.GetSchema().IsSerializable(), "Could not serialize the operator: `"
                                                    + GetOpDisplayName(spec, true) + "`.");

    dali::SerializeToProtobuf(op_def, p.instance_name, spec, p.logical_id);
  }

  // loop over outputs used to create the graph
  for (auto& output : output_descs_) {
    dali_proto::InputOutput *out = pipe.add_pipe_outputs();

    out->set_name(output.name);
    out->set_device(to_string(output.device));
    out->set_is_argument_input(false);
    out->set_dtype(output.dtype);
    out->set_ndim(output.ndim);
  }
  pipe.set_device_id(this->device_id());
  string output = pipe.SerializeAsString();

  return output;
}

graph::OpNode *Pipeline::GetOperatorNode(std::string_view name) {
  return graph_.GetOp(name);
}

OperatorBase *Pipeline::GetOperator(std::string_view name) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"GetOperator()\".");
  return executor_->GetOperator(name);
}

const graph::OpNode *Pipeline::GetInputOperatorNode(std::string_view name) {
  auto it = input_operators_.find(name);
  if (it != input_operators_.end())
    return it->second;
  else
    return nullptr;
}

std::map<std::string_view, ReaderMeta, std::less<>> Pipeline::GetReaderMeta() {
  std::map<std::string_view, ReaderMeta, std::less<>> ret;
  for (auto &op_node : graph_.OpNodes()) {
    auto *op = GetOperator(op_node.instance_name);
    if (!op)  // optimized-out or not yet instantiated
      continue;
    ReaderMeta meta = op->GetReaderMeta();
    if (meta) {
      ret.emplace(op_node.instance_name, meta);
    }
  }
  return ret;
}

ReaderMeta Pipeline::GetReaderMeta(std::string_view name) {
  ReaderMeta meta{};
  if (auto *op = executor_->GetOperator(name)) {
    meta = op->GetReaderMeta();
  }
  return meta;
}

int Pipeline::InputFeedCount(std::string_view name) {
  return executor_->InputFeedCount(name);
}

const TensorLayout &Pipeline::GetInputLayout(std::string_view name) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"GetInputLayout()\".");
  auto *op = executor_->GetOperator(name);
  if (const auto *in_op = dynamic_cast<InputOperator<CPUBackend> *>(op))
    return in_op->in_layout();
  if (const auto *in_op = dynamic_cast<InputOperator<MixedBackend> *>(op))
    return in_op->in_layout();
  if (const auto *in_op = dynamic_cast<InputOperator<GPUBackend> *>(op))
    return in_op->in_layout();
  DALI_FAIL(make_string("Could not find an input operator named \"", name, "\"."));
}


int Pipeline::GetInputNdim(std::string_view name) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"GetInputNdim()\".");
  const auto *node = GetOperatorNode(name);
  auto *op = executor_->GetOperator(name);
  if (node->op_type == OpType::CPU) {
    const auto *in_op = dynamic_cast<InputOperator<CPUBackend> *>(op);
    if (in_op) {
      return in_op->in_ndim();
    }
  } else if (node->op_type == OpType::MIXED) {
    const auto *in_op = dynamic_cast<InputOperator<MixedBackend> *>(op);
    if (in_op) {
      return in_op->in_ndim();
    }
  } else if (node->op_type == OpType::GPU) {
    const auto *in_op = dynamic_cast<InputOperator<GPUBackend> *>(op);
    if (in_op) {
      return in_op->in_ndim();
    }
  }
  DALI_FAIL(make_string("Could not find an input operator named \"", name, "\"."));
}


DALIDataType Pipeline::GetInputDtype(std::string_view name) {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"GetInputDtype()\".");
  const auto *node = GetOperatorNode(name);
  auto *op = executor_->GetOperator(name);
  if (node->op_type == OpType::CPU) {
    const auto *in_op = dynamic_cast<InputOperator<CPUBackend> *>(op);
    if (in_op) {
      return in_op->in_dtype();
    }
  } else if (node->op_type == OpType::MIXED) {
    const auto *in_op = dynamic_cast<InputOperator<MixedBackend> *>(op);
    if (in_op) {
      return in_op->in_dtype();
    }
  } else if (node->op_type == OpType::GPU) {
    const auto *in_op = dynamic_cast<InputOperator<GPUBackend> *>(op);
    if (in_op) {
      return in_op->in_dtype();
    }
  }
  DALI_FAIL(make_string("Could not find an input operator named \"", name, "\"."));
}

const std::string &Pipeline::input_name(int n) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"input_name(int)\".");
  DALI_ENFORCE(n >= 0,
               make_string("Id of an input operator must be a non-negative integer. Got: ", n));
  DALI_ENFORCE(static_cast<size_t>(n) < input_operators_.size(),
               make_string("Trying to fetch the name of an input operator with id=", n,
                           " while the id has to be smaller than ", num_inputs(), "."));
  auto it = input_operators_.begin();
  std::advance(it, n);
  return it->first;
}

const std::string &Pipeline::output_name(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_name()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_descs_.size());
  return output_descs_[id].name;
}

StorageDevice Pipeline::output_device(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_device()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_descs_.size());
  return output_descs_[id].device;
}

DALIDataType Pipeline::output_dtype(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_dtype()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_descs_.size());
  return output_descs_[id].dtype;
}

int Pipeline::output_ndim(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_ndim()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_descs_.size());
  return output_descs_[id].ndim;
}


int Pipeline::num_inputs() const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"num_inputs()\".");
  return input_operators_.size();
}


int Pipeline::num_outputs() const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"num_outputs()\".");
  return output_descs_.size();
}

const std::vector<PipelineOutputDesc> &Pipeline::output_descs() const & {
  return output_descs_;
}

void Pipeline::SaveGraphToDotFile(const std::string &filename,
                                  bool show_tensors,
                                  bool use_colors) {
  std::ofstream ofs(filename, std::ios::out);
  if (!ofs)
    DALI_FAIL("Cannot open \"", filename, "\" for writing.");
  GenerateDOTFromGraph(ofs, graph_, show_tensors, use_colors);
}

int Pipeline::GetNextLogicalId() {
  int ret = next_logical_id_;
  next_logical_id_++;
  return ret;
}

int Pipeline::GetNextInternalLogicalId() {
  int ret = next_internal_logical_id_;
  next_internal_logical_id_--;
  return ret;
}


bool Pipeline::IsDeserializable(const std::string &serialized_pipeline) {
  dali_proto::PipelineDef def;
  return DeserializePipeline(serialized_pipeline, def);
}

void Pipeline::Shutdown() {
  DeviceGuard dg(device_id());
  for (auto &[name, node] : input_operators_) {
    OperatorBase *op_ptr = executor_->GetOperator(name);
    if (!op_ptr)
      continue;
    if (auto *cpu_op_ptr = dynamic_cast<InputOperator<CPUBackend> *>(op_ptr)) {
      cpu_op_ptr->BreakWaiting();
    } else if (auto *mixed_op_ptr = dynamic_cast<InputOperator<MixedBackend> *>(op_ptr)) {
      mixed_op_ptr->BreakWaiting();
    } else if (auto *gpu_op_ptr = dynamic_cast<InputOperator<GPUBackend> *>(op_ptr)) {
      gpu_op_ptr->BreakWaiting();
    } else {
      assert(false);  // This shouldn't happen.
    }
  }

  if (executor_)
    executor_->Shutdown();
}

std::tuple<OpSpec, std::string, std::string> Pipeline::PrepareMakeContiguousNode(
    EdgeMeta &meta, std::string_view input_name, StorageDevice input_dev,
    std::string_view device, StorageDevice output_dev) {
  // Prefix for the output name to be generated, so it is distinct after being made contiguous.
  const char *cpu_to_cpu_out = "contiguous_cpu_to_cpu_";
  const char *gpu_to_gpu_out = "contiguous_gpu_to_gpu_";
  // regular "transfer", other operator expect nodes named "<operator_name>_gpu"
  const char *cpu_to_gpu_out = "";

  const char *cpu_to_cpu_name = "__MakeContiguous_CpuToCpu_";
  const char *cpu_to_gpu_name = "__MakeContiguous_CpuToGpu_";
  const char *gpu_to_gpu_name = "__MakeContiguous_GpuToGpu_";

  const char *output_prefix = nullptr;
  const char *op_name_prefix = nullptr;

  if (input_dev == StorageDevice::CPU && output_dev == StorageDevice::CPU) {
    output_prefix = cpu_to_cpu_out;
    op_name_prefix = cpu_to_cpu_name;
  } else if (input_dev == StorageDevice::CPU && output_dev == StorageDevice::GPU) {
    output_prefix = cpu_to_gpu_out;
    op_name_prefix = cpu_to_gpu_name;
  } else {
    output_prefix = gpu_to_gpu_out;
    op_name_prefix = gpu_to_gpu_name;
  }

  std::string output_name = make_string(output_prefix, input_name);
  std::string op_name = make_string(op_name_prefix, input_name);

  OpSpec spec = OpSpec("MakeContiguous")
                    .AddArg("device", device)
                    .AddInput(std::string(input_name), input_dev)
                    .AddOutput(output_name, output_dev);
  return {spec, op_name, output_name};
}


std::string Pipeline::AddMakeContiguousNode(EdgeMeta &meta,
                                            std::string_view input_name,
                                            StorageDevice input_dev,
                                            std::string_view device,
                                            StorageDevice output_dev) {
  auto [spec, op_name, output_name] =
      PrepareMakeContiguousNode(meta, input_name, input_dev, device, output_dev);
  std::string output_name_and_device =  make_string(output_name, "_", output_dev);

  if ((output_dev == StorageDevice::CPU && meta.has_make_contiguous_cpu) ||
      (output_dev == StorageDevice::GPU && meta.has_make_contiguous_gpu)) {
    return output_name_and_device;
  }

  // Add a make contiguous op to produce this output
  auto id = GetNextInternalLogicalId();
  AddToOpSpecs(op_name, std::move(spec), id);

  if (output_dev == StorageDevice::CPU) {
    meta.has_make_contiguous_cpu = true;
  }
  if (output_dev == StorageDevice::GPU) {
    meta.has_make_contiguous_gpu = true;
  }
  return output_name_and_device;
}


void Pipeline::DiscoverInputOperators() noexcept {
  auto& op_nodes = graph_.OpNodes();
  for (const auto &node : op_nodes) {
    auto *op = executor_->GetOperator(node.instance_name);
    if (IsInputOperator(op)) {
      input_operators_.insert(std::make_pair(node.instance_name, &node));
    }
  }
}

void Pipeline::ProcessException(std::exception_ptr eptr) {
  PropagateError(
      {eptr, "Critical error in pipeline:\n", "\nCurrent pipeline object is no longer valid."});
}

void Pipeline::RepeatLastInputs::FindNodes(const graph::OpGraph &graph, ExecutorBase &exec) {
  for (const auto &node : graph.OpNodes()) {
    if (node.spec.SchemaName() != "ExternalSource")
      continue;

    bool repeat_last = false;
    if (!node.spec.TryGetArgument(repeat_last, "repeat_last") || !repeat_last)
      continue;

    auto op = exec.GetOperator(node.instance_name);

    if (node.op_type == OpType::GPU)
      gpu_nodes_[node.instance_name] = { &node, dynamic_cast<Operator<GPUBackend>*>(op) };
    else if (node.op_type == OpType::CPU)
      cpu_nodes_[node.instance_name] = { &node, dynamic_cast<Operator<CPUBackend>*>(op) };
    else if (node.op_type == OpType::MIXED)
      mixed_nodes_[node.instance_name] = { &node, dynamic_cast<Operator<MixedBackend>*>(op) };
    else
      assert(!"Unexpected backend for an ExternalSource node.");
  }
}

template <typename Backend>
void Pipeline::RepeatLastInputs::Refeed(Pipeline &owner, bool fill_queue) {
  auto &nodes = GetNodes<Backend>();
  for (auto &[name, node] : nodes) {
    int count = fill_queue ? owner.InputFeedCount(name) : 1;
    for (int i = 0; i < count; i++)
      owner.SetExternalInputHelper(
          node.op_node->instance_name,
          node.last_input,
          node.data_id,
          node.last_input.order(),
          InputOperatorSettingMode{false, false, InputOperatorCopyMode::FORCE_NO_COPY},
          true);
  }
}

void Pipeline::RepeatLastInputs::Refeed(Pipeline &owner, bool fill_queue) {
  Refeed<CPUBackend>(owner, fill_queue);
  Refeed<MixedBackend>(owner, fill_queue);
  Refeed<GPUBackend>(owner, fill_queue);
}

}  // namespace dali
