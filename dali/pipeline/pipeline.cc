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

#include "dali/pipeline/pipeline.h"

#include <google/protobuf/message.h>
#include <google/protobuf/io/coded_stream.h>

#include <algorithm>
#include <functional>
#include <memory>

#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"
#include "dali/pipeline/executor/executor_factory.h"
#include "dali/pipeline/executor/pipelined_executor.h"

#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/common.h"
#include "dali/core/device_guard.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

  void DeserializeOpSpec(const dali_proto::OpDef& def, OpSpec* spec) {
    std::string name = def.name();

    // Due to the fact that External Source existed in DALI as two entities we need to have a place
    // where we merge it back into one. "ExternalSource" special handling for serialization was
    // removed so we can merge back _ExternalSource into it.
    // We need to rename the spec that we construct at some point to not serialize it back
    // with the doubled operator.
    if (name == "_ExternalSource") {
      name = "ExternalSource";
    }

    spec->set_name(name);

    // Extract all the arguments with correct types
    for (auto &arg : def.args()) {
      auto name = arg.name();
      const DaliProtoPriv arg_wrap(&arg);

      spec->AddInitializedArg(name, DeserializeProtobuf(arg_wrap));
    }

    for (int i = 0; i < def.input_size(); ++i) {
      if (!def.input(i).is_argument_input()) {
        spec->AddInput(def.input(i).name(), def.input(i).device());
      }
    }

    for (int i = 0; i < def.input_size(); ++i) {
      if (def.input(i).is_argument_input()) {
        spec->AddArgumentInput(def.input(i).arg_name(), def.input(i).name());
      }
    }

    for (int i = 0; i < def.output_size(); ++i) {
      spec->AddOutput(def.output(i).name(), def.output(i).device());
    }
  }


Pipeline::Pipeline(const string &serialized_pipe, int batch_size, int num_threads, int device_id,
                   bool pipelined_execution, int prefetch_queue_depth, bool async_execution,
                   size_t bytes_per_sample_hint, bool set_affinity, int max_num_stream,
                   int default_cuda_stream_priority, int64_t seed)
        : built_(false), separated_execution_(false) {
    dali_proto::PipelineDef def;
    //  Reading Protobuf file has a limitation of 64 MB
    //  Following instructions will increase the
    //  limits to support the given pipeline
    google::protobuf::io::CodedInputStream coded_input(
            reinterpret_cast<const uint8_t *>(serialized_pipe.c_str()), serialized_pipe.size());
    coded_input.SetTotalBytesLimit(serialized_pipe.size());
    auto res = def.ParseFromCodedStream(&coded_input);
    DALI_ENFORCE(res, "Error parsing serialized pipeline.");

    // If not given, take parameters from the serialized pipeline
    this->max_batch_size_ = batch_size == -1 ? def.batch_size() : batch_size;
    DALI_ENFORCE(this->max_batch_size_ > 0,
                 make_string("You are trying to create a pipeline with an incorrect batch size (",
                             this->max_batch_size_, "). Please set the batch_size argument "
                             "to a positive integer."));

    this->device_id_ = device_id == -1 ? def.device_id() : device_id;
    DALI_ENFORCE(this->device_id_ >= 0 || this->device_id_ == CPU_ONLY_DEVICE_ID,
                 make_string("You are trying to create a pipeline with a negative device id (",
                             this->device_id_, "). Please set a correct device_id."));

    this->num_threads_ = num_threads == -1 ? static_cast<int>(def.num_threads()) : num_threads;
    DALI_ENFORCE(this->num_threads_ > 0,
                 make_string("You are trying to create a pipeline with an incorrect number "
                             "of worker threads (", this->num_threads_, "). Please set the "
                             "num_threads argument to a positive integer."));
    seed = seed == -1 ? def.seed() : seed;

    Init(this->max_batch_size_, this->num_threads_,
         this->device_id_, seed,
         pipelined_execution,
         separated_execution_,
         async_execution,
         bytes_per_sample_hint,
         set_affinity,
         max_num_stream,
         default_cuda_stream_priority,
         QueueSizes{prefetch_queue_depth});

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
    for (auto& output : def.pipe_outputs()) {
      this->output_names_.emplace_back(output.name(), output.device());
    }
  }

void Pipeline::Init(int max_batch_size, int num_threads, int device_id, int64_t seed,
                    bool pipelined_execution, bool separated_execution, bool async_execution,
                    size_t bytes_per_sample_hint, bool set_affinity, int max_num_stream,
                    int default_cuda_stream_priority, QueueSizes prefetch_queue_depth) {
    // guard cudaDeviceGetStreamPriorityRange call
    DeviceGuard g(device_id);
    this->max_batch_size_ = max_batch_size;
    this->num_threads_ = num_threads;
    this->device_id_ = device_id;
    using Clock = std::chrono::high_resolution_clock;
    this->original_seed_ = seed < 0 ? Clock::now().time_since_epoch().count() : seed;
    this->pipelined_execution_ = pipelined_execution;
    this->separated_execution_ = separated_execution;
    this->async_execution_ = async_execution;
    this->bytes_per_sample_hint_ = bytes_per_sample_hint;
    this->set_affinity_ = set_affinity;
    this->max_num_stream_ = max_num_stream;
    this->default_cuda_stream_priority_ = default_cuda_stream_priority;
    this->prefetch_queue_depth_ = prefetch_queue_depth;
    DALI_ENFORCE(max_batch_size_ > 0, "Max batch size must be greater than 0");

    int lowest_cuda_stream_priority = 0, highest_cuda_stream_priority = 0;
    // do it only for the GPU pipeline
    if (device_id != CPU_ONLY_DEVICE_ID) {
      CUDA_CALL(cudaDeviceGetStreamPriorityRange(&lowest_cuda_stream_priority,
                                                 &highest_cuda_stream_priority));
    }
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
    std::seed_seq ss{this->original_seed_};
    ss.generate(seed_.begin(), seed_.end());
  }

static bool has_prefix(const std::string &operator_name, const std::string& prefix) {
  if (operator_name.size() < prefix.size()) return false;
  return std::equal(operator_name.begin(), operator_name.begin() + prefix.size(),
                    prefix.begin());
}

int Pipeline::AddOperator(const OpSpec &spec, const std::string& inst_name) {
  return AddOperator(spec, inst_name, GetNextLogicalId());
}

int Pipeline::AddOperator(const OpSpec &spec, int logical_id) {
  return AddOperator(spec, make_string("__", spec.name(), "_", logical_id), logical_id);
}

int Pipeline::AddOperator(const OpSpec &spec) {
  return AddOperator(spec, GetNextLogicalId());
}


int Pipeline::AddOperator(const OpSpec &const_spec, const std::string& inst_name, int logical_id) {
  DALI_ENFORCE(!built_, "Alterations to the pipeline after "
      "\"Build()\" has been called are not allowed");
  DALI_ENFORCE(0 <= logical_id,
               "Logical id of the node must be positive, got " + std::to_string(logical_id) + ".");

  if (logical_id > next_logical_id_) {
    next_logical_id_ = logical_id + 1;
  }

  // we modify spec so copy it
  OpSpec spec = const_spec;
  auto operator_name = spec.name();
  // Take a copy of the passed OpSpec for serialization purposes before any modification
  this->op_specs_for_serialization_.push_back({inst_name, spec, logical_id});

  // Validate op device
  string device = spec.GetArgument<string>("device");
  DALI_ENFORCE(device == "cpu" || device == "gpu" || device == "mixed" || device == "support",
    "Invalid device argument \"" +  device +
    "\". Valid options are \"cpu\", \"gpu\" or \"mixed\"");

  DALI_ENFORCE(device != "gpu" || device_id_ != CPU_ONLY_DEVICE_ID,
               make_string("Cannot add a GPU operator ", operator_name, ", device_id should not be"
               " equal CPU_ONLY_DEVICE_ID."));

  if (device == "support") {
    DALI_WARN("\"support\" device is deprecated; use \"cpu\" or leave blank instead");
    device = "cpu";
    spec.SetArg("device", "cpu");
  }

  DeviceGuard g(device_id_);

  // Verify the regular inputs to the op
  for (int i = 0; i < spec.NumInput(); ++i) {
    if (spec.IsArgumentInput(i)) {
      // We only look at regular inputs here
      continue;
    }
    string input_name = spec.InputName(i);
    string input_device = spec.InputDevice(i);
    auto it = edge_names_.find(input_name);

    DALI_ENFORCE(it != edge_names_.end(), "Input '" + input_name +
        "' to op '" + spec.name() + "' is not known to the pipeline.");

    // Table of possible scenarios:
    // Op location / requested input type / data location
    // cpu / cpu / cpu -> everything is fine
    // cpu / cpu / gpu -> error, data does not exist on cpu
    // cpu / gpu / cpu -> error, cpu op not allowed to have gpu inputs
    // cpu / gpu / gpu -> both of above errors
    // gpu / cpu / cpu -> need to use contiguous version
    // gpu / cpu / gpu -> error, data not in specified location
    // gpu / gpu / cpu -> need to insert copy to device
    // gpu / gpu / gpu -> everything is fine
    // mixed / cpu / cpu -> everything is fine
    // mixed / cpu / gpu -> error, data does not exist on cpu
    // mixed / gpu / cpu -> error, mixed op not allowed to have gpu inputs
    // mixed / gpu / gpu -> both of above errors
    string error_str = "(op: '" + spec.name() + "', input: '" +
      input_name + "')";

    if (device == "cpu" || device == "mixed") {
      DALI_ENFORCE(input_device == "cpu", "cpu/mixed ops can only take cpu "
          "inputs. CPU op cannot follow GPU op. " + error_str);
      DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. CPU op cannot follow GPU op. " + error_str);
      DALI_ENFORCE(device_id_ != CPU_ONLY_DEVICE_ID || device == "cpu",
                   make_string("Cannot add a mixed operator ", operator_name,
                   " with a GPU output, device_id should not be CPU_ONLY_DEVICE_ID."));
    } else if (input_device == "cpu") {
      // device == gpu
      DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. CPU op cannot follow GPU op. " + error_str);
      SetupCPUInput(it, i, &spec);
    } else {
      SetupGPUInput(it);
    }
  }

  // Verify the argument inputs to the op
  for (const auto& arg_pair : spec.ArgumentInputs()) {
    Index input_idx = arg_pair.second;
    std::string input_name = spec.InputName(input_idx);
    auto it = edge_names_.find(input_name);

    DALI_ENFORCE(it != edge_names_.end(), "Input '" + input_name +
        "' to op '" + spec.name() + "' is not known to the pipeline.");

    string error_str = "(op: '" + spec.name() + "', input: '" +
      input_name + "')";

    DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
        "only on GPU. " + error_str);

    if (device == "gpu" && separated_execution_)
      SetupCPUInput(it, input_idx, &spec);
  }

  // Verify and record the outputs of the op
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string output_name = spec.OutputName(i);
    string output_device = spec.OutputDevice(i);
    string error_str = "(op: '" + spec.name() + "', output: '" +
      output_name + "')";

    auto it = edge_names_.find(output_name);
    DALI_ENFORCE(it == edge_names_.end(), "Output name '" +
        output_name + "' conflicts with existing intermediate "
        "result name. " + error_str);

    // Validate output data conforms to graph constraints
    bool mark_explicitly_contiguous = false;
    if (device == "cpu") {
      DALI_ENFORCE(output_device == "cpu", "Only CPU operators can produce CPU outputs." +
                                            error_str);
    } else if (device == "gpu") {
      if (output_device == "cpu") {
        mark_explicitly_contiguous = true;
      }
    }

    EdgeMeta meta = NewEdge(output_device);
    if (mark_explicitly_contiguous) {
      meta.has_contiguous = true;
    }

    DALI_ENFORCE(edge_names_.insert({output_name, meta}).second,
        "Output name insertion failure.");
  }

  // store updated spec
  AddToOpSpecs(inst_name, spec, logical_id);
  return logical_id;
}

bool Pipeline::IsLogicalIdUsed(int logical_id) const {
  return logical_ids_.find(logical_id) != logical_ids_.end();
}

void Pipeline::AddToOpSpecs(const std::string &inst_name, const OpSpec &spec, int logical_id) {
  auto& logical_group = logical_ids_[logical_id];
  if (logical_group.size() > 0) {
    const auto &group_name = op_specs_[logical_group.front()].spec.name();
    DALI_ENFORCE(
        group_name == spec.name(),
        "Different Operator types cannot be groupped with the same logical id. Tried to group " +
            spec.name() + " using logical_id=" + std::to_string(logical_id) +
            " which is already assigned to " + group_name + ".");
    const OpSchema &schema = SchemaRegistry::GetSchema(spec.name());
    DALI_ENFORCE(schema.AllowsInstanceGrouping(),
                 "Operator " + spec.name() + " does not support synced random execution required "
                      "for multiple input sets processing.");
  }
  op_specs_.push_back({inst_name, spec, logical_id});
  logical_ids_[logical_id].push_back(op_specs_.size() - 1);
}

inline int GetMemoryHint(OpSpec &spec, int index) {
  if (!spec.HasArgument("bytes_per_sample_hint"))
    return 0;
  std::vector<int> hints;
  GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", spec.NumOutput());

  DALI_ENFORCE(index < static_cast<int>(hints.size()),
               "Output index out of range: " + std::to_string(index));
  return hints[index];
}

inline void SetMemoryHint(OpSpec &spec, int index, int value) {
  std::vector<int> hints;
  int no = spec.NumOutput();

  DALI_ENFORCE(index < no, "Output index out of range: " +
    std::to_string(index) + " >= " + std::to_string(no));

  GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", no);
  hints[index] = value;
  spec.SetArg("bytes_per_sample_hint", hints);
}

void Pipeline::PropagateMemoryHint(OpNode &node) {
  assert(node.parents.size() == 1);
  for (int inp_idx = 0; inp_idx < node.spec.NumRegularInput(); inp_idx++) {
    auto input_name = node.spec.Input(inp_idx);
    OpNodeId parent_node_id = graph_.TensorSourceID(input_name);
    int idx = graph_.TensorIdxInSource(input_name);
    auto &src = graph_.Node(parent_node_id);
    int hint = GetMemoryHint(src.spec, idx);
    if (hint) {
      // inp_idx == out_idx for MakeContiguous
      SetMemoryHint(node.spec, inp_idx, hint);
    }
  }
}

void Pipeline::Build(vector<std::pair<string, string>> output_names) {
  DeviceGuard d(device_id_);
  output_names_ = output_names;
  DALI_ENFORCE(!built_, "\"Build()\" can only be called once.");
  DALI_ENFORCE(output_names.size() > 0, "User specified zero outputs.");

  executor_ =
      GetExecutor(pipelined_execution_, separated_execution_, async_execution_, max_batch_size_,
                  num_threads_, device_id_, bytes_per_sample_hint_, set_affinity_, max_num_stream_,
                  default_cuda_stream_priority_, prefetch_queue_depth_);
  executor_->EnableMemoryStats(enable_memory_stats_);
  executor_->Init();

  // Creating the graph
  for (auto& name_op_spec : op_specs_) {
    string& inst_name = name_op_spec.instance_name;
    OpSpec op_spec = name_op_spec.spec;
    PrepareOpSpec(&op_spec, name_op_spec.logical_id);
    try {
      graph_.AddOp(op_spec, inst_name);
    } catch (std::exception &e) {
      int same_op_count = 0;
      for (const auto& elem : op_specs_) {
        if (elem.spec.name() == op_spec.name()) {
          same_op_count++;
        }
        if (same_op_count > 1) {
          break;
        }
      }
      if (same_op_count > 1) {
        throw std::runtime_error(make_string(
            "Critical error when building pipeline:\nError when adding operator: ", op_spec.name(),
            ", instance name: \"", inst_name, "\", encountered:\n", e.what(),
            "\nCurrent pipeline object is no longer valid."));
      } else {
        throw std::runtime_error(make_string(
            "Critical error when building pipeline:\nError when adding operator: ", op_spec.name(),
            "\" encountered:\n", e.what(),
            "\nCurrent pipeline object is no longer valid."));
      }
    } catch (...) {
      throw std::runtime_error("Unknown critical error when building pipeline.");
    }
  }

  // Validate the output tensors names
  vector<string> outputs;
  for (const auto &name_pair : output_names) {
    string name = name_pair.first;
    string device = name_pair.second;
    auto it = edge_names_.find(name);
    DALI_ENFORCE(it != edge_names_.end(), "Requested output name '" +
        name + "' is not known to the pipeline.");

    if (device == "cpu") {
      DALI_ENFORCE(it->second.has_cpu, "Requested cpu output '" +
          name + "' only exists on gpu.");

      if (!it->second.has_contiguous) {
        // Add a make contiguous op to produce this output
        OpSpec spec =
          OpSpec("MakeContiguous")
          .AddArg("device", "cpu")
          .AddInput(name, "cpu")
          .AddOutput("contiguous_" + name, "cpu");
        PrepareOpSpec(&spec, GetNextInternalLogicalId());

        graph_.AddOp(spec, "__MakeContiguous_" + name);
        it->second.has_contiguous = true;
      }
      outputs.push_back("contiguous_" + name + "_" + device);
    } else if (device == "gpu") {
      DALI_ENFORCE(device_id_ != CPU_ONLY_DEVICE_ID,
                   make_string("Cannot make a gpu output for ", name, " operator,"
                   " device_id should not be equal CPU_ONLY_DEVICE_ID."));
      if (!it->second.has_gpu) {
        DALI_ENFORCE(it->second.has_cpu, "Output '" + name +
            "' exists on neither cpu or gpu, internal error");
        // Add a copy to device to create the gpu output
        OpSpec spec = OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput(name, "cpu")
          .AddOutput(name, "gpu");
        PrepareOpSpec(&spec, GetNextLogicalId());
        graph_.AddOp(spec, "__MakeContiguous_" + name);
      }
      outputs.push_back(name + "_" + device);
    } else {
      DALI_FAIL("Invalid device argument \"" + device +
          "\". Valid options are \"cpu\" or \"gpu\"");
    }
  }

  for (int i = 0; i < graph_.NumOp(OpType::MIXED); i++) {
    OpNode &node = graph_.Node(OpType::MIXED, i);
    if (node.spec.name() == "MakeContiguous") {
      PropagateMemoryHint(node);
    }
  }

  graph_.InstantiateOperators();

  // Load the final graph into the executor
  executor_->Build(&graph_, outputs);
  built_ = true;
}

void Pipeline::SetOutputNames(const vector<std::pair<string, string>> &output_names) {
  output_names_ = output_names;
}

void Pipeline::RunCPU() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->RunCPU();
}

void Pipeline::RunGPU() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->RunMixed();
  executor_->RunGPU();
}

void Pipeline::SetCompletionCallback(ExecutorBase::ExecutorCallback cb) {
  executor_->SetCompletionCallback(std::move(cb));
}

void Pipeline::Outputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->Outputs(ws);
    } catch (std::exception &e) {
      throw std::runtime_error("Critical error in pipeline:\n"
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown critical error in pipeline.");
    }
}

void Pipeline::ShareOutputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->ShareOutputs(ws);
    } catch (std::exception &e) {
      throw std::runtime_error("Critical error in pipeline:\n"
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown critical error in pipeline.");
    }
}

void Pipeline::ReleaseOutputs() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->ReleaseOutputs();
    } catch (std::exception &e) {
      throw std::runtime_error("Critical error in pipeline:\n"
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown critical error in pipeline.");
    }
}

void Pipeline::SetupCPUInput(std::map<string, EdgeMeta>::iterator it, int input_idx, OpSpec *spec) {
  if (!it->second.has_contiguous) {
    OpSpec make_contiguous_spec =
      OpSpec("MakeContiguous")
      .AddArg("device", "cpu")
      .AddInput(it->first, "cpu")
      .AddOutput("contiguous_" + it->first, "cpu");
    // don't put it into op_specs_for_serialization_, only op_specs_
    AddToOpSpecs("__MakeContiguous_" + it->first, make_contiguous_spec, GetNextInternalLogicalId());
    it->second.has_contiguous = true;
  }

  // Update the OpSpec to use the contiguous input
  auto& input_strs = spec->MutableInput(input_idx);
  DALI_ENFORCE(input_strs.name == it->first, "Input at index " +
      std::to_string(input_idx) + " does not match input iterator "
      "name (" + input_strs.name + " v. " + it->first + ").");
  input_strs.name = "contiguous_" + input_strs.name;
}

void Pipeline::SetupGPUInput(std::map<string, EdgeMeta>::iterator it) {
  if (it->second.has_gpu) return;
  OpSpec copy_to_dev_spec =
    OpSpec("MakeContiguous")
    .AddArg("device", "mixed")
    .AddInput(it->first, "cpu")
    .AddOutput(it->first, "gpu");
  // don't put it into op_specs_for_serialization_, only op_specs_
  AddToOpSpecs("__Copy_" + it->first, copy_to_dev_spec, GetNextInternalLogicalId());
  it->second.has_gpu = true;
}

void Pipeline::PrepareOpSpec(OpSpec *spec, int logical_id) {
  if (logical_id_to_seed_.find(logical_id) == logical_id_to_seed_.end()) {
    logical_id_to_seed_[logical_id] = seed_[current_seed_];
  }
  spec->AddArg("max_batch_size", max_batch_size_)
    .AddArg("num_threads", num_threads_)
    .AddArg("device_id", device_id_)
    .AddArgIfNotExisting("seed", logical_id_to_seed_[logical_id]);
  string dev = spec->GetArgument<string>("device");
  if (dev == "cpu" || dev == "mixed")
    spec->AddArg("cpu_prefetch_queue_depth", prefetch_queue_depth_.cpu_size);
  if (dev == "gpu" || dev == "mixed")
    spec->AddArg("gpu_prefetch_queue_depth", prefetch_queue_depth_.gpu_size);
  current_seed_ = (current_seed_+1) % MAX_SEEDS;
}

/**
 * @brief Helper method that serialized OpSpec
 * decouples spec class from dali_proto
 */
void SerializeToProtobuf(dali_proto::OpDef *op, const string &inst_name, const OpSpec &spec,
                         int logical_id) {
  op->set_name(spec.name());
  op->set_inst_name(inst_name);
  op->set_logical_id(logical_id);

  for (int i = 0; i < spec.NumInput(); ++i) {
    dali_proto::InputOutput *in = op->add_input();
    in->set_name(spec.InputName(i));
    in->set_device(spec.InputDevice(i));
    if (spec.IsArgumentInput(i)) {
        in->set_arg_name(spec.ArgumentInputName(i));
    }
    in->set_is_argument_input(spec.IsArgumentInput(i));
  }

  for (int i = 0; i < spec.NumOutput(); ++i) {
    dali_proto::InputOutput *out = op->add_output();
    out->set_name(spec.OutputName(i));
    out->set_device(spec.OutputDevice(i));
    out->set_is_argument_input(false);
  }

  for (auto& a : spec.Arguments()) {
    // filter out args that need to be dealt with on
    // loading a serialized pipeline
    if (a.first == "max_batch_size" ||
        a.first == "num_threads" ||
        a.first == "bytes_per_sample_hint") {
      continue;
    }

    dali_proto::Argument *arg = op->add_args();
    DaliProtoPriv arg_wrap(arg);

    a.second->SerializeToProtobuf(&arg_wrap);
  }
}

string Pipeline::SerializeToProtobuf() const {
  dali_proto::PipelineDef pipe;
  pipe.set_num_threads(this->num_threads());
  pipe.set_batch_size(this->max_batch_size());
  pipe.set_device_id(this->device_id());
  pipe.set_seed(this->original_seed_);

  // loop over ops, create messages and append
  for (size_t i = 0; i < this->op_specs_for_serialization_.size(); ++i) {
    dali_proto::OpDef *op_def = pipe.add_op();

    const auto& p = this->op_specs_for_serialization_[i];
    const OpSpec& spec = p.spec;

    DALI_ENFORCE(spec.GetSchema().IsSerializable(), "Could not serialize the operator: "
                                                    + spec.name());

    dali::SerializeToProtobuf(op_def, p.instance_name, spec, p.logical_id);
  }

  // loop over outputs used to create the graph
  for (auto& output : output_names_) {
    dali_proto::InputOutput *out = pipe.add_pipe_outputs();

    out->set_name(output.first);
    out->set_device(output.second);
    out->set_is_argument_input(false);
  }
  pipe.set_device_id(this->device_id_);
  string output = pipe.SerializeAsString();

#ifndef NDEBUG
  // print out debug string
  printf("%s\n", pipe.DebugString().c_str());
#endif

  return output;
}

OpNode * Pipeline::GetOperatorNode(const std::string& name) {
  return &(graph_.Node(name));
}

std::map<std::string, ReaderMeta> Pipeline::GetReaderMeta() {
  std::map<std::string, ReaderMeta> ret;
  for (Index i = 0; i < graph_.NumOp(); ++i) {
    const OpNode &current = graph_.Node(i);
    ReaderMeta meta = current.op->GetReaderMeta();
    if (meta) {
      ret.insert(make_pair(current.instance_name, meta));
    }
  }
  return ret;
}

ReaderMeta Pipeline::GetReaderMeta(std::string name) {
  ReaderMeta meta;
  for (Index i = 0; i < graph_.NumOp(); ++i) {
    const OpNode &current = graph_.Node(i);
    if (current.instance_name == name) {
      meta = current.op->GetReaderMeta();
      break;
    }
  }
  return meta;
}

const std::string &Pipeline::output_name(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_name()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_names_.size());
  return output_names_[id].first;
}

const std::string &Pipeline::output_device(int id) const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"output_device()\".");
  DALI_ENFORCE_VALID_INDEX(id, output_names_.size());
  return output_names_[id].second;
}

int Pipeline::num_outputs() const {
  DALI_ENFORCE(built_, "\"Build()\" must be called prior to calling \"num_outputs()\".");
  return output_names_.size();
}

void Pipeline::SaveGraphToDotFile(const std::string &filename, bool show_tensors, bool show_ids,
                                  bool use_colors) {
  graph_.SaveToDotFile(filename, show_tensors, show_ids, use_colors);
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

}  // namespace dali
