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

#include "dali/pipeline/pipeline.h"

#include <google/protobuf/message.h>

#include <algorithm>
#include <functional>
#include <memory>

#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"
#include "dali/pipeline/executor/executor_factory.h"
#include "dali/pipeline/executor/pipelined_executor.h"

#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/operators/common.h"
#include "dali/util/device_guard.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

  void DeserializeOpSpec(const dali_proto::OpDef& def, OpSpec* spec) {
    spec->set_name(def.name());

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
                     size_t bytes_per_sample_hint, bool set_affinity, int max_num_stream)
      : built_(false), separated_execution_(false) {
    dali_proto::PipelineDef def;
    def.ParseFromString(serialized_pipe);

    // If not given, take parameters from the
    // serialized pipeline
    if (batch_size == -1) {
      this->batch_size_ = def.batch_size();
    } else {
      this->batch_size_ = batch_size;
    }
    if (device_id == -1) {
      this->device_id_ = def.device_id();
    } else {
      this->device_id_ = device_id;
    }
    if (num_threads == -1) {
      this->num_threads_ = def.num_threads();
    } else {
      this->num_threads_ = num_threads;
    }

    Init(this->batch_size_, this->num_threads_,
         this->device_id_, def.seed(),
         pipelined_execution,
         separated_execution_,  // We use false as default for now
         async_execution,
         bytes_per_sample_hint,
         set_affinity,
         max_num_stream,
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

      this->AddOperator(spec, op_def.inst_name());
    }
    // output names
    for (auto& output : def.pipe_outputs()) {
      this->output_names_.push_back(std::make_pair(output.name(), output.device()));
    }
  }

void Pipeline::AddOperator(OpSpec spec, const std::string& inst_name) {
  DALI_ENFORCE(!built_, "Alterations to the pipeline after "
      "\"Build()\" has been called are not allowed");

#if 1
  // TODO(spanev): - nvJPEGDecoderSplitted by nvJPEGDecoderNew
  //               - make it as an arg of nvJPEGDecoderNew

  // nvJGPEGDecoder operator require a special case to be divided in two stages (CPU and Mixed-GPU)
  std::string fullname = spec.name();
  // TODO(janton): make this cleaner and smarter
  if (fullname == "nvJPEGDecoderSplitted") {
    AddSplittedNvJpegDecoder(spec, inst_name, fullname, "nvJPEGDecoderCPUStage",
                             "nvJPEGDecoderGPUStage");
    return;
  } else if (fullname == "nvJPEGDecoderSplittedCrop") {
    AddSplittedNvJpegDecoder(spec, inst_name, fullname, "nvJPEGDecoderCPUStageCrop",
                             "nvJPEGDecoderGPUStage");
    return;
  } else if (fullname == "nvJPEGDecoderSplittedSlice") {
    AddSplittedNvJpegDecoder(spec, inst_name, fullname, "nvJPEGDecoderCPUStageSlice",
                             "nvJPEGDecoderGPUStage");
    return;
  } else if (fullname == "nvJPEGDecoderSplittedRandomCrop") {
    AddSplittedNvJpegDecoder(spec, inst_name, fullname, "nvJPEGDecoderCPUStageRandomCrop",
                             "nvJPEGDecoderGPUStage");
    return;
  }
#endif

  // Validate op device
  string device = spec.GetArgument<string>("device");
  DALI_ENFORCE(device == "cpu" ||
               device == "gpu" ||
               device == "mixed" ||
               device == "support", "Invalid "
      "device argument \"" + device + "\". Valid options are "
      "\"cpu\", \"gpu\", \"mixed\" or \"support\"");

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
          "inputs. " + error_str);
      DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
      DALI_ENFORCE(!it->second.is_support,
          "Argument input can only be used as regular input by support ops. " + error_str);
    } else if (device == "support") {
      DALI_ENFORCE(input_device == "cpu", "Support ops can only take cpu inputs. " + error_str);
      DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
      DALI_ENFORCE(it->second.is_support,
          "Support ops can only take inputs produced by other support ops." + error_str);
    } else if (input_device == "cpu") {
      // device == gpu
      DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
      SetupCPUInput(it, i, &spec);
    } else {
      SetupGPUInput(it);
    }
  }

  // Verify the argument inputs to the op
  for (const auto& arg_pair : spec.ArgumentInputs()) {
    std::string input_name = spec.InputName(arg_pair.second);
    auto it = edge_names_.find(input_name);

    DALI_ENFORCE(it != edge_names_.end(), "Input '" + input_name +
        "' to op '" + spec.name() + "' is not known to the pipeline.");

    string error_str = "(op: '" + spec.name() + "', input: '" +
      input_name + "')";

    DALI_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
        "only on GPU. " + error_str);
    DALI_ENFORCE(it->second.is_support, "Argument input may only be produced by support op.");
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
    if (device == "cpu" || device == "support") {
      DALI_ENFORCE(output_device == "cpu", "CPU and support ops can only produce "
          "CPU outputs." + error_str);
    } else if (device == "gpu") {
      if (output_device == "cpu") {
        mark_explicitly_contiguous = true;
      }
    }

    EdgeMeta meta = NewEdge(output_device);
    if (mark_explicitly_contiguous) {
      meta.has_contiguous = true;
    }
    if (device == "support") {
      meta.has_contiguous = true;
      meta.is_support = true;
    }

    DALI_ENFORCE(edge_names_.insert({output_name, meta}).second,
        "Output name insertion failure.");
  }

  // Take a copy of the passed OpSpec for serialization purposes
  this->op_specs_.push_back(make_pair(inst_name, spec));
  this->op_specs_to_serialize_.push_back(true);
}

inline void Pipeline::AddSplittedNvJpegDecoder(OpSpec& spec, const std::string& inst_name,
                                               const std::string& full_name,
                                               const std::string& cpu_stage_name,
                                               const std::string& gpu_stage_name) {
  spec.set_name(cpu_stage_name);
  spec.SetArg("device", "cpu");

  auto& op_output = spec.MutableOutput(0);
  string op_output_name = op_output.first;

  const std::string mangled_outputname("nvJPEGCPUOutput" + inst_name);
  op_output.first = mangled_outputname + "0";
  op_output.second = "cpu";
  spec.AddOutput(mangled_outputname + "1", "cpu");
  spec.AddOutput(mangled_outputname + "2", "cpu");

  OpSpec gpu_spec = OpSpec(gpu_stage_name)
    .ShareArguments(spec)
    .AddInput(spec.OutputName(0), "cpu")
    .AddInput(spec.OutputName(1), "cpu")
    .AddInput(spec.OutputName(2), "cpu")
    .AddOutput(op_output_name, "gpu");
  gpu_spec.SetArg("device", "mixed");

  // TODO(spanev): handle serialization for nvJPEGDecoderNew
  this->AddOperator(spec, inst_name);
  this->AddOperator(gpu_spec, inst_name + "_gpu");
}

inline int GetMemoryHint(OpSpec &spec, int index) {
  if (!spec.HasArgument("bytes_per_sample_hint"))
    return 0;
  std::vector<int> hints;
  GetSingleOrRepeatedArg(spec, &hints, "bytes_per_sample_hint", spec.NumOutput());

  DALI_ENFORCE(index < static_cast<int>(hints.size()),
               "Output index out of range: " + std::to_string(index));
  return hints[index];
}

inline void SetMemoryHint(OpSpec &spec, int index, int value) {
  std::vector<int> hints;
  int no = spec.NumOutput();

  DALI_ENFORCE(index < no, "Output index out of range: " +
    std::to_string(index) + " >= " + std::to_string(no));

  GetSingleOrRepeatedArg(spec, &hints, "bytes_per_sample_hint", no);
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

  separated_execution_ = true;
  prefetch_queue_depth_ = {7, 4};
  executor_ = GetExecutor(pipelined_execution_, separated_execution_, async_execution_, batch_size_,
                          num_threads_, device_id_, bytes_per_sample_hint_, set_affinity_,
                          max_num_stream_, prefetch_queue_depth_);
  executor_->Init();

  // Creating the graph
  for (auto& name_op_spec : op_specs_) {
    string& inst_name = name_op_spec.first;
    OpSpec op_spec = name_op_spec.second;
    PrepareOpSpec(&op_spec);
    try {
      graph_.AddOp(op_spec, inst_name);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Critical error in pipeline: "
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown Critical error in pipeline");
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
          .AddArg("device", "mixed")
          .AddInput(name, "cpu")
          .AddOutput("contiguous_" + name, "cpu");
        PrepareOpSpec(&spec);

        graph_.AddOp(spec, "__MakeContiguous_" + name);

        outputs.push_back("contiguous_" + name + "_" + device);
      } else {
        // handle contiguous output from gpu ops
        outputs.push_back(name + "_" + device);
      }
    } else if (device == "gpu") {
      if (!it->second.has_gpu) {
        DALI_ENFORCE(it->second.has_cpu, "Output '" + name +
            "' exists on neither cpu or gpu, internal error");
        // Add a copy to device to create the gpu output
        OpSpec spec = OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput(name, "cpu")
          .AddOutput(name, "gpu");
        PrepareOpSpec(&spec);
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

void Pipeline::SetOutputNames(vector<std::pair<string, string>> output_names) {
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
  executor_->SetCompletionCallback(cb);
}

void Pipeline::Outputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->Outputs(ws);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Critical error in pipeline: "
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown Critical error in pipeline");
    }
}

void Pipeline::ShareOutputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->ShareOutputs(ws);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Critical error in pipeline: "
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown Critical error in pipeline");
    }
}

void Pipeline::ReleaseOutputs() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
    try {
      executor_->ReleaseOutputs();
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Critical error in pipeline: "
          + std::string(e.what())
          + "\nCurrent pipeline object is no longer valid.");
    } catch (...) {
      throw std::runtime_error("Unknown Critical error in pipeline");
    }
}

void Pipeline::SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
    int input_idx, OpSpec *spec) {
  if (!it->second.has_contiguous) {
    OpSpec make_contiguous_spec =
      OpSpec("MakeContiguous")
      .AddArg("device", "mixed")
      .AddInput(it->first, "cpu")
      .AddOutput("contiguous_" + it->first, "cpu");
    this->op_specs_.push_back(make_pair("__MakeContiguous_" + it->first, make_contiguous_spec));
    this->op_specs_to_serialize_.push_back(false);
    it->second.has_contiguous = true;
  }

  // Update the OpSpec to use the contiguous input
  auto& input_strs = spec->MutableInput(input_idx);
  DALI_ENFORCE(input_strs.first == it->first, "Input at index " +
      std::to_string(input_idx) + " does not match input iterator "
      "name (" + input_strs.first + " v. " + it->first + ").");
  input_strs.first = "contiguous_" + input_strs.first;
}

void Pipeline::SetupGPUInput(std::map<string, EdgeMeta>::iterator it) {
  if (it->second.has_gpu) return;
  OpSpec copy_to_dev_spec =
    OpSpec("MakeContiguous")
    .AddArg("device", "mixed")
    .AddInput(it->first, "cpu")
    .AddOutput(it->first, "gpu");
  this->op_specs_.push_back(make_pair("__Copy_" + it->first, copy_to_dev_spec));
  this->op_specs_to_serialize_.push_back(false);
  it->second.has_gpu = true;
}

void Pipeline::PrepareOpSpec(OpSpec *spec) {
  spec->AddArg("batch_size", batch_size_)
    .AddArg("num_threads", num_threads_)
    .AddArg("device_id", device_id_)
    .AddArgIfNotExisting("seed", seed_[current_seed_]);
  current_seed_ = (current_seed_+1) % MAX_SEEDS;
}

/**
 * @brief Helper method that serialized OpSpec
 * decouples spec class from dali_proto
 */
void SerializeToProtobuf(dali_proto::OpDef *op, const string& inst_name,
                            const OpSpec& spec) {
  op->set_name(spec.name());
  op->set_inst_name(inst_name);

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
    if (a.first == "batch_size" ||
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
  pipe.set_batch_size(this->batch_size());
  pipe.set_device_id(this->device_id());
  pipe.set_seed(this->original_seed_);

  // loop over external inputs
  for (auto &name : external_inputs_) {
    pipe.add_external_inputs(name);
  }

  // loop over ops, create messages and append
  for (size_t i = 0; i < this->op_specs_.size(); ++i) {
    if (op_specs_to_serialize_[i]) {
      dali_proto::OpDef *op_def = pipe.add_op();

      const auto& p = this->op_specs_[i];
      const OpSpec& spec = p.second;

      // As long as spec isn't an ExternalSource node, serialize
      if (spec.name() != "ExternalSource") {
        dali::SerializeToProtobuf(op_def, p.first, spec);
      }
    }
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

  return pipe.SerializeAsString();
}

OpNode * Pipeline::GetOperatorNode(const std::string& name) {
  return &(graph_.Node(name));
}

std::map<std::string, Index> Pipeline::EpochSize() {
  std::map<std::string, Index> ret;
  for (Index i = 0; i < graph_.NumOp(OpType::CPU); ++i) {
    const OpNode &current = graph_.Node(OpType::CPU, i);
    Index epoch_size = current.op->epoch_size();
    if (epoch_size != -1) {
      ret.insert(make_pair(current.instance_name, epoch_size));
    }
  }
  for (Index i = 0; i < graph_.NumOp(OpType::GPU); ++i) {
    const OpNode &current = graph_.Node(OpType::GPU, i);
    Index epoch_size = current.op->epoch_size();
    if (epoch_size != -1) {
      ret.insert(make_pair(current.instance_name, epoch_size));
    }
  }
  return ret;
}

void Pipeline::SaveGraphToDotFile(const std::string filename) {
  graph_.SaveToDotFile(filename);
}

}  // namespace dali
