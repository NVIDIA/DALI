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

#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/util/device_guard.h"

namespace dali {

void Pipeline::AddOperator(OpSpec spec, const std::string& inst_name) {
  DALI_ENFORCE(!built_, "Alterations to the pipeline after "
      "\"Build()\" has been called are not allowed");

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

void Pipeline::Build(vector<std::pair<string, string>> output_names) {
  output_names_ = output_names;
  DALI_ENFORCE(!built_, "\"Build()\" can only be called once.");
  DALI_ENFORCE(output_names.size() > 0, "User specified zero outputs.");


  // Creating the executor
  if (pipelined_execution_ && async_execution_) {
    executor_.reset(new AsyncPipelinedExecutor(
            batch_size_, num_threads_,
            device_id_, bytes_per_sample_hint_,
            set_affinity_, max_num_stream_));
    executor_->Init();
  } else if (pipelined_execution_) {
    executor_.reset(new PipelinedExecutor(
            batch_size_, num_threads_,
            device_id_, bytes_per_sample_hint_,
            set_affinity_, max_num_stream_));
  } else if (async_execution_) {
    DALI_FAIL("Not implemented.");
  } else {
    executor_.reset(new Executor(
            batch_size_, num_threads_,
            device_id_, bytes_per_sample_hint_,
            set_affinity_, max_num_stream_));
  }

  // Creating the graph
  for (auto& name_op_spec : op_specs_) {
    string& inst_name = name_op_spec.first;
    OpSpec op_spec = name_op_spec.second;
    PrepareOpSpec(&op_spec);
    graph_.AddOp(op_spec, inst_name);
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

  DeviceGuard d(device_id_);
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
  executor_->RunMixed();
}

void Pipeline::RunGPU() {
  DALI_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->RunGPU();
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

void Pipeline::SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
    int input_idx, OpSpec *spec) {
  if (!it->second.has_contiguous) {
    // We check if the make contiguous op already exists
    std::string op_name = "__MakeContiguous_" + it->first;
    if (std::find_if(op_specs_.begin(), op_specs_.end(),
          [&op_name] (const std::pair<string, OpSpec>& p) {
                  return p.first == op_name;}) != op_specs_.end()) {
      return;
    }

    OpSpec make_contiguous_spec =
      OpSpec("MakeContiguous")
      .AddArg("device", "mixed")
      .AddInput(it->first, "cpu")
      .AddOutput("contiguous_" + it->first, "cpu");
    this->op_specs_.push_back(make_pair("__MakeContiguous_" + it->first, make_contiguous_spec));
    this->op_specs_to_serialize_.push_back(false);
  }

  // Update the OpSpec to use the contiguous input
  auto input_strs = spec->mutable_input(input_idx);
  DALI_ENFORCE(input_strs->first == it->first, "Input at index " +
      std::to_string(input_idx) + " does not match input iterator "
      "name (" + input_strs->first + " v. " + it->first + ").");
  input_strs->first = "contiguous_" + input_strs->first;
}

void Pipeline::SetupGPUInput(std::map<string, EdgeMeta>::iterator it) {
  if (it->second.has_gpu) return;
  // We check if the copy_to_dev op already exists
  std::string op_name = "__Copy_" + it->first;
  if (std::find_if(op_specs_.begin(), op_specs_.end(),
        [&op_name] (const std::pair<string, OpSpec>& p) {
                return p.first == op_name;}) != op_specs_.end()) {
    return;
  }

  OpSpec copy_to_dev_spec =
    OpSpec("MakeContiguous")
    .AddArg("device", "mixed")
    .AddInput(it->first, "cpu")
    .AddOutput(it->first, "gpu");
  this->op_specs_.push_back(make_pair("__Copy_" + it->first, copy_to_dev_spec));
  this->op_specs_to_serialize_.push_back(false);
}

void Pipeline::PrepareOpSpec(OpSpec *spec) {
  spec->AddArg("batch_size", batch_size_)
    .AddArg("num_threads", num_threads_)
    .AddArg("bytes_per_sample_hint", bytes_per_sample_hint_)
    .AddArg("seed", seed_[current_seed])
    .AddArg("device_id", device_id_);
  current_seed = (current_seed+1) % MAX_SEEDS;
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
        spec.SerializeToProtobuf(op_def, p.first);
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
  return &(graph_.node(name));
}

std::map<std::string, Index> Pipeline::EpochSize() {
  std::map<std::string, Index> ret;
  for (Index i = 0; i < graph_.NumCPUOp(); ++i) {
    const OpNode& current = graph_.cpu_node(i);
    Index epoch_size = current.op->epoch_size();
    if (epoch_size != -1) {
      ret.insert(make_pair(current.instance_name, epoch_size));
    }
  }
  for (Index i = 0; i < graph_.NumGPUOp(); ++i) {
    const OpNode& current = graph_.gpu_node(i);
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
