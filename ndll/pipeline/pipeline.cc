// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/pipeline.h"

#include <google/protobuf/message.h>

#include <algorithm>
#include <functional>
#include <memory>

#include "ndll/pipeline/argument.h"

namespace ndll {

// TODO(tgale): The constraint that GPU ops cannot produce CPU
// outputs is arbitrary. We could easily enable cpu/gpu outputs
// for gpu ops, do we want to do this?
void Pipeline::AddOperator(OpSpec spec, const std::string& inst_name) {
  NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
      "\"Build()\" has been called are not allowed");

  // Take a copy of the passed OpSpec for serialization purposes
  this->op_specs_.push_back(make_pair(inst_name, spec));

  // Validate op device
  string device = spec.GetArgument<string>("device", "cpu");
  NDLL_ENFORCE(device == "cpu" || device == "gpu" || device == "mixed", "Invalid "
      "device argument \"" + device + "\". Valid options are "
      "\"cpu\", \"gpu\" or \"mixed\"");

  int old_device;
  CUDA_CALL(cudaGetDevice(&old_device));
  CUDA_CALL(cudaSetDevice(device_id_));

  // Verify the inputs to the op
  for (int i = 0; i < spec.NumInput(); ++i) {
    string input_name = spec.InputName(i);
    string input_device = spec.InputDevice(i);
    auto it = edge_names_.find(input_name);

    NDLL_ENFORCE(it != edge_names_.end(), "Input '" + input_name +
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
      NDLL_ENFORCE(input_device == "cpu", "cpu/mixed ops can only take cpu "
          "inputs. " + error_str);
      NDLL_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
    } else if (input_device == "cpu") {
      NDLL_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
      SetupCPUInput(it, i, &spec);
    } else {
      SetupGPUInput(it);
    }
  }

  // Verify and record the outputs of the op
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string output_name = spec.OutputName(i);
    string output_device = spec.OutputDevice(i);
    string error_str = "(op: '" + spec.name() + "', output: '" +
      output_name + "')";

    auto it = edge_names_.find(output_name);
    NDLL_ENFORCE(it == edge_names_.end(), "Output name '" +
        output_name + "' conflicts with existing intermediate "
        "result name. " + error_str);

    // Validate output data conforms to graph constraints
    if (device == "cpu") {
      NDLL_ENFORCE(output_device == "cpu", "cpu ops can only produce "
          "cpu outputs." + error_str);
    } else {
      NDLL_ENFORCE(output_device == "gpu", "gpu ops can only produce "
          "gpu outputs." + error_str);
    }

    EdgeMeta meta = NewEdge(device);
    NDLL_ENFORCE(edge_names_.insert({output_name, meta}).second,
        "Output name insertion failure.");
  }

  // Add the operator to the graph
  PrepareOpSpec(&spec);
  graph_.AddOp(spec, inst_name);

  // Restore the original device
  CUDA_CALL(cudaSetDevice(old_device));
}

void Pipeline::Build(vector<std::pair<string, string>> output_names) {
  int old_device;
  CUDA_CALL(cudaGetDevice(&old_device));
  CUDA_CALL(cudaSetDevice(device_id_));
  output_names_ = output_names;
  NDLL_ENFORCE(!built_, "\"Build()\" can only be called once.");
  NDLL_ENFORCE(output_names.size() > 0, "User specified zero outputs.");

  // Validate the output tensors names
  vector<string> outputs;
  for (const auto &name_pair : output_names) {
    string name = name_pair.first;
    string device = name_pair.second;
    auto it = edge_names_.find(name);
    NDLL_ENFORCE(it != edge_names_.end(), "Requested output name '" +
        name + "' is not known to the pipeline.");

    if (device == "cpu") {
      NDLL_ENFORCE(it->second.has_cpu, "Requested cpu output '" +
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
      }
      outputs.push_back("contiguous_" + name + "_" + device);
    } else if (device == "gpu") {
      if (!it->second.has_gpu) {
        NDLL_ENFORCE(it->second.has_cpu, "Output '" + name +
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
      NDLL_FAIL("Invalid device argument \"" + device +
          "\". Valid options are \"cpu\" or \"gpu\"");
    }
  }

  // Load the final graph into the executor
  executor_->Build(&graph_, outputs);
  built_ = true;

  // Restore the original device
  CUDA_CALL(cudaSetDevice(old_device));
}

void Pipeline::RunCPU() {
  NDLL_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->RunCPU();
  executor_->RunMixed();
}

void Pipeline::RunGPU() {
  NDLL_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->RunGPU();
}

void Pipeline::Outputs(DeviceWorkspace *ws) {
  NDLL_ENFORCE(built_,
      "\"Build()\" must be called prior to executing the pipeline.");
  executor_->Outputs(ws);
}

void Pipeline::SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
    int input_idx, OpSpec *spec) {
  if (!it->second.has_contiguous) {
    if (graph_.TensorExists(OpSpec::TensorName("contiguous_" + it->first, "cpu"))) return;
    OpSpec make_contiguous_spec =
      OpSpec("MakeContiguous")
      .AddArg("device", "mixed")
      .AddInput(it->first, "cpu")
      .AddOutput("contiguous_" + it->first, "cpu");
    PrepareOpSpec(&make_contiguous_spec);
    graph_.AddOp(make_contiguous_spec, "__MakeContiguous_" + it->first);
  }

  // Update the OpSpec to use the contiguous input
  auto input_strs = spec->mutable_input(input_idx);
  NDLL_ENFORCE(input_strs->first == it->first, "Input at index " +
      std::to_string(input_idx) + " does not match input iterator "
      "name (" + input_strs->first + " v. " + it->first + ").");
  input_strs->first = "contiguous_" + input_strs->first;
}

void Pipeline::SetupGPUInput(std::map<string, EdgeMeta>::iterator it) {
  if (it->second.has_gpu) return;
    if (graph_.TensorExists(OpSpec::TensorName(it->first, "gpu"))) return;
  OpSpec copy_to_dev_spec =
    OpSpec("MakeContiguous")
    .AddArg("device", "mixed")
    .AddInput(it->first, "cpu")
    .AddOutput(it->first, "gpu");
  PrepareOpSpec(&copy_to_dev_spec);
  graph_.AddOp(copy_to_dev_spec, "__Copy_" + it->first);
}

void Pipeline::PrepareOpSpec(OpSpec *spec) {
  spec->AddArg("batch_size", batch_size_)
    .AddArg("num_threads", num_threads_)
    .AddArg("bytes_per_sample_hint", bytes_per_sample_hint_);
}

string Pipeline::SerializeToProtobuf() const {
  ndll_proto::PipelineDef pipe;
  pipe.set_num_threads(this->num_threads());
  pipe.set_batch_size(this->batch_size());

  // loop over external inputs
  for (auto &name : external_inputs_) {
    pipe.add_external_inputs(name);
  }

  // loop over ops, create messages and append
  for (size_t i = 0; i < this->op_specs_.size(); ++i) {
    ndll_proto::OpDef *op_def = pipe.add_op();

    const auto& p = this->op_specs_[i];
    const OpSpec& spec = p.second;

    // As long as spec isn't an ExternalSource node, serialize
    if (spec.name() != "ExternalSource") {
      spec.SerializeToProtobuf(op_def, p.first);
    }
  }

  // loop over outputs used to create the graph
  for (auto& output : output_names_) {
    ndll_proto::InputOutput *out = pipe.add_pipe_outputs();

    out->set_name(output.first);
    out->set_device(output.second);
  }

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

}  // namespace ndll
