// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/op_spec.h"

#include "ndll/pipeline/data/types.h"

namespace ndll {

OpSpec& OpSpec::AddInput(const string &name, const string &device, bool regular_input) {
  NDLL_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for input \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  if (regular_input) {
    // We rely on the fact that regular inputs are first in inputs_ vector
    NDLL_ENFORCE(argument_inputs_indexes_.empty(),
        "All regular inputs need to be added to the op before argument inputs.");
  }
  StrPair name_device_pair = std::make_pair(name, device);

  inputs_.push_back(std::make_pair(name, device));
  return *this;
}

OpSpec& OpSpec::AddOutput(const string &name, const string &device) {
  NDLL_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for output \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  StrPair name_device_pair = std::make_pair(name, device);
  NDLL_ENFORCE(output_name_idx_.count(name_device_pair) == 0,
      "Output '" + name + "' with device '" + device + "' "
      "already added to OpSpec");

  outputs_.push_back(std::make_pair(name, device));
  auto ret = output_name_idx_.insert({name_device_pair, outputs_.size()-1});
  NDLL_ENFORCE(ret.second, "Output name/device insertion failed.");
  return *this;
}

OpSpec& OpSpec::AddArgumentInput(const string &arg_name, const string &inp_name) {
  NDLL_ENFORCE(!this->HasArgument(arg_name),
      "Argument " + arg_name + " was already added to the op.");
  const OpSchema& schema = SchemaRegistry::GetSchema(this->name());
  NDLL_ENFORCE(schema.HasArgument(arg_name),
      "Argument " + arg_name + " is not part of the op schema");
  argument_inputs_[arg_name] = inputs_.size();
  argument_inputs_indexes_.insert(inputs_.size());
  AddInput(inp_name, "cpu", false);
  return *this;
}

}  // namespace ndll
