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

#include "dali/pipeline/operator/op_spec.h"

#include "dali/pipeline/data/types.h"

namespace dali {

OpSpec& OpSpec::AddInput(const string &name, const string &device, bool regular_input) {
  DALI_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for input \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  if (regular_input) {
    // We rely on the fact that regular inputs are first in inputs_ vector
    DALI_ENFORCE(argument_inputs_indexes_.empty(),
        "All regular inputs (particularly, `" + name + "`) need to be added to the op `" +
        this->name() + "` before argument inputs.");
  }

  inputs_.push_back({name, device});
  return *this;
}

OpSpec& OpSpec::AddOutput(const string &name, const string &device) {
  DALI_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for output \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  InOutDeviceDesc name_device_pair = {name, device};
  DALI_ENFORCE(output_name_idx_.count(name_device_pair) == 0,
      "Output '" + name + "' with device '" + device + "' "
      "already added to OpSpec");

  outputs_.push_back({name, device});
  auto ret = output_name_idx_.insert({name_device_pair, outputs_.size()-1});
  DALI_ENFORCE(ret.second, "Output name/device insertion failed.");
  return *this;
}

OpSpec& OpSpec::AddArgumentInput(const string &arg_name, const string &inp_name) {
  DALI_ENFORCE(!this->HasArgument(arg_name), make_string(
      "Argument ", arg_name, " is already specified."));
  const OpSchema& schema = GetSchema();
  DALI_ENFORCE(schema.HasArgument(arg_name), make_string(
      "Argument '", arg_name, "' is not part of the op schema '", schema.name(), "'"));
  DALI_ENFORCE(schema.IsTensorArgument(arg_name), make_string(
      "Argument `", arg_name, "` in operator `", schema.name(), "` is not a a tensor argument."));
  argument_inputs_[arg_name] = inputs_.size();
  argument_inputs_indexes_.insert(inputs_.size());
  AddInput(inp_name, "cpu", false);
  return *this;
}

}  // namespace dali
