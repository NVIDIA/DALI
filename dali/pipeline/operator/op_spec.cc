// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/operator/name_utils.h"

namespace dali {

OpSpec& OpSpec::AddInput(const string &name, const string &device, bool regular_input) {
  DALI_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for input \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  if (regular_input) {
    // We rely on the fact that regular inputs are first in inputs_ vector
    DALI_ENFORCE(NumArgumentInput() == 0,
        "All regular inputs (particularly, `" + name + "`) need to be added to the op `" +
        GetOpDisplayName(*this, true) + "` before argument inputs.");
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
      "Argument '", arg_name, "' is already specified."));
  const OpSchema& schema = GetSchema();
  DALI_ENFORCE(schema.HasArgument(arg_name),
               make_string("Argument '", arg_name, "' is not supported by operator `",
                           GetOpDisplayName(*this, true), "`."));
  DALI_ENFORCE(schema.IsTensorArgument(arg_name),
               make_string("Argument '", arg_name, "' in operator `", GetOpDisplayName(*this, true),
                           "` is not an argument input."));
  int idx = inputs_.size();
  argument_inputs_.push_back({ arg_name, idx });
  argument_input_idxs_[arg_name] = idx;
  AddInput(inp_name, "cpu", false);
  return *this;
}

OpSpec& OpSpec::SetInitializedArg(const string& arg_name, std::shared_ptr<Argument> arg) {
  if (schema_ && schema_->IsDeprecatedArg(arg_name)) {
    const auto& deprecation_meta = schema_->DeprecatedArgMeta(arg_name);
    // Argument was removed, and we can discard it
    if (deprecation_meta.removed) {
      return *this;
    }
    if (!deprecation_meta.renamed_to.empty()) {
      const auto& new_arg_name = deprecation_meta.renamed_to;
      DALI_ENFORCE(argument_idxs_.find(new_arg_name) == argument_idxs_.end(),
                   make_string("Operator `", GetOpDisplayName(*this, true), "` got an unexpected '",
                               arg_name, "' deprecated argument when '", new_arg_name,
                               "' was already provided."));

      set_through_deprecated_arguments_[new_arg_name] = arg_name;
      // Adjust the arg so it carries the proper name for serialization
      if (arg->has_name()) {
        arg->set_name(new_arg_name);
      }
      auto [it, inserted] = argument_idxs_.insert({new_arg_name, arguments_.size()});
      if (inserted)
        arguments_.push_back(std::move(arg));
      else
        arguments_[it->second] = std::move(arg);
      return *this;
    }
  }
  EnforceNoAliasWithDeprecated(arg_name);
  auto [it, inserted] = argument_idxs_.insert({arg_name, arguments_.size()});
  if (inserted)
    arguments_.push_back(std::move(arg));
  else
    arguments_[it->second] = std::move(arg);
  return *this;
}

void OpSpec::EnforceNoAliasWithDeprecated(const string& arg_name) {
  auto set_through = set_through_deprecated_arguments_.find(arg_name);
  DALI_ENFORCE(set_through == set_through_deprecated_arguments_.end(),
               make_string("Operator `", GetOpDisplayName(*this, true), "` got an unexpected '",
                           set_through->second, "' deprecated argument when '", arg_name,
                           "' was already provided."));
}


}  // namespace dali
