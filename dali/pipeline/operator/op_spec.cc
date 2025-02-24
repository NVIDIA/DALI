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

#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/name_utils.h"

namespace dali {

inline bool IsCompatibleDevice(StorageDevice provided, InputDevice required, OpType op_device) {
  switch (required) {
  case InputDevice::CPU:
    return provided == StorageDevice::CPU;
  case InputDevice::GPU:
    return provided == StorageDevice::GPU;
  case InputDevice::MatchBackend:
    return op_device == OpType::GPU
      ? provided == StorageDevice::GPU
      : provided == StorageDevice::CPU;
  case InputDevice::MatchBackendOrCPU:
    return op_device == OpType::CPU ? provided == StorageDevice::CPU : true;
  case InputDevice::Any:
  case InputDevice::Metadata:
    return true;
  default:
    return false;
  }
}

inline std::string ValidDevices(InputDevice required, OpType op_device) {
  switch (required) {
    case InputDevice::CPU:
      return "\"cpu\"";
    case InputDevice::GPU:
      return "\"gpu\"";
    case InputDevice::MatchBackend:
      return op_device == OpType::GPU ? "\"gpu\"" : "\"cpu\"";
    case InputDevice::MatchBackendOrCPU:
      return op_device == OpType::GPU ? "\"gpu\" or \"cpu\"" : "\"cpu\"";
    case InputDevice::Any:
    case InputDevice::Metadata:
      return "\"gpu\" or \"cpu\"";
    default:
      assert(!"Unrechable");
      return "<invalid>";
  }
}

OpSpec &OpSpec::AddInput(std::string name, StorageDevice device, bool regular_input) {
  if (regular_input) {
    // We rely on the fact that regular inputs are first in inputs_ vector
    DALI_ENFORCE(NumArgumentInput() == 0,
        "All regular inputs (particularly, \"" + name + "\") need to be added to the op `" +
        GetOpDisplayName(*this, true) + "` before argument inputs.");

    if (schema_) {
      int idx = inputs_.size();
      DALI_ENFORCE(idx < schema_->MaxNumInput(), make_string(
        "The operator `", GetOpDisplayName(*this, true), "` takes up to ", schema_->MaxNumInput(),
        " inputs. The input \"", name , "\" is out of range."));

      if (HasArgument("device")) {
        auto op_type_str = GetArgument<std::string>("device");
        OpType op_type = ParseOpType(op_type_str);
        auto inp_dev = schema_->GetInputDevice(idx);
        DALI_ENFORCE(IsCompatibleDevice(device, inp_dev, op_type),
          make_string("The input ", idx, " for ", op_type_str, " operator `",
          GetOpDisplayName(*this, true), "` is stored on incompatible device \"", device,
          "\". Valid device is ", ValidDevices(inp_dev, op_type), "."));
      }
    }
  } else {
    DALI_ENFORCE(device == StorageDevice::CPU, make_string("Invalid storage device \"", device,
      "\" for a named input \"", name, "\". All named inputs must be on CPU."));
  }

  inputs_.push_back({std::move(name), device});
  return *this;
}

OpSpec &OpSpec::AddOutput(std::string name, StorageDevice device) {
  auto [it, inserted] = output_name_idx_.emplace(
    InOutDeviceDesc{std::string(name), device},
    outputs_.size());
  if (!inserted)
    throw std::invalid_argument(make_string("Duplicate output name and device: \"",
                                            TensorName(name, device), "\""));

  outputs_.push_back({std::move(name), device});
  return *this;
}

OpSpec &OpSpec::AddArgumentInput(std::string arg_name, std::string inp_name) {
  DALI_ENFORCE(!this->HasArgument(arg_name), make_string(
      "Argument '", arg_name, "' is already specified."));
  const OpSchema &schema = GetSchemaOrDefault();
  DALI_ENFORCE(schema.HasArgument(arg_name),
               make_string("Argument '", arg_name, "' is not defined for operator `",
                           GetOpDisplayName(*this, true), "`."));
  DALI_ENFORCE(schema.IsTensorArgument(arg_name),
               make_string("Argument '", arg_name, "' in operator `", GetOpDisplayName(*this, true),
                           "` is not an argument input."));
  int idx = inputs_.size();
  argument_input_idxs_[arg_name] = idx;
  argument_inputs_.push_back({ std::move(arg_name), idx });
  AddInput(std::move(inp_name), StorageDevice::CPU, false);
  return *this;
}

OpSpec &OpSpec::SetInitializedArg(std::string_view arg_name, std::shared_ptr<Argument> arg) {
  if (schema_ && schema_->IsDeprecatedArg(arg_name)) {
    const auto &deprecation_meta = schema_->DeprecatedArgInfo(arg_name);
    // Argument was removed, and we can discard it
    if (deprecation_meta.removed) {
      return *this;
    }
    if (!deprecation_meta.renamed_to.empty()) {
      const auto &new_arg_name = deprecation_meta.renamed_to;
      DALI_ENFORCE(argument_idxs_.find(new_arg_name) == argument_idxs_.end(),
                   make_string("Operator `", GetOpDisplayName(*this, true), "` got an unexpected '",
                               arg_name, "' deprecated argument when '", new_arg_name,
                               "' was already provided."));

      set_through_deprecated_arguments_[new_arg_name] = arg_name;
      // Adjust the arg so it carries the proper name for serialization
      if (arg->has_name()) {
        arg->set_name(new_arg_name);
      }
      auto [it, inserted] = argument_idxs_.emplace(new_arg_name, arguments_.size());
      if (inserted)
        arguments_.push_back(std::move(arg));
      else
        arguments_[it->second] = std::move(arg);
      return *this;
    }
  }
  EnforceNoAliasWithDeprecated(arg_name);
  auto [it, inserted] = argument_idxs_.emplace(arg_name, arguments_.size());
  if (inserted)
    arguments_.push_back(std::move(arg));
  else
    arguments_[it->second] = std::move(arg);
  return *this;
}

void OpSpec::EnforceNoAliasWithDeprecated(std::string_view arg_name) {
  auto set_through = set_through_deprecated_arguments_.find(arg_name);
  DALI_ENFORCE(set_through == set_through_deprecated_arguments_.end(),
               make_string("Operator `", GetOpDisplayName(*this, true), "` got an unexpected '",
                           set_through->second, "' deprecated argument when '", arg_name,
                           "' was already provided."));
}


}  // namespace dali
