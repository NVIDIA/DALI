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

#include "dali/pipeline/operators/op_schema.h"
#include "dali/pipeline/operators/op_spec.h"

#include <string>

namespace dali {

std::map<string, OpSchema>& SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

int OpSchema::CalculateOutputs(const OpSpec &spec) const {
  int num_input_sets = 1;
  if (allow_multiple_input_sets_) {
    num_input_sets = spec.GetArgument<int>("num_input_sets");
  }

  if (!output_fn_) {
    return num_input_sets * num_output_;
  } else {
    return num_input_sets * output_fn_(spec);
  }
}

void OpSchema::CheckArgs(const OpSpec &spec) const {
  std::vector<string> vec = spec.ListArguments();
  std::set<std::string> req_arguments_left;
  auto required_arguments = GetRequiredArguments();
  for (auto& arg_pair : required_arguments) {
    req_arguments_left.insert(arg_pair.first);
  }
  for (std::string s : vec) {
    DALI_ENFORCE(HasArgument(s) ||
        internal_arguments_.find(s) != internal_arguments_.end(),
        "Got an unexpected argument \"" + s + "\"");
    std::set<std::string>::iterator it = req_arguments_left.find(s);
    if (it != req_arguments_left.end()) {
      req_arguments_left.erase(it);
    }
  }
  if (!req_arguments_left.empty()) {
    std::string ret = "Not all required arguments were specified for op \""
      + this->name() + "\". Please specify values for arguments: ";
    for (auto& str : req_arguments_left) {
      ret += "\"" + str + "\", ";
    }
    ret.erase(ret.size()-2);
    ret += ".";
    DALI_FAIL(ret);
  }
}

string OpSchema::Dox() const {
  return dox_;
}

std::map<std::string, std::pair<std::string, DALIDataType> >
OpSchema::GetRequiredArguments() const {
  auto ret = arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetRequiredArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}

std::map<std::string, std::pair<std::string, Value*>> OpSchema::GetOptionalArguments() const {
  auto ret = optional_arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetOptionalArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}

std::string OpSchema::GetArgumentDox(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name), "Argument \"" + name +
      "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).first;
  } else {
    // optional argument
    return GetOptionalArguments().at(name).first;
  }
}

DALIDataType OpSchema::GetArgumentType(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name), "Argument \"" + name +
      "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).second;
  } else {
    // optional argument
    return GetOptionalArguments().at(name).second->GetTypeID();
  }
}

std::string OpSchema::GetArgumentDefaultValueString(const std::string &name) const {
  DALI_ENFORCE(HasOptionalArgument(name), "Argument \"" + name +
      "\" is either not supported by operator \"" + this->name() + "\" or is not optional.");
  return GetOptionalArguments().at(name).second->ToString();
}

std::vector<std::string> OpSchema::GetArgumentNames() const {
  std::vector<std::string> ret;
  for (auto &arg_pair : GetRequiredArguments()) {
    ret.push_back(arg_pair.first);
  }
  for (auto &arg_pair : GetOptionalArguments()) {
    ret.push_back(arg_pair.first);
  }
  return ret;
}

bool OpSchema::HasRequiredArgument(const std::string &name, const bool local_only) const {
  bool ret = arguments_.find(name) != arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasRequiredArgument(name);
  }
  return ret;
}

bool OpSchema::HasOptionalArgument(const std::string &name, const bool local_only) const {
  bool ret = optional_arguments_.find(name) != optional_arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasOptionalArgument(name);
  }
  return ret;
}

bool OpSchema::IsTensorArgument(const std::string &name) const {
  bool ret = tensor_arguments_.find(name) != tensor_arguments_.end();
  if (ret) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.IsTensorArgument(name);
  }
  return ret;
}

}  // namespace dali
