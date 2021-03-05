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

#include "dali/pipeline/operator/op_schema.h"

#include <string>
#include "dali/pipeline/operator/op_spec.h"
#include "dali/core/python_util.h"

namespace dali {

std::map<string, OpSchema>& SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

OpSchema& SchemaRegistry::RegisterSchema(const std::string &name) {
  auto &schema_map = registry();
  DALI_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
      "registered for operator '" + name + "'. DALI_SCHEMA(op) "
      "should only be called once per op.");

  // Insert the op schema and return a reference to it
  schema_map.emplace(std::make_pair(name, OpSchema(name)));
  return schema_map.at(name);
}

const OpSchema& SchemaRegistry::GetSchema(const std::string &name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  DALI_ENFORCE(it != schema_map.end(), "Schema for operator '" +
      name + "' not registered");
  return it->second;
}

const OpSchema* SchemaRegistry::TryGetSchema(const std::string &name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  return it != schema_map.end() ? &it->second : nullptr;
}

int OpSchema::CalculateOutputs(const OpSpec &spec) const {
  if (!output_fn_) {
    return num_output_;
  } else {
    return output_fn_(spec);
  }
}

void OpSchema::CheckArgs(const OpSpec &spec) const {
  std::vector<string> vec = spec.ListArguments();
  std::set<std::string> req_arguments_left;
  auto required_arguments = GetRequiredArguments();
  for (auto& arg_pair : required_arguments) {
    req_arguments_left.insert(arg_pair.first);
  }
  for (const auto &s : vec) {
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

std::map<std::string, RequiredArgumentDef>
OpSchema::GetRequiredArguments() const {
  auto ret = arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetRequiredArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}

std::map<std::string, DefaultedArgumentDef> OpSchema::GetOptionalArguments() const {
  auto ret = optional_arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetOptionalArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}

std::map<std::string, DeprecatedArgDef> OpSchema::GetDeprecatedArguments() const {
  auto ret = deprecated_arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetDeprecatedArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}

DLL_PUBLIC bool OpSchema::IsDeprecatedArg(const std::string &arg_name) const {
  if (deprecated_arguments_.find(arg_name) != deprecated_arguments_.end())
    return true;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    if (parent.IsDeprecatedArg(arg_name))
      return true;
  }
  return false;
}

DLL_PUBLIC const DeprecatedArgDef &OpSchema::DeprecatedArgMeta(const std::string &arg_name) const {
  auto it = deprecated_arguments_.find(arg_name);
  if (it != deprecated_arguments_.end()) {
    return it->second;
  }
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    if (parent.IsDeprecatedArg(arg_name))
      return parent.DeprecatedArgMeta(arg_name);
  }
  DALI_FAIL(make_string("No deprecation metadata for argument \"", arg_name, "\" found."));
}

std::string OpSchema::GetArgumentDox(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name), "Argument \"" + name +
      "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).doc;
  } else {
    return GetOptionalArguments().at(name).doc;
  }
}

DALIDataType OpSchema::GetArgumentType(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name), "Argument \"" + name +
      "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).dtype;
  } else {
    return GetOptionalArguments().at(name).dtype;
  }
}

bool OpSchema::HasArgumentDefaultValue(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name, true), "Argument \"" + name +
      "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return false;
  }
  if (HasInternalArgument(name, true)) {
    return true;
  }
  auto *value_ptr = GetOptionalArguments().at(name).default_value;
  return value_ptr != nullptr;
}

std::string OpSchema::GetArgumentDefaultValueString(const std::string &name) const {
  DALI_ENFORCE(HasOptionalArgument(name), "Argument \"" + name +
      "\" is either not supported by operator \"" + this->name() + "\" or is not optional.");

  auto *value_ptr = GetOptionalArguments().at(name).default_value;

  DALI_ENFORCE(value_ptr, make_string("Argument \"", name,
      "\" in operator \"" + this->name() + "\" has no default value."));

  auto &val = *value_ptr;
  auto str = val.ToString();
  if (val.GetTypeID() == DALI_STRING ||
      val.GetTypeID() == DALI_TENSOR_LAYOUT)
    return python_repr(str);
  else
    return str;
}

std::vector<std::string> OpSchema::GetArgumentNames() const {
  std::vector<std::string> ret;
  const auto& required   = GetRequiredArguments();
  const auto& optional   = GetOptionalArguments();
  const auto& deprecated = GetDeprecatedArguments();
  for (auto &arg_pair : required) {
    ret.push_back(arg_pair.first);
  }
  for (auto &arg_pair : optional) {
    ret.push_back(arg_pair.first);
  }
  for (auto &arg_pair : deprecated) {
    // Deprecated aliases only appear in `deprecated` but regular
    // deprecated arguments appear both in `deprecated` and either `required` or `optional`.
    if (required.find(arg_pair.first) == required.end() &&
        optional.find(arg_pair.first) == optional.end())
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

bool OpSchema::HasInternalArgument(const std::string &name, const bool local_only) const {
  bool ret = internal_arguments_.find(name) != internal_arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasInternalArgument(name);
  }
  return ret;
}

std::pair<const OpSchema *, const Value *>
OpSchema::FindDefaultValue(const std::string &name, bool local_only, bool include_internal) const {
  auto it = optional_arguments_.find(name);
  if (it != optional_arguments_.end()) {
    return { this, it->second.default_value };
  }
  if (include_internal) {
    it = internal_arguments_.find(name);
    if (it != internal_arguments_.end()) {
      return { this, it->second.default_value };
    }
  }
  if (local_only)
    return { nullptr, nullptr };

  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    auto schema_val = parent.FindDefaultValue(name, false, include_internal);
    if (schema_val.first && schema_val.second)
      return schema_val;
  }
  return { nullptr, nullptr };
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
