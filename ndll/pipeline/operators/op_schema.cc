// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/op_schema.h"
#include "ndll/pipeline/operators/op_spec.h"

#include <string>

namespace ndll {

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
    NDLL_ENFORCE(required_arguments.find(s) != required_arguments.end() ||
        OptionalArgumentExists(s) ||
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
    NDLL_FAIL(ret);
  }
}

string OpSchema::Dox() const {
  std::string ret = "# " + Name();
  ret += "\n\nOverview\n--------\n";
  ret += dox_;
  ret += "\n\nRequired Parameters\n-------------------\n";
  for (auto arg_pair : GetRequiredArguments()) {
    ret += " - `" + arg_pair.first + "` : " + arg_pair.second + "\n";
  }
  ret += "\n\nOptional Parameters\n-------------------\n";
  for (auto arg_pair : GetOptionalArguments()) {
    ret += " - `" + arg_pair.first + "` : " + arg_pair.second.first + "\n";
  }
  return ret;
}

std::map<std::string, std::string> OpSchema::GetRequiredArguments() const {
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

}  // namespace ndll
