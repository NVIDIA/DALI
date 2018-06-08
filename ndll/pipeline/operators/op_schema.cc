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
  for (auto& arg_pair : arguments_) {
    req_arguments_left.insert(arg_pair.first);
  }
  for (std::string s : vec) {
    NDLL_ENFORCE(arguments_.find(s) != arguments_.end() ||
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

}  // namespace ndll
