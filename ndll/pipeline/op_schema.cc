// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/op_schema.h"
#include "ndll/pipeline/op_spec.h"

#include <string>

namespace ndll {

std::map<string, OpSchema>& SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

int OpSchema::CalculateOutputs(const OpSpec &spec) const {
  int num_input_sets = 1;
  if (allow_multiple_input_sets_) {
    if (MinNumInput() == MaxNumInput()) {
      num_input_sets = spec.NumInput() / MinNumInput();
    } else {
      num_input_sets = spec.GetArgument<int>("num_input_sets");
    }
  }

  if (!output_fn_) {
    return num_input_sets * num_output_;
  } else {
    return num_input_sets * output_fn_(spec);
  }
}

}  // namespace ndll
