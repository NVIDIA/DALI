// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/op_schema.h"

#include <string>

namespace ndll {

std::map<string, OpSchema>& SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

}  // namespace ndll
