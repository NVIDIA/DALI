#include "ndll/pipeline/op_schema.h"

namespace ndll {

std::map<string, OpSchema>& SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

} // namespace ndll
