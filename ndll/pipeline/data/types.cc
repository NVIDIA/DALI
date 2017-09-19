#include "ndll/pipeline/data/types.h"

namespace ndll {
std::unordered_map<std::type_index, TypeID> TypeTable::type_map_;

// Instantiate some basic types
DEFINE_TYPE(uint8);
DEFINE_TYPE(int16);
DEFINE_TYPE(int);
DEFINE_TYPE(float16);
DEFINE_TYPE(float);
DEFINE_TYPE(double);

} // namespace ndll
