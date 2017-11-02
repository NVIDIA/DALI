#include "ndll/pipeline/data/types.h"

#include <map>

namespace ndll {
std::mutex TypeTable::mutex_;
int TypeTable::id_ = 0;
std::unordered_map<std::type_index, TypeID> TypeTable::type_map_;

// Instantiate some basic types
NDLL_REGISTER_TYPE(NoType);
NDLL_REGISTER_TYPE(uint8);
NDLL_REGISTER_TYPE(int16);
NDLL_REGISTER_TYPE(int);
NDLL_REGISTER_TYPE(long);
NDLL_REGISTER_TYPE(long long);
NDLL_REGISTER_TYPE(float16);
NDLL_REGISTER_TYPE(float);
NDLL_REGISTER_TYPE(double);

} // namespace ndll
