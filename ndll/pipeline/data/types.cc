#include "ndll/pipeline/data/types.h"

#include <map>

namespace ndll {
std::mutex TypeTable::mutex_;
int TypeTable::id_ = 0;
std::unordered_map<std::type_index, TypeID> TypeTable::type_map_;

// Instantiate some basic types
DEFINE_TYPE(uint8);
DEFINE_TYPE(int16);
DEFINE_TYPE(int);
DEFINE_TYPE(long);
DEFINE_TYPE(long long);
DEFINE_TYPE(float16);
DEFINE_TYPE(float);
DEFINE_TYPE(double);

NDLLDataType NDLLTypeForMeta(TypeMeta type_meta) {
  static const std::map<TypeID, NDLLDataType> type_map = {
    {TypeTable::GetTypeID<uint8>(), NDLL_UINT8},
    {TypeTable::GetTypeID<float16>(), NDLL_FLOAT16},
    {TypeTable::GetTypeID<float>(), NDLL_FLOAT},
  };

  auto it = type_map.find(type_meta.id());
  return (it == type_map.end()) ? NDLL_NO_TYPE : it->second;
}

TypeMeta NDLLMetaForType(NDLLDataType type) {
  TypeMeta type_meta;
  switch (type) {
  case NDLL_UINT8:
    type_meta.SetType<uint8>();
    break;
  case NDLL_FLOAT16:
    type_meta.SetType<float16>();
    break;
  case NDLL_FLOAT:
    type_meta.SetType<float>();
    break;
  default:
    break;
  };
  return type_meta;
}

} // namespace ndll
