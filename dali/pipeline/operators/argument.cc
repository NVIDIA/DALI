// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/operators/reader/parser/tf_feature.h"

namespace dali {

template<typename T>
inline Argument * DeserializeProtobufImpl(const dali_proto::Argument& arg) {
  const T& t = T::DeserializeFromProtobuf(arg);
  return Argument::Store(arg.name(), t);
}

#define DESERIALIZE_PROTOBUF(type, field)                                         \
template <>                                                                       \
inline Argument *DeserializeProtobufImpl<type>(const dali_proto::Argument& arg) { \
  Argument* new_arg = Argument::Store(arg.name(), arg.field(0));                   \
  return new_arg;                                                                 \
}

DESERIALIZE_PROTOBUF(int64_t, ints);
DESERIALIZE_PROTOBUF(float, floats);
DESERIALIZE_PROTOBUF(bool, bools);
DESERIALIZE_PROTOBUF(string, strings);

#undef DESERIALIZE_PROTOBUF

template<typename T>
inline Argument *DeserializeProtobufVectorImpl(const dali_proto::Argument& arg) {
  auto& args = arg.extra_args();
  std::vector<T> ret_val;
  for (auto& a : args) {
    const T& elem = DeserializeProtobuf(a)->Get<T>();
    ret_val.push_back(elem);
  }
  return Argument::Store(arg.name(), ret_val);
}

#define ADD_SERIALIZABLE_ARG(T)                \
  {{serialize_type(T()), false}, DeserializeProtobufImpl<T>},       \
  {{serialize_type(T()), true},  DeserializeProtobufVectorImpl<T>},

Argument *DeserializeProtobuf(const dali_proto::Argument& arg) {
  // map
  std::map<std::pair<string, bool>, std::function<Argument*(const dali_proto::Argument&)>> fn_map{
    ADD_SERIALIZABLE_ARG(int64)
    ADD_SERIALIZABLE_ARG(float)
    ADD_SERIALIZABLE_ARG(string)
    ADD_SERIALIZABLE_ARG(bool)
#ifdef DALI_BUILD_PROTO3
    ADD_SERIALIZABLE_ARG(TFUtil::Feature)
#endif  // DALI_BUILD_PROTO3
  };

  auto it = fn_map.find({arg.type(), arg.is_vector()});
  DALI_ENFORCE(it != fn_map.end(), "Invalid argument \"type\" in protobuf");

  return fn_map[{arg.type(), arg.is_vector()}](arg);
}

#undef ADD_SERIALIZABLE_ARG
}  // namespace dali
