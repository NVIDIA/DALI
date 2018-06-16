// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_
#define DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_

#include <string>

#include "dali/pipeline/dali.pb.h"
#include "dali/common.h"

namespace dali {

inline std::string serialize_type(const int64&) {
  return "int64";
}

inline std::string serialize_type(const bool&) {
  return "bool";
}

inline std::string serialize_type(const std::string&) {
  return "string";
}

inline std::string serialize_type(const float&) {
  return "float";
}

template<typename T>
inline auto serialize_type(const T& t)
  -> decltype(t.SerializeType()) {
  return t.SerializeType();
}

#define SERIALIZE_ARGUMENT(type, field)                               \
inline dali_proto::Argument * SerializeToProtobuf(const type& t, dali_proto::Argument *arg) {  \
  arg->set_type(serialize_type(t));                                   \
  arg->set_is_vector(false);                                          \
  arg->add_##field(t);                                      \
  return arg;                                                         \
}

#define SERIALIZE_ARGUMENT_AS_INT64(type) \
inline dali_proto::Argument * SerializeToProtobuf(const type& t, dali_proto::Argument *arg) { \
  return dali::SerializeToProtobuf(static_cast<int64>(t), arg); \
}

SERIALIZE_ARGUMENT(int64, ints);
SERIALIZE_ARGUMENT(float, floats);
SERIALIZE_ARGUMENT(bool, bools);
SERIALIZE_ARGUMENT(string, strings);

SERIALIZE_ARGUMENT_AS_INT64(int);
SERIALIZE_ARGUMENT_AS_INT64(unsigned int);
SERIALIZE_ARGUMENT_AS_INT64(uint64);

template<typename T>
inline auto SerializeToProtobuf(const T& t, dali_proto::Argument *arg)
  -> decltype(t.SerializeToProtobuf(arg)) {
    return t.SerializeToProtobuf(arg);
}

#undef SERIALIZE_ARGUMENT
}  // namespace dali

#endif  // DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_
