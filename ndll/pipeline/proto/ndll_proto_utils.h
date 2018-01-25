// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_PROTO_NDLL_PROTO_UTILS_H_
#define NDLL_PIPELINE_PROTO_NDLL_PROTO_UTILS_H_

#include <string>

#include "ndll/pipeline/ndll.pb.h"
#include "ndll/common.h"

namespace ndll {

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
inline ndll_proto::Argument * SerializeToProtobuf(const type& t, ndll_proto::Argument *arg) {  \
  arg->set_type(serialize_type(t));                                   \
  arg->set_is_vector(false);                                          \
  arg->add_##field(t);                                      \
  return arg;                                                         \
}

#define SERIALIZE_ARGUMENT_AS_INT64(type) \
inline ndll_proto::Argument * SerializeToProtobuf(const type& t, ndll_proto::Argument *arg) { \
  return ndll::SerializeToProtobuf(static_cast<int64>(t), arg); \
}

SERIALIZE_ARGUMENT(int64, ints);
SERIALIZE_ARGUMENT(float, floats);
SERIALIZE_ARGUMENT(bool, bools);
SERIALIZE_ARGUMENT(string, strings);

SERIALIZE_ARGUMENT_AS_INT64(int);
SERIALIZE_ARGUMENT_AS_INT64(unsigned int);
SERIALIZE_ARGUMENT_AS_INT64(uint64);

template<typename T>
inline auto SerializeToProtobuf(const T& t, ndll_proto::Argument *arg)
  -> decltype(t.SerializeToProtobuf(arg)) {
    return t.SerializeToProtobuf(arg);
}

#undef SERIALIZE_ARGUMENT
}  // namespace ndll

#endif  // NDLL_PIPELINE_PROTO_NDLL_PROTO_UTILS_H_
