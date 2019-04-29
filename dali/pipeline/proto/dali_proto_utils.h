// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_
#define DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_

#include <string>

#include "dali/core/common.h"
#include "dali/pipeline/proto/dali_proto_intern.h"

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
inline DaliProtoPriv * SerializeToProtobuf(const type& t, DaliProtoPriv *arg) {  \
  arg->set_type(serialize_type(t));                                   \
  arg->set_is_vector(false);                                          \
  arg->add_##field(t);                                      \
  return arg;                                                         \
}

#define SERIALIZE_ARGUMENT_AS_INT64(type) \
inline DaliProtoPriv * SerializeToProtobuf(const type& t, DaliProtoPriv *arg) { \
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
inline auto SerializeToProtobuf(const T& t, DaliProtoPriv *arg)
  -> decltype(t.SerializeToProtobuf(arg)) {
    return t.SerializeToProtobuf(arg);
}

#undef SERIALIZE_ARGUMENT
}  // namespace dali

#endif  // DALI_PIPELINE_PROTO_DALI_PROTO_UTILS_H_
