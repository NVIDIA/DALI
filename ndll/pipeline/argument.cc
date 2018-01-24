// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/argument.h"

#ifdef NDLL_BUILD_PROTO3
#include "ndll/pipeline/operators/reader/parser/tf_feature.h"
#endif  // NDLL_BUILD_PROTO3

namespace ndll {


template <typename T>
void ArgumentInst<T>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  NDLL_FAIL("Default ArgumentInst::SerializeToProtobuf should never be called\n");
}

#define SERIALIZE_ARGUMENT(type, field)                                           \
template <>                                                                       \
inline void ArgumentInst<type>::SerializeToProtobuf(ndll_proto::Argument *arg) {  \
  arg->set_name(Argument::ToString());                                            \
  arg->set_type(#type);                                                         \
  arg->set_is_vector(false);                                                      \
  arg->set_##field(this->Get());                                                  \
}

#define SERIALIZE_VECTOR_ARGUMENT(type, field)                                                \
template <>                                                                                   \
inline void ArgumentInst<std::vector<type>>::SerializeToProtobuf(ndll_proto::Argument *arg) { \
  arg->set_name(Argument::ToString());                                                        \
  arg->set_type(#type);                                                                    \
  arg->set_is_vector(true);                                                                   \
  auto vec = this->Get();                                                                     \
  for (size_t i = 0; i < vec.size(); ++i) {                                                   \
    arg->add_##field(vec[i]);                                                                 \
  }                                                                                           \
}

SERIALIZE_ARGUMENT(int64_t, i);
SERIALIZE_ARGUMENT(float, f);
SERIALIZE_ARGUMENT(bool, b);
SERIALIZE_ARGUMENT(string, s);


SERIALIZE_VECTOR_ARGUMENT(int64_t, ints);
SERIALIZE_VECTOR_ARGUMENT(float, floats);
SERIALIZE_VECTOR_ARGUMENT(bool, bools);
SERIALIZE_VECTOR_ARGUMENT(string, strings);

#undef SERIALIZE_ARGUMENT
#undef SERIALIZE_VECTOR_ARGUMENT

template <typename T>
inline Argument *DeserializeProtobufImpl(const ndll_proto::Argument&) {
  NDLL_FAIL("Base DeserializeProtobufImpl should never be called");
  return nullptr;
}

template <typename T>
inline Argument *DeserializeProtobufVectorImpl(const ndll_proto::Argument& arg) {
  NDLL_FAIL("Base DeserializeProtobufVectorImpl should never be called");
}

#define DESERIALIZE_PROTOBUF(type, field)                                         \
template <>                                                                       \
inline Argument *DeserializeProtobufImpl<type>(const ndll_proto::Argument& arg) { \
  Argument* new_arg = Argument::Store(arg.name(), arg.field());                   \
  return new_arg;                                                                 \
}

#define DESERIALIZE_VECTOR_PROTOBUF(type, field)                                          \
template <>                                                                               \
inline Argument *DeserializeProtobufVectorImpl<type>(                                     \
    const ndll_proto::Argument& arg) {                                                    \
  auto& f = arg.field();                                                                  \
  Argument* new_arg =                                                                     \
      Argument::Store(arg.name(), vector<type>(f.begin(), f.end()));                      \
  return new_arg;                                                                         \
}

DESERIALIZE_PROTOBUF(int64_t, i);
DESERIALIZE_PROTOBUF(float, f);
DESERIALIZE_PROTOBUF(bool, b);
DESERIALIZE_PROTOBUF(string, s);

DESERIALIZE_VECTOR_PROTOBUF(int64_t, ints);
DESERIALIZE_VECTOR_PROTOBUF(float, floats);
DESERIALIZE_VECTOR_PROTOBUF(bool, bools);
DESERIALIZE_VECTOR_PROTOBUF(string, strings);

#undef DESERIALIZE_PROTOBUF
#undef DESERIALIZE_VECTOR_PROTOBUF

#ifdef NDLL_BUILD_PROTO3
template <>
inline Argument *DeserializeProtobufImpl<TFUtil::Feature>(const ndll_proto::Argument& arg) {
  // type argument
  ndll_proto::Argument type_arg = arg.extra_args(0);
  TFUtil::FeatureType type = static_cast<TFUtil::FeatureType>(type_arg.i());

  // shape
  ndll_proto::Argument shape_arg = arg.extra_args(1);
  std::vector<Index> shape{shape_arg.ints().begin(), shape_arg.ints().end()};

  // default value
  ndll_proto::Argument value_arg = arg.extra_args(2);
  TFUtil::Feature::Value val;
  if (type == TFUtil::FeatureType::int64) {
    val.int64 = value_arg.i();
  } else if (type == TFUtil::FeatureType::string) {
    val.str = value_arg.s();
  } else if (type == TFUtil::FeatureType::float32) {
    val.float32 = value_arg.f();
  } else {
    NDLL_FAIL("Unknown TFUtil::FeatureType value");
  }

  TFUtil::Feature feature(shape, type, val);

  Argument* new_arg = Argument::Store(arg.name(), feature);
  return new_arg;
}
#endif

Argument *DeserializeProtobuf(const ndll_proto::Argument& arg) {
  // map
  std::map<std::pair<string, bool>, std::function<Argument*(const ndll_proto::Argument&)>> fn_map{
    {{"int", false}, DeserializeProtobufImpl<int64_t>},
    {{"int", true}, DeserializeProtobufVectorImpl<int64_t>},
    {{"float", false}, DeserializeProtobufImpl<float>},
    {{"float", true}, DeserializeProtobufVectorImpl<float>},
    {{"string", false}, DeserializeProtobufImpl<string>},
    {{"string", true}, DeserializeProtobufVectorImpl<string>},
    {{"bool", false}, DeserializeProtobufImpl<bool>},
    {{"bool", true}, DeserializeProtobufVectorImpl<bool>},
  };
#ifdef NDLL_BUILD_PROTO3
  fn_map[{"TFRecord", false}] = DeserializeProtobufImpl<TFUtil::Feature>;
#endif

  auto it = fn_map.find({arg.type(), arg.is_vector()});
  NDLL_ENFORCE(it != fn_map.end(), "Invalid argument \"type\" in protobuf");

  return fn_map[{arg.type(), arg.is_vector()}](arg);
}


#ifdef NDLL_BUILD_PROTO3
template <>
inline void ArgumentInst<TFUtil::Feature>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_type("TFFeature");

  TFUtil::Feature self = this->Get();
  // set the datatype of the record
  auto *type_arg = arg->add_extra_args();
  type_arg->set_name("type");
  type_arg->set_i(static_cast<int>(self.GetType()));

  // set the shape
  auto *shape_arg = arg->add_extra_args();
  shape_arg->set_name("shape");
  auto& shape = self.Shape();
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_arg->set_ints(i, shape[i]);
  }

  // set the default value
  auto *default_arg = arg->add_extra_args();
  default_arg->set_name("default_value");
  if (self.GetType() == TFUtil::FeatureType::int64) {
    default_arg->set_i(self.GetValue().int64);
  } else if (self.GetType() == TFUtil::FeatureType::string) {
    default_arg->set_s(self.GetValue().str);
  } else if (self.GetType() == TFUtil::FeatureType::float32) {
    default_arg->set_f(self.GetValue().float32);
  } else {
    NDLL_FAIL("Unknown TFUtil::FeatureType value");
  }
}
#endif  // NDLL_BUILD_PROTO3

}  // namespace ndll
