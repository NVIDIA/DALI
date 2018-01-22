// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_ARGUMENT_H_
#define NDLL_PIPELINE_ARGUMENT_H_

#include <vector>
#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/ndll.pb.h"

#ifdef NDLL_BUILD_PROTO3
#include "ndll/pipeline/operators/reader/parser/tfrecord_parser.h"
#endif  // NDLL_BUILD_PROTO3

namespace ndll {

/**
 * @brief Stores a single argument.
 *
 * Argument class is the wrapper parent class
 * for wrapper classes storing arguments
 * given to ops.
 * In order to add a new type of argument,
 * one needs to expose the type to Python
 * in python/ndll_backend.cc file
 * by using py::class_<new_type> and
 * NDLL_OPSPEC_ADDARG macro. For integral
 * types (like enums), one needs to use
 * INSTANTIATE_ARGUMENT_AS_INT64 macro
 * in pipeline/op_spec.h instead.
 */

class Argument {
 public:
  // Setters & getters for name
  inline bool has_name() const { return has_name_; }
  inline const string get_name() const {
    return has_name()?name_:"<no name>";
  }
  inline void set_name(string name) {
    has_name_ = true;
    name_ = name;
  }
  inline void clear_name() {
    has_name_ = false;
    name_ = "";
  }

  virtual std::string ToString() const {
    return get_name();
  }

  virtual void SerializeToProtobuf(ndll_proto::Argument *arg) = 0;

  template<typename T>
  T Get();

  template<typename T>
  static Argument * Store(const std::string& s,
      const T& val);


 protected:
  Argument() :
    has_name_(false)
    {}

  explicit Argument(const std::string& s) :
    name_(s),
    has_name_(true)
    {}

 private:
  std::string name_;
  bool has_name_;
};

template<typename T>
class ArgumentInst : public Argument {
 public:
  explicit ArgumentInst(const std::string& s, const T& v) :
    Argument(s),
    val(v)
    {}

  T Get() { return val; }

  std::string ToString() const override {
    string ret = Argument::ToString();
    ret += ": ";
    ret += to_string(val);
    return ret;
  }

  virtual void SerializeToProtobuf(ndll_proto::Argument *arg) override {
    NDLL_FAIL("Default ArgumentInst::SerializeToProtobuf should never be called\n");
  }

 private:
  T val;
};

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
inline Argument *DeserializeProtobufImpl<TFRecordParser::Feature>(const ndll_proto::Argument& arg) {
  // type argument
  ndll_proto::Argument type_arg = arg.extra_args(0);
  TFRecordParser::FeatureType type = static_cast<TFRecordParser::FeatureType>(type_arg.i());

  // shape
  ndll_proto::Argument shape_arg = arg.extra_args(1);
  std::vector<Index> shape{shape_arg.ints().begin(), shape_arg.ints().end()};

  // default value
  ndll_proto::Argument value_arg = arg.extra_args(2);
  TFRecordParser::Feature::Value val;
  if (type == TFRecordParser::FeatureType::int64) {
    val.int64 = value_arg.i();
  } else if (type == TFRecordParser::FeatureType::string) {
    val.str = value_arg.s();
  } else if (type == TFRecordParser::FeatureType::float32) {
    val.float32 = value_arg.f();
  } else {
    NDLL_FAIL("Unknown TFRecordParser::FeatureType value");
  }

  TFRecordParser::Feature feature(shape, type, val);

  Argument* new_arg = Argument::Store(arg.name(), feature);
  return new_arg;
}
#endif

inline Argument *DeserializeProtobuf(const ndll_proto::Argument& arg) {
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
    {{"TFRecord", false}, DeserializeProtobufImpl<TFRecordParser::Feature>}
  };

  return fn_map[{arg.type(), arg.is_vector()}](arg);
}


#ifdef NDLL_BUILD_PROTO3
template <>
inline void ArgumentInst<TFRecordParser::Feature>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_type("TFFeature");

  TFRecordParser::Feature self = this->Get();
  // set the datatype of the record
  auto *type_arg = arg->add_extra_args();
  type_arg->set_name("type");
  type_arg->set_i(static_cast<int>(self.GetType()));

  // set the shape
  auto *shape_arg = arg->add_extra_args();
  shape_arg->set_name("shape");
  auto& shape = self.GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_arg->set_ints(i, shape[i]);
  }

  // set the default value
  auto *default_arg = arg->add_extra_args();
  default_arg->set_name("default_value");
  if (self.GetType() == TFRecordParser::FeatureType::int64) {
    default_arg->set_i(self.GetValue().int64);
  } else if (self.GetType() == TFRecordParser::FeatureType::string) {
    default_arg->set_s(self.GetValue().str);
  } else if (self.GetType() == TFRecordParser::FeatureType::float32) {
    default_arg->set_f(self.GetValue().float32);
  } else {
    NDLL_FAIL("Unknown TFRecordParser::FeatureType value");
  }
}
#endif  // NDLL_BUILD_PROTO3

template<typename T>
T Argument::Get() {
  ArgumentInst<T> * self = dynamic_cast<ArgumentInst<T>*>(this);
  if (self == nullptr) {
    NDLL_FAIL("Invalid type of argument " + this->get_name() + ". Expected " + typeid(T).name());
  }
  return self->Get();
}

template<typename T>
Argument * Argument::Store(const std::string& s, const T& val) {
  return new ArgumentInst<T>(s, val);
}

}  // namespace ndll

#endif  // NDLL_PIPELINE_ARGUMENT_H_
