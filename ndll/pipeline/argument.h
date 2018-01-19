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

template <>
inline void ArgumentInst<bool>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_b(this->Get());
}

template <>
inline void ArgumentInst<int64_t>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_i(this->Get());
}

template <>
inline void ArgumentInst<float>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_f(this->Get());
}

template <>
inline void ArgumentInst<string>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  arg->set_s(this->Get());
}

template <>
inline void ArgumentInst<std::vector<float>>::SerializeToProtobuf(ndll_proto::Argument *arg) {
  arg->set_name(Argument::ToString());
  auto vec = this->Get();
  for (size_t i = 0; i < vec.size(); ++i) {
    arg->add_floats(vec[i]);
  }
}

#ifdef NDLL_BUILD_PROTO3
template <>
inline void ArgumentInst<TFRecordParser::Feature>::SerializeToProtobuf(ndll_proto::Argument *arg) {

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
