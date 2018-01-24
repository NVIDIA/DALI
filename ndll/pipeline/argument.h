// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_ARGUMENT_H_
#define NDLL_PIPELINE_ARGUMENT_H_

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/ndll.pb.h"

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

  void SerializeToProtobuf(ndll_proto::Argument *arg) override;

 private:
  T val;
};

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

Argument *DeserializeProtobuf(const ndll_proto::Argument& arg);

}  // namespace ndll

#endif  // NDLL_PIPELINE_ARGUMENT_H_
