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

#ifndef DALI_PIPELINE_OPERATOR_ARGUMENT_H_
#define DALI_PIPELINE_OPERATOR_ARGUMENT_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/proto/dali_proto_utils.h"

namespace dali {


template <typename T>
struct argument_storage {
  using type = std::conditional_t<
      std::is_integral<T>::value || std::is_enum<T>::value,
      int64_t, T>;
};

template <>
struct argument_storage<bool> {
  using type = bool;
};

template <>
struct argument_storage<TensorLayout> {
  using type = std::string;
};

template <typename T>
using argument_storage_t = typename argument_storage<T>::type;

class Value {
 public:
  virtual std::string ToString() const = 0;
  template <typename T>
  static inline std::unique_ptr<Value> construct(const T& val);
  DALIDataType GetTypeID() const {
    return type_;
  }
  virtual ~Value() = default;

 protected:
  Value() : type_(DALI_NO_TYPE) {}

  void SetTypeID(DALIDataType dtype) {
    type_ = dtype;
  }

  DALIDataType type_;
};

template <typename T>
class ValueInst : public Value {
 public:
  template <typename... Args>
  explicit ValueInst(Args&&... args) : val_(std::forward<Args>(args)...) {
    this->type_ = TypeTable::GetTypeID<T>();
  }

  std::string ToString() const override {
    return to_string(val_);
  }

  const T &Get() const {
    return val_;
  }

 private:
  T val_;
};

template <typename T>
inline std::unique_ptr<Value> Value::construct(const T& val) {
  using S = argument_storage_t<T>;
  return std::unique_ptr<Value>(new ValueInst<S>(val));
}

/**
 * @brief Stores a single argument.
 *
 * Argument class is the wrapper parent class for wrapper classes storing arguments given to ops.
 * In order to add a new type of argument, one needs to expose the type to Python
 * in python/dali_backend.cc file by using py::class_<new_type> and DALI_OPSPEC_ADDARG macro.
 * For integral types (like enums), one needs to use INSTANTIATE_ARGUMENT_AS_INT64 macro
 * in operators/op_spec.h instead.
 */
class Argument {
 public:
  // Setters & getters for name
  inline bool has_name() const {
    return has_name_;
  }

  inline const string get_name() const {
    return has_name() ? name_ : "<no name>";
  }

  inline void set_name(string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  inline void clear_name() {
    has_name_ = false;
    name_.clear();
  }

  virtual std::string ToString() const {
    return get_name();
  }

  virtual DALIDataType GetTypeID() const = 0;

  virtual void SerializeToProtobuf(DaliProtoPriv* arg) = 0;

  template <typename T>
  T Get();

  template <typename T>
  bool IsType();

  template <typename T>
  static Argument* Store(const std::string& s, const T& val);

  virtual ~Argument() = default;

 protected:
  Argument() : has_name_(false) {}

  explicit Argument(const std::string& s) : name_(s), has_name_(true) {}

 private:
  std::string name_;
  bool has_name_;
};

template <typename T>
class ArgumentInst : public Argument {
 public:
  explicit ArgumentInst(const std::string& s, const T& v) : Argument(s), val(v) {}

  T Get() {
    return val.Get();
  }

  std::string ToString() const override {
    string ret = Argument::ToString();
    ret += ": ";
    ret += val.ToString();
    return ret;
  }

  DALIDataType GetTypeID() const override {
    return val.GetTypeID();
  }

  void SerializeToProtobuf(DaliProtoPriv* arg) override {
    arg->set_name(Argument::ToString());
    dali::SerializeToProtobuf(val.Get(), arg);
  }

 private:
  ValueInst<T> val;
};

template <typename T>
class ArgumentInst<std::vector<T>> : public Argument {
 public:
  explicit ArgumentInst(const std::string& s, const std::vector<T>& v) : Argument(s), val(v) {}

  std::vector<T> Get() {
    return val.Get();
  }

  std::string ToString() const override {
    string ret = Argument::ToString();
    ret += ": ";
    ret += val.ToString();
    return ret;
  }

  DALIDataType GetTypeID() const override {
    return val.GetTypeID();
  }

  void SerializeToProtobuf(DaliProtoPriv* arg) override {
    const std::vector<T>& vec = val.Get();
    arg->set_name(Argument::ToString());
    arg->set_type(dali::serialize_type(T()));
    arg->set_is_vector(true);
    for (size_t i = 0; i < vec.size(); ++i) {
      ArgumentInst<T> tmp("element " + to_string(i), vec[i]);
      auto extra_arg = arg->add_extra_args();
      tmp.SerializeToProtobuf(&extra_arg);
    }
  }

 private:
  ValueInst<std::vector<T>> val;
};

DLL_PUBLIC Argument* DeserializeProtobuf(const DaliProtoPriv &arg);

template <typename T>
bool Argument::IsType() {
  return dynamic_cast<ArgumentInst<T>*>(this) != nullptr;
}

template <typename T>
T Argument::Get() {
  ArgumentInst<T>* self = dynamic_cast<ArgumentInst<T>*>(this);
  if (self == nullptr) {
    DALI_FAIL("Invalid type of argument \"" + this->get_name() + "\". Expected " +
              typeid(T).name());
  }
  return self->Get();
}

template <typename T>
Argument* Argument::Store(const std::string& s, const T& val) {
  return new ArgumentInst<T>(s, val);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARGUMENT_H_
