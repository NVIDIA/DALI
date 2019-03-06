// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_OPERATOR_ARGUMENT_H_
#define DALI_TEST_OPERATOR_ARGUMENT_H_

#include <string>
#include <memory>
#include <vector>

#include "dali/kernels/util.h"
#include "dali/pipeline/operators/op_spec.h"

namespace dali {
namespace testing {

template<typename T>
struct InputArg {
  // TODO(mszolucha): For Dali's argument input
};

template <typename T, typename Enable = void>
struct TestOpArgToStringImpl {
  static std::string to_string(const T&) {
    return {};
  }
};

template <typename T>
struct TestOpArgToStringImpl<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static std::string to_string(const T& val) {
    return std::to_string(val);
  }
};

template <>
struct TestOpArgToStringImpl<std::string> {
  static std::string to_string(const std::string& val) {
    return "\"" + val + "\"";
  }
};

template <typename T>
struct TestOpArgToStringImpl<T, if_iterable<T, void>> {
  static std::string to_string(const T& val) {
    std::stringstream ss;
    if (val.size() == 0) {
      return "[]";
    } else {
      ss << "[ ";
    }
    using element_type = element_t<T>;
    bool first = true;
    for (const auto& e : val) {
      if (!first) {
        ss << ", ";
      } else {
        first = false;
      }
      ss << TestOpArgToStringImpl<element_type>::to_string(e);
    }
    ss << " ]";
    return ss.str();
  }
};


template<typename T>
class TestOpArgValue;

class TestOpArgBase {
 public:
  virtual void SetArg(const std::string &name, OpSpec &opspec, ArgumentWorkspace *ws) const = 0;

  virtual bool IsArgumentInput() const = 0;

  virtual ~TestOpArgBase() = default;


  template<typename T>
  T GetValue() const {
    return dynamic_cast<const TestOpArgValue<T> &>(*this).value;
  }

  virtual std::string to_string() const = 0;

 protected:
  TestOpArgBase() = default;
};

template<typename T>
struct is_argument_input_type : std::false_type {
};

template<typename T>
struct is_argument_input_type<InputArg<T>> : std::true_type {
};

template<typename T, bool isArgInput = is_argument_input_type<T>::value>
class TestOpArgValueImpl : public TestOpArgBase {
 public:
  TestOpArgValueImpl() = default;


  TestOpArgValueImpl(const T &value) :  // NOLINT non-explicit ctor
          value(value) {
  }


  void
  SetArg(const std::string &argument_name, OpSpec &opspec, ArgumentWorkspace *ws) const override {
    opspec.AddArg(argument_name, value);
  }

  std::string to_string() const override {
    return TestOpArgToStringImpl<T>::to_string(value);
  }

  bool IsArgumentInput() const override { return isArgInput; }


  T value = {};
};

template<typename T>
class TestOpArgValueImpl<T, true> : public TestOpArgBase {
  TestOpArgValueImpl() = default;


  TestOpArgValueImpl(const T &value) :  // NOLINT non-explicit ctor
          value(value) {
  }


  void SetArg(const std::string &name, OpSpec &spec, ArgumentWorkspace *ws) const override {
    // TODO(mszolucha): For ArgumentInput
  }


  bool IsArgumentInput() const override { return true; }


  T value = {};
};

template<typename T>
class TestOpArgValue : public TestOpArgValueImpl<T> {
  using TestOpArgValueImpl<T>::TestOpArgValueImpl;
};

class TestOpArg {
 public:
  TestOpArg() = default;

  template<size_t N>
  TestOpArg(const char (&text)[N]) : TestOpArg(std::string(text)) {}  // NOLINT

  TestOpArg(const char *text) : TestOpArg(std::string(text)) {}  // NOLINT

  template<typename T>
  TestOpArg(const T &value) :  // NOLINT non-explicit ctor
          val(new TestOpArgValue<T>(value)) {
  }


  void SetArg(const std::string &argument_name, OpSpec &spec, ArgumentWorkspace *ws) {
    val->SetArg(argument_name, spec, ws);
  }

  std::string to_string() const {
    return val->to_string();
  }


  template<typename T>
  T GetValue() const {
    assert(val && "Value not set");
    return val->GetValue<T>();
  }


  std::shared_ptr<TestOpArgBase> val;
};

inline std::ostream& operator<<(std::ostream& os, const TestOpArg& op_arg) {
  os << op_arg.to_string();
  return os;
}

}  // namespace testing
}  // namespace dali

#endif  //  DALI_TEST_OPERATOR_ARGUMENT_H_
