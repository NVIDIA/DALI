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

#ifndef DALI_PIPELINE_OPERATORS_OP_SPEC_H_
#define DALI_PIPELINE_OPERATORS_OP_SPEC_H_

#include <map>
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operators/op_schema.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Defines all parameters needed to construct an Operator,
 * DataReader, Parser, or Allocator including the object name,
 * any additional input and output tensors it may need, and any
 * number of additional arguments.
 */
class DLL_PUBLIC OpSpec {
 public:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  using StrPair = std::pair<string, string>;

  DLL_PUBLIC inline OpSpec() {}

  /**
   * @brief Returns a full tensor name
   * given its name and device
   */
  DLL_PUBLIC static std::string TensorName(std::string name, std::string device) {
    return name + "_" + device;
  }

  /**
   * @brief Constructs a specification for an op with the given name.
   */
  DLL_PUBLIC explicit inline OpSpec(const string &name)
    : name_(name) {}

  /**
   * @brief Getter for the name of the Operator.
   */
  DLL_PUBLIC inline const string& name() const { return name_; }

  /**
   * @brief Sets the name of the Operator.
   */
  DLL_PUBLIC inline void set_name(const string &name) {
    name_ = name;
  }

  /**
   * @brief Add an argument with the given name and value.
   */
  template <typename T>
  DLL_PUBLIC inline OpSpec& AddArg(const string &name, const T &val) {
    DALI_ENFORCE(arguments_.find(name) == arguments_.end(),
        "AddArg failed. Argument with name \"" + name +
        "\" already exists. ");
    return SetArg(name, val);
  }

  /**
   * @brief Add an argument with the given name and value if it doesn't exist already.
   */
  template <typename T>
  DLL_PUBLIC inline OpSpec& AddArgIfNotExisting(const string &name, const T &val) {
    if (arguments_.find(name) != arguments_.end()) {
      return *this;
    }
    return SetArg(name, val);
  }

  /**
   * @brief Sets or adds an argument with the given name and value.
   */
  template <typename T>
  DLL_PUBLIC inline OpSpec& SetArg(const string &name, const T &val) {
    return SetInitializedArg(name, Argument::Store(name, val));
  }

  /**
   * @brief Add an instantiated argument with given name
   */
  DLL_PUBLIC inline OpSpec& AddInitializedArg(const string& name, Argument* arg) {
    DALI_ENFORCE(arguments_.find(name) == arguments_.end(),
        "AddArg failed. Argument with name \"" + name +
        "\" already exists. ");
    arguments_[name].reset(arg);
    return *this;
  }

  /**
   * @brief Sets or adds an argument with given name
   */
  DLL_PUBLIC inline OpSpec& SetInitializedArg(const string& name, Argument* arg) {
    arguments_[name].reset(arg);
    return *this;
  }
  // Forward to string implementation
  template <unsigned N>
  DLL_PUBLIC inline OpSpec& SetArg(const string &name, const char (&c_str)[N]) {
    return this->SetArg<std::string>(name, c_str);
  }

  /**
   * @brief Specifies the name and device (cpu or gpu) of an
   * input to the op. Intermediate data all have unique names,
   * so a tensor with name "cropped" will refer to the same
   * tensor regardless of whether device is "cpu" or "gpu".
   * The ordering of inputs is also strict. The order in
   * which inputs are added to the OpSpec is the order in
   * which the Operator will receive them.
   */
  DLL_PUBLIC OpSpec& AddInput(const string &name, const string &device, bool regular_input = true);

  /**
   * @brief Specifies the argument input to the op.
   * Argument inputs are named inputs that are treated as
   * per-iteration arguments. The input may be added only if
   * corresponding argument exists in the schema.
   */
  DLL_PUBLIC OpSpec& AddArgumentInput(const string &arg_name, const string &inp_name);

  /**
   * @brief Specifies the name and device (cpu or gpu) of an
   * output to the op. Intermediate data all have unique names,
   * so a tensor with name "cropped" will refer to the same
   * tensor regardless of whether device is "cpu" or "gpu".
   * The ordering of outputs is also strict. The order in
   * which outputs are added to the OpSpec is the order in
   * which the Operator will receive them.
   */
  DLL_PUBLIC OpSpec& AddOutput(const string &name, const string &device);

  DLL_PUBLIC inline int NumInput() const { return inputs_.size(); }

  DLL_PUBLIC inline int NumArgumentInput() const {
    return argument_inputs_indexes_.size();
  }

  DLL_PUBLIC inline int NumRegularInput() const {
    return NumInput() - NumArgumentInput();
  }

  DLL_PUBLIC inline int NumOutput() const { return outputs_.size(); }

  DLL_PUBLIC inline string Input(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return TensorName(inputs_[idx].first, inputs_[idx].second);
  }

  DLL_PUBLIC inline string InputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].first;
  }

  DLL_PUBLIC inline string InputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].second;
  }

  DLL_PUBLIC inline bool IsArgumentInput(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return argument_inputs_indexes_.find(idx) != argument_inputs_indexes_.end();
  }

  DLL_PUBLIC inline std::string ArgumentInputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    auto idx_ptr = argument_inputs_indexes_.find(idx);
    DALI_ENFORCE(idx_ptr != argument_inputs_indexes_.end(),
        "Index " + to_string(idx) + " does not correspond to valid argument input.");
    for (const auto& arg_pair : argument_inputs_) {
      if (arg_pair.second == idx) {
        return arg_pair.first;
      }
    }
    DALI_FAIL("Internal error - found argument input index for non-existent argument input.");
  }

  DLL_PUBLIC inline string Output(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return TensorName(outputs_[idx].first, outputs_[idx].second);
  }

  DLL_PUBLIC inline string OutputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].first;
  }

  DLL_PUBLIC inline string OutputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].second;
  }

  DLL_PUBLIC inline const std::unordered_map<string, Index>& ArgumentInputs() const {
    return argument_inputs_;
  }

  DLL_PUBLIC inline const std::unordered_map<string, std::shared_ptr<Argument>>& Arguments() const {
    return arguments_;
  }

  DLL_PUBLIC inline int OutputIdxForName(const string &name, const string &device) {
    auto it = output_name_idx_.find(std::make_pair(name, device));
    DALI_ENFORCE(it != output_name_idx_.end(), "Output with name '" +
        name + "' and device '" + device + "' does not exist.");
    return it->second;
  }

  /**
   * @brief Checks the spec to see if an argument has been specified
   */
  DLL_PUBLIC bool HasArgument(const string &name) const {
    auto arg_it = arguments_.find(name);
    return arg_it != arguments_.end();
  }

  /**
   * @brief Checks the spec to see if a tensor argument has been specified
   */
  DLL_PUBLIC bool HasTensorArgument(const std::string &name) const {
    auto arg_it = argument_inputs_.find(name);
    return arg_it != argument_inputs_.end();
  }

  /**
   * @brief Checks the spec to see if an argument has been specified by one of two possible ways
   */
  DLL_PUBLIC bool ArgumentDefined(const std::string &name) const {
    return HasArgument(name) || HasTensorArgument(name);
  }

  /**
   * @brief Lists all arguments specified in this spec.
   */
  DLL_PUBLIC std::vector<std::string> ListArguments() const {
    std::vector<std::string> ret;
    for (auto &a : arguments_) {
      ret.push_back(a.first);
    }
    for (auto &a : argument_inputs_) {
      ret.push_back(a.first);
    }
    return ret;
  }

  /**
   * @brief Checks the Spec for an argument with the given name/type.
   * Returns the default if an argument with the given name/type does
   * not exist.
   */
  template <typename T>
  DLL_PUBLIC inline T GetArgument(const string &name,
                       const ArgumentWorkspace *ws = nullptr,
                       Index idx = 0) const {
    return GetArgument<T, T>(name, ws, idx);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  DLL_PUBLIC inline std::vector<T> GetRepeatedArgument(
      const string &name, const ArgumentWorkspace *ws = nullptr, Index idx = 0) const {
    DALI_ENFORCE(idx == 0, "Tensor arguments cannot be used for vector values");
    return GetArgument<T, std::vector<T>>(name, ws, idx);
  }

  DLL_PUBLIC OpSpec& ShareArguments(OpSpec& other) {
    this->arguments_ = other.arguments_;
    this->argument_inputs_ = other.argument_inputs_;
    this->argument_inputs_indexes_ = other.argument_inputs_indexes_;
    return *this;
  }

  DLL_PUBLIC inline StrPair& MutableInput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx];
  }

  DLL_PUBLIC inline StrPair& MutableOutput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx];
  }

  DLL_PUBLIC string ToString() const {
    string ret;
    ret += "OpSpec for " + name() + ":\n  Inputs:\n";
    for (size_t i = 0; i < inputs_.size(); ++i) {
      ret += "    " + Input(i) + "\n";
    }
    ret += "  Outputs:\n";
    for (size_t i = 0; i < outputs_.size(); ++i) {
      ret += "    " + Output(i) + "\n";
    }
    ret += "  Arguments:\n";
    for (auto& a : arguments_) {
      ret += "    ";
      ret += a.second->ToString();
      ret += "\n";
    }
    return ret;
  }

  DLL_PUBLIC OpSpec& operator=(const OpSpec& other) {
    this->name_ = other.name_;
    this->arguments_ = other.arguments_;
    this->argument_inputs_ = other.argument_inputs_;
    this->argument_inputs_indexes_ = other.argument_inputs_indexes_;
    this->output_name_idx_ = other.output_name_idx_;
    this->inputs_ = other.inputs_;
    this->outputs_ = other.outputs_;
    return *this;
  }

 private:
  template <typename T, typename S>
  inline S GetArgument(const string &name, const ArgumentWorkspace *ws, Index idx) const;

  string name_;
  std::unordered_map<string, std::shared_ptr<Argument>> arguments_;
  std::unordered_map<string, Index> argument_inputs_;
  std::set<Index> argument_inputs_indexes_;

  std::map<StrPair, int> output_name_idx_;
  vector<StrPair> inputs_, outputs_;
};

template <typename T, typename S>
inline S OpSpec::GetArgument(const string &name, const ArgumentWorkspace *ws, Index idx) const {
  // Search for the argument in tensor arguments first
  if (this->HasTensorArgument(name)) {
    DALI_ENFORCE(ws != nullptr, "Tensor value is unexpected for argument \"" + name + "\".");
    const auto& value = ws->ArgumentInput(name);
    DALI_ENFORCE(IsType<S>(value.type()),
        "Unexpected type of argument \"" + name + "\". Expected " +
        TypeTable::GetTypeName<S>() + " and got " + value.type().name());
    return value.template data<S>()[idx];
  }
  // Search for the argument locally
  auto arg_it = arguments_.find(name);
  if (arg_it != arguments_.end()) {
    // Found locally - return
    return arg_it->second->template Get<S>();
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = SchemaRegistry::GetSchema(this->name());
    return schema.GetDefaultValueForOptionalArgument<S>(name);
  }
}

#define INSTANTIATE_ARGUMENT_AS_INT64(T)                                                        \
  template<>                                                                                    \
  inline OpSpec& OpSpec::SetArg(const string& name, const T& val) {                             \
    return this->SetArg<int64>(name, static_cast<int64>(val));                                  \
  }                                                                                             \
  template<>                                                                                    \
  inline OpSpec& OpSpec::SetArg(const string& name, const std::vector<T>& val) {                \
    vector<int64> tmp;                                                                          \
    for (auto t : val) {                                                                        \
      tmp.push_back(static_cast<int64>(t));                                                     \
    }                                                                                           \
    Argument * arg = Argument::Store(name, tmp);                                                \
    arguments_[name].reset(arg);                                                                \
    return *this;                                                                               \
  }                                                                                             \
  template<>                                                                                    \
  inline T OpSpec::GetArgument(const string& name, const ArgumentWorkspace *ws, Index idx) const { \
    if (this->HasTensorArgument(name)) {                                                           \
      DALI_ENFORCE(ws != nullptr, "Tensor value is unexpected for argument \"" + name + "\".");    \
      const auto& value = ws->ArgumentInput(name);                                                 \
      if (IsType<T>(value.type())) {                                                               \
        return value.template data<T>()[idx];                                                      \
      }                                                                                            \
    }                                                                                              \
    int64 tmp = this->GetArgument<int64>(name, ws, idx);                                           \
    return static_cast<T>(tmp);                                                                    \
  }                                                                                                \
  template<>                                                                                       \
  inline std::vector<T> OpSpec::GetRepeatedArgument(                                               \
      const string& name, const ArgumentWorkspace *ws, Index idx) const {                          \
    vector<int64> tmp = this->GetRepeatedArgument<int64>(name, ws, idx);                           \
    vector<T> ret;                                                                                 \
    for (auto t : tmp) {                                                                           \
      ret.push_back(static_cast<T>(t));                                                            \
    }                                                                                              \
    return ret;                                                                                    \
  }

INSTANTIATE_ARGUMENT_AS_INT64(int);
INSTANTIATE_ARGUMENT_AS_INT64(unsigned int);
INSTANTIATE_ARGUMENT_AS_INT64(uint64_t);
INSTANTIATE_ARGUMENT_AS_INT64(int8_t);
INSTANTIATE_ARGUMENT_AS_INT64(uint8_t);
INSTANTIATE_ARGUMENT_AS_INT64(DALIImageType);
INSTANTIATE_ARGUMENT_AS_INT64(DALIDataType);
INSTANTIATE_ARGUMENT_AS_INT64(DALIInterpType);
INSTANTIATE_ARGUMENT_AS_INT64(DALITensorLayout);
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OP_SPEC_H_
