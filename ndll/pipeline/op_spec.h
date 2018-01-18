// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OP_SPEC_H_
#define NDLL_PIPELINE_OP_SPEC_H_

#include <map>
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/argument.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

/**
 * @brief Defines all parameters needed to construct an Operator,
 * DataReader, Parser, or Allocator including the object name,
 * any additional input and output tensors it may need, and any
 * number of additional arguments.
 */
class OpSpec {
 public:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  using StrPair = std::pair<string, string>;

  inline OpSpec() {}

  /**
   * @brief Constructs a specification for an op with the given name.
   */
  explicit inline OpSpec(const string &name)
    : name_(name) {}

  /**
   * @brief Getter for the name of the Operator.
   */
  inline const string& name() const { return name_; }

  /**
   * @brief Sets the name of the Operator.
   */
  inline void set_name(const string &name) {
    name_ = name;
  }

  /**
   * @brief Add an argument with the given name and value.
   */
  template <typename T>
  inline OpSpec& AddArg(const string &name, const T &val) {
    Argument * arg = Argument::Store(name, val);
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),
        "AddArg failed. Argument with name \"" + name +
        "\" already exists. ");
    arguments_[name] = arg;
    return *this;
  }

  // Forward to string implementation
  template <unsigned N>
  inline OpSpec& AddArg(const string &name, const char (&c_str)[N]) {
    return this->AddArg<std::string>(name, c_str);
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
  OpSpec& AddInput(const string &name, const string &device);

  /**
   * @brief Specifies the name and device (cpu or gpu) of an
   * output to the op. Intermediate data all have unique names,
   * so a tensor with name "cropped" will refer to the same
   * tensor regardless of whether device is "cpu" or "gpu".
   * The ordering of outputs is also strict. The order in
   * which outputs are added to the OpSpec is the order in
   * which the Operator will receive them.
   */
  OpSpec& AddOutput(const string &name, const string &device);

  inline int NumInput() const { return inputs_.size(); }

  inline int NumOutput() const { return outputs_.size(); }

  inline string Input(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].first + "_" + inputs_[idx].second;
  }

  inline string InputName(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].first;
  }

  inline string InputDevice(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].second;
  }

  inline int InputIdxForName(const string &name, const string &device) {
    auto it = input_name_idx_.find(std::make_pair(name, device));
    NDLL_ENFORCE(it != input_name_idx_.end(), "Input with name '" +
        name + "' and device '" + device + "' does not exist.");
    return it->second;
  }

  inline string Output(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].first + "_" + outputs_[idx].second;
  }

  inline string OutputName(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].first;
  }

  inline string OutputDevice(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].second;
  }

  inline int OutputIdxForName(const string &name, const string &device) {
    auto it = output_name_idx_.find(std::make_pair(name, device));
    NDLL_ENFORCE(it != output_name_idx_.end(), "Output with name '" +
        name + "' and device '" + device + "' does not exist.");
    return it->second;
  }

  /**
   * @brief Checks the spec to see if an argument has been specified
   */
  bool HasArgument(const string &name) const {
    auto arg_it = arguments_.find(name);

    if (arg_it == arguments_.end()) {
      return false;
    } else {
      return true;
    }
  }

  /**
   * @brief Checks the Spec for an argument with the given name/type.
   * Returns the default if an argument with the given name/type does
   * not exist.
   */
  template <typename T>
  inline T GetArgument(const string &name, const T &default_value) const;

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  inline vector<T> GetRepeatedArgument(const string &name,
      const vector<T> &default_value = {}) const;

  inline StrPair* mutable_input(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, NumInput());
    return &inputs_[idx];
  }

  inline StrPair* mutable_output(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, NumOutput());
    return &outputs_[idx];
  }

  string ToString() {
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

 private:
  string name_;
  std::unordered_map<string, Argument*> arguments_;

  std::map<StrPair, int> input_name_idx_, output_name_idx_;
  vector<StrPair> inputs_, outputs_;
};

template <typename T>
inline T OpSpec::GetArgument(const string &name, const T &default_value) const {
  // Search for the argument by name
  auto arg_it = arguments_.find(name);

  if (arg_it == arguments_.end()) {
    return default_value;
  }

  return arg_it->second->template Get<T>();
}

template <typename T>
inline vector<T> OpSpec::GetRepeatedArgument(const string &name,
    const vector<T> &default_value) const {
  // Search for the argument by name
  auto arg_it = arguments_.find(name);

  if (arg_it == arguments_.end()) {
    return default_value;
  }

  return arg_it->second->template Get<vector<T>>();
}

#define INSTANTIATE_ARGUMENT_AS_INT64(T)                                              \
  template<>                                                                          \
  inline OpSpec& OpSpec::AddArg(const string& name, const T& val) {                   \
    return this->AddArg<int64>(name, static_cast<int64>(val));                        \
  }                                                                                   \
  template<>                                                                          \
  inline OpSpec& OpSpec::AddArg(const string& name, const std::vector<T>& val) {      \
    vector<int64> tmp;                                                                \
    for (auto t : val) {                                                              \
      tmp.push_back(static_cast<int64>(t));                                           \
    }                                                                                 \
    Argument * arg = Argument::Store(name, tmp);                                      \
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),                           \
        "AddArg failed. Argument with name \"" + name +                               \
        "\" already exists. ");                                                       \
    arguments_[name] = arg;                                                           \
    return *this;                                                                     \
  }                                                                                   \
  template<>                                                                          \
  inline T OpSpec::GetArgument(const string& name, const T& default_value) const {    \
    int64 tmp = this->GetArgument<int64>(name, static_cast<int64>(default_value));    \
    return static_cast<T>(tmp);                                                       \
  }                                                                                   \
  template<>                                                                          \
  inline std::vector<T> OpSpec::GetRepeatedArgument(                                  \
      const string& name,                                                             \
      const std::vector<T>& default_value) const {                                    \
    auto arg_it = arguments_.find(name);                                              \
    if (arg_it == arguments_.end()) {                                                 \
      return default_value;                                                           \
    }                                                                                 \
    vector<int64> tmp = arg_it->second->template Get<vector<int64>>();                \
    vector<T> ret;                                                                    \
    for (auto t: tmp) {                                                               \
      ret.push_back(static_cast<T>(t));                                               \
    }                                                                                 \
    return ret;                                                                       \
  }

INSTANTIATE_ARGUMENT_AS_INT64(int);
INSTANTIATE_ARGUMENT_AS_INT64(unsigned int);
INSTANTIATE_ARGUMENT_AS_INT64(uint64_t);
INSTANTIATE_ARGUMENT_AS_INT64(NDLLImageType);
INSTANTIATE_ARGUMENT_AS_INT64(NDLLDataType);
INSTANTIATE_ARGUMENT_AS_INT64(NDLLInterpType);
}  // namespace ndll

#endif  // NDLL_PIPELINE_OP_SPEC_H_
