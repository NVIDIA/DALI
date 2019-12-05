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

#ifndef DALI_PIPELINE_OPERATOR_OP_SPEC_H_
#define DALI_PIPELINE_OPERATOR_OP_SPEC_H_

#include <map>
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>
#include <type_traits>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

namespace detail {

template <typename T, typename S>
void copy_vector(std::vector<T> &out, const std::vector<S> &in) {
  out.reserve(in.size());
  out.clear();
  for (decltype(auto) v : in) {
    out.emplace_back(v);
  }
}

/** @brief This overload simply forwards the reference */
template <typename T>
std::vector<T> &&convert_vector(std::vector<T> &&v) {
  return std::move(v);
}

/** @brief This overload simply forwards the reference */
template <typename T>
const std::vector<T> &convert_vector(const std::vector<T> &v) {
  return v;
}

/** @brief This overload converts elements from v and returns a vector of converted objects */
template <typename T, typename S>
std::enable_if_t<!std::is_same<T, S>::value, std::vector<T>>
convert_vector(const std::vector<S> &v) {
  std::vector<T> out;
  copy_vector(out, v);
  return out;
}

}  // namespace detail

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
  struct InOutDeviceDesc {
    std::string name;
    std::string device;
    bool operator<(const InOutDeviceDesc &other) const {
      return std::make_pair(name, device) < std::make_pair(other.name, other.device);
    }
  };

  DLL_PUBLIC inline OpSpec() {}

  /**
   * @brief Returns a full tensor name
   * given its name and device
   */
  DLL_PUBLIC static std::string TensorName(const std::string &name, const std::string &device) {
    return name + "_" + device;
  }

  /**
   * @brief Constructs a specification for an op with the given name.
   */
  DLL_PUBLIC explicit inline OpSpec(const string &name) {
    set_name(name);
  }

  /**
   * @brief Getter for the name of the Operator.
   */
  DLL_PUBLIC inline const string& name() const { return name_; }

  /**
   * @brief Sets the name of the Operator.
   */
  DLL_PUBLIC inline void set_name(const string &name) {
    name_ = name;
    schema_ = name.empty() ? nullptr : SchemaRegistry::TryGetSchema(name);
  }

  DLL_PUBLIC inline const OpSchema &GetSchema() const {
    DALI_ENFORCE(schema_ != nullptr, "No schema found for operator \"" + name() + "\"");
    return *schema_;
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
    using S = argument_storage_t<T>;
    return SetInitializedArg(name, Argument::Store<S>(name, static_cast<S>(val)));
  }

  /**
   * @brief Sets or adds an argument with the given name and value.
   */
  template <typename T>
  DLL_PUBLIC inline OpSpec& SetArg(const string &name, const std::vector<T> &val) {
    using S = argument_storage_t<T>;
    using V = std::vector<S>;
    return SetInitializedArg(name, Argument::Store<V>(name, detail::convert_vector<S>(val)));
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
    return TensorName(inputs_[idx].name, inputs_[idx].device);
  }

  DLL_PUBLIC inline string InputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].name;
  }

  DLL_PUBLIC inline string InputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].device;
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
    return TensorName(outputs_[idx].name, outputs_[idx].device);
  }

  DLL_PUBLIC inline string OutputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].name;
  }

  DLL_PUBLIC inline string OutputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].device;
  }

  DLL_PUBLIC inline const std::unordered_map<string, Index>& ArgumentInputs() const {
    return argument_inputs_;
  }

  DLL_PUBLIC inline const std::unordered_map<string, std::shared_ptr<Argument>>& Arguments() const {
    return arguments_;
  }

  DLL_PUBLIC inline int OutputIdxForName(const string &name, const string &device) {
    auto it = output_name_idx_.find({name, device});
    DALI_ENFORCE(it != output_name_idx_.end(), "Output with name '" +
        name + "' and device '" + device + "' does not exist.");
    return it->second;
  }

  /**
   * @brief Checks the spec to see if an argument has been specified
   *
   * @remark If user does not explicitly specify value for OptionalArgument,
   *         this will return false.
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
    using S = argument_storage_t<T>;
    return GetArgumentImpl<T, S>(name, ws, idx);
  }

  template <typename T>
  DLL_PUBLIC inline bool TryGetArgument(T &result,
                                        const string &name,
                                        const ArgumentWorkspace *ws = nullptr,
                                        Index idx = 0) const {
    using S = argument_storage_t<T>;
    return TryGetArgumentImpl<T, S>(result, name, ws, idx);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  DLL_PUBLIC inline std::vector<T> GetRepeatedArgument(const string &name) const {
    using S = argument_storage_t<T>;
    return GetRepeatedArgumentImpl<T, S>(name);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  DLL_PUBLIC bool TryGetRepeatedArgument(
      std::vector<T> &result,
      const string &name) const {
    using S = argument_storage_t<T>;
    return TryGetRepeatedArgumentImpl<T, S>(result, name);
  }

  DLL_PUBLIC OpSpec& ShareArguments(OpSpec& other) {
    this->arguments_ = other.arguments_;
    this->argument_inputs_ = other.argument_inputs_;
    this->argument_inputs_indexes_ = other.argument_inputs_indexes_;
    return *this;
  }

  DLL_PUBLIC inline InOutDeviceDesc& MutableInput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx];
  }

  DLL_PUBLIC inline InOutDeviceDesc& MutableOutput(int idx) {
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

 private:
  template <typename T, typename S>
  inline T GetArgumentImpl(const string &name, const ArgumentWorkspace *ws, Index idx) const;

  /**
   * @brief Check if the ArgumentInput of given shape can be used with GetArgument(),
   *        representing a batch of scalars
   *
   * @argument should_throw whether this function should throw an error if the shape doesn't match
   * @return true iff the shape is allowed to be used as Argument
   */
  bool CheckArgumentShape(const TensorListShape<> &shape, int batch_size,
                          const std::string &name, bool should_throw = false) const {
    DALI_ENFORCE(is_uniform(shape),
                 "Arguments should be passed as uniform TensorLists. Argument \"" + name +
                     "\" is not uniform. To access non-uniform argument inputs use "
                     "ArgumentWorkspace::ArgumentInput method directly.");
    if (shape.num_samples() == 1) {
      // TODO(klecki): 1 sample version will be of no use after the switch to CPU ops unless we
      // generalize accepted batch sizes throughout the pipeline
      bool is_one_sample_with_batch = shape[0] == TensorShape<>(batch_size);
      if (should_throw) {
        DALI_ENFORCE(is_one_sample_with_batch,
            "Unexpected shape of argument \"" + name +
                "\". If only one tensor is passed, it should have a shape equal to {" +
                std::to_string(batch_size) +
                "}.  When accessing arguments as scalars 1 tensor of shape {" +
                std::to_string(batch_size) + "} or " + std::to_string(batch_size) +
                " tensors of shape {1} are expected. To access argument inputs where samples are "
                "not scalars use ArgumentWorkspace::ArgumentInput method directly.");
      } else if (!is_one_sample_with_batch) {
        return false;
      }
    } else {
      bool is_batch_of_scalars =
          shape.num_samples() == batch_size && shape.sample_dim() == 1 && shape[0][0] == 1;
      if (should_throw) {
        DALI_ENFORCE(
            is_batch_of_scalars,
            "Unexpected shape of argument \"" + name + "\". Expected batch of " +
                std::to_string(batch_size) + " tensors of shape {1}, got " +
                std::to_string(shape.num_samples()) + " samples of " +
                std::to_string(shape.sample_dim()) +
                "D tensors. Alternatively, a single 1D tensor with " + std::to_string(batch_size) +
                " elements can be passed. To access argument inputs where samples are not scalars "
                "use ArgumentWorkspace::ArgumentInput method directly.");
      } else if (!is_batch_of_scalars) {
        return false;
      }
    }
    return true;
  }

  template <typename T, typename S>
  inline bool TryGetArgumentImpl(T &result,
                                 const string &name,
                                 const ArgumentWorkspace *ws,
                                 Index idx) const;

  template <typename T, typename S>
  inline std::vector<T> GetRepeatedArgumentImpl(const string &name) const;

  template <typename T, typename S>
  inline bool TryGetRepeatedArgumentImpl(std::vector<T> &result, const string &name) const;

  string name_;
  const OpSchema *schema_ = nullptr;
  std::unordered_map<string, std::shared_ptr<Argument>> arguments_;
  std::unordered_map<string, Index> argument_inputs_;
  std::set<Index> argument_inputs_indexes_;

  std::map<InOutDeviceDesc, int> output_name_idx_;
  vector<InOutDeviceDesc> inputs_, outputs_;
};


template <typename T, typename S>
inline T OpSpec::GetArgumentImpl(
      const string &name,
      const ArgumentWorkspace *ws,
      Index idx) const {
  // Search for the argument in tensor arguments first
  if (this->HasTensorArgument(name)) {
    DALI_ENFORCE(ws != nullptr, "Tensor value is unexpected for argument \"" + name + "\".");
    const auto &value = ws->ArgumentInput(name);
    CheckArgumentShape(value.shape(), GetArgument<int>("batch_size"), name, true);
    DALI_ENFORCE(IsType<T>(value.type()),
        "Unexpected type of argument \"" + name + "\". Expected " +
        TypeTable::GetTypeName<T>() + " and got " + value.type().name());
    return static_cast<T>(value[idx].data<T>()[0]);
  }
  // Search for the argument locally
  auto arg_it = arguments_.find(name);
  if (arg_it != arguments_.end()) {
    // Found locally - return
    return static_cast<T>(arg_it->second->template Get<S>());
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchema();
    return static_cast<T>(schema.GetDefaultValueForOptionalArgument<S>(name));
  }
}

template <typename T, typename S>
inline bool OpSpec::TryGetArgumentImpl(
      T &result,
      const string &name,
      const ArgumentWorkspace *ws,
      Index idx) const {
  // Search for the argument in tensor arguments first
  if (this->HasTensorArgument(name)) {
    if (ws == nullptr)
      return false;
    const auto& value = ws->ArgumentInput(name);
    if (!CheckArgumentShape(value.shape(), GetArgument<int>("batch_size"), name, false)) {
      return false;
    }
    if (!IsType<T>(value.type()))
      return false;
    result = value[idx].data<T>()[0];
    return true;
  }
  // Search for the argument locally
  auto arg_it = arguments_.find(name);
  if (arg_it != arguments_.end()) {
    // Found locally - return
    if (arg_it->second->template IsType<S>()) {
      result = static_cast<T>(arg_it->second->template Get<S>());
      return true;
    }
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchema();
    auto schema_val = schema.FindDefaultValue(name);
    using VT = const ValueInst<S>;
    if (VT *vt = dynamic_cast<VT *>(schema_val.second)) {
      result = static_cast<T>(vt->Get());
      return true;
    }
  }
  return false;
}


template <typename T, typename S>
inline std::vector<T> OpSpec::GetRepeatedArgumentImpl(const string &name) const {
  using V = std::vector<S>;
  // Search for the argument locally
  auto arg_it = arguments_.find(name);
  if (arg_it != arguments_.end()) {
    // Found locally - return
    return detail::convert_vector<T>(arg_it->second->template Get<V>());
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchema();
    return detail::convert_vector<T>(schema.GetDefaultValueForOptionalArgument<V>(name));
  }
}

template <typename T, typename S>
inline bool OpSpec::TryGetRepeatedArgumentImpl(std::vector<T> &result, const string &name) const {
  using V = std::vector<S>;
  // Search for the argument locally
  auto arg_it = arguments_.find(name);
  if (arg_it != arguments_.end()) {
    // Found locally - return
    if (arg_it->second->template IsType<V>()) {
      detail::copy_vector(result, arg_it->second->template Get<V>());
      return true;
    }
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchema();
    auto schema_val = schema.FindDefaultValue(name);
    using VT = const ValueInst<V>;
    if (VT *vt = dynamic_cast<VT *>(schema_val.second)) {
      detail::copy_vector(result, vt->Get());
      return true;
    }
  }
  return false;
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OP_SPEC_H_
