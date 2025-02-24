// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/copy_vector_helper.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operator/op_schema.h"
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
  struct InOutDeviceDesc {
    std::string name;
    StorageDevice device;

    bool operator<(const InOutDeviceDesc &other) const {
      return std::tie(name, device) < std::tie(other.name, other.device);
    }

    template <typename NameT, typename DeviceT>
    bool operator<(const std::tuple<NameT, DeviceT> &other) const {
      return std::tie(name, device) < other;
    }

    template <typename NameT, typename DeviceT>
    friend bool operator<(const std::tuple<NameT, DeviceT> &l, const InOutDeviceDesc &r) {
      return l < std::tie(r.name, r.device);
    }
  };

  OpSpec() = default;

  /** Returns a full tensor name given its name and device */
  static std::string TensorName(std::string_view name, StorageDevice device) {
    return make_string(name, (device == StorageDevice::GPU ? "_gpu" : "_cpu"));
  }

  /** Constructs a specification for an op with the given schema name. */
  explicit inline OpSpec(std::string_view schema_name) {
    SetSchema(schema_name);
  }

  /** Getter for the schema name of the Operator. */
  const string &SchemaName() const { return schema_name_; }

  /** Sets the schema of the Operator. */
  void SetSchema(std::string_view schema_name) {
    schema_name_ = std::string(schema_name);
    schema_ = schema_name_.empty() ? nullptr : SchemaRegistry::TryGetSchema(schema_name_);
  }

  /** Sets the schema of the Operator. */
  void SetSchema(OpSchema *schema) {
    schema_ = schema;
    if (schema)
      schema_name_ = schema->name();
    else
      schema_name_ = {};
  }

  const OpSchema &GetSchema() const {
    DALI_ENFORCE(schema_ != nullptr,
                 make_string("No schema found for operator \"", SchemaName(), "\""));
    return *schema_;
  }

  const OpSchema &GetSchemaOrDefault() const {
    return schema_ ? *schema_ : OpSchema::Default();
  }

  /** Add an argument with the given name and value. */
  template <typename T>
  OpSpec &AddArg(std::string_view name, const T &val) {
    EnforceNoAliasWithDeprecated(name);
    DALI_ENFORCE(argument_idxs_.find(name) == argument_idxs_.end(),
                 make_string("AddArg failed. Argument with name \"", name, "\" already exists. "));
    return SetArg(name, val);
  }

  /** Add an argument with the given name and value if it doesn't exist already. */
  template <typename T>
  OpSpec &AddArgIfNotExisting(std::string_view name, const T &val) {
    if (argument_idxs_.find(name) != argument_idxs_.end()) {
      return *this;
    }
    return SetArg(name, val);
  }

  /** Sets or adds an argument with the given name and value. */
  template <typename T>
  OpSpec &SetArg(std::string_view name, const T &val) {
    using S = argument_storage_t<T>;
    return SetInitializedArg(name, Argument::Store<S>(std::string(name), static_cast<S>(val)));
  }

  /** Sets or adds an argument with the given name and value. */
  template <typename T>
  OpSpec &SetArg(std::string_view name, const std::vector<T> &val) {
    using S = argument_storage_t<T>;
    using V = std::vector<S>;
    return SetInitializedArg(
        name,
        Argument::Store<V>(std::string(name),
        detail::convert_vector<S>(val)));
  }


  /** Add an instantiated argument with given name */
  OpSpec &AddInitializedArg(std::string_view name, std::shared_ptr<Argument> arg) {
    EnforceNoAliasWithDeprecated(name);
    DALI_ENFORCE(argument_idxs_.find(name) == argument_idxs_.end(),
                 make_string("AddArg failed. Argument with name \"", name, "\" already exists. "));
    return SetInitializedArg(name, arg);
  }

  /**
   * @brief Sets or adds an argument with given name
   *
   * @remarks Deprecated arguments are renamed (or dropped, if no longer used).
   */
  OpSpec &SetInitializedArg(std::string_view arg_name, std::shared_ptr<Argument> arg);

  /** Check if the `arg_name` was already set through a deprecated argument */
  void EnforceNoAliasWithDeprecated(std::string_view arg_name);

  // Forward to string implementation
  template <size_t N>
  OpSpec &SetArg(std::string_view name, const char (&c_str)[N]) {
    return this->SetArg(name, std::string(c_str));
  }

  // Forward to string implementation
  OpSpec &SetArg(std::string_view name, const char *c_str) {
    return this->SetArg(name, std::string(c_str));
  }

  // Forward to string implementation
  OpSpec &SetArg(std::string_view name, const std::string_view &str) {
    return this->SetArg(name, std::string(str));
  }

  /**
   * @brief Specifies the name and device (cpu or gpu) of an
   * input to the op. Intermediate data all have unique names,
   * so a tensor with name "cropped" will refer to the same
   * tensor regardless of whether device is CPU or GPU.
   * The ordering of inputs is also strict. The order in
   * which inputs are added to the OpSpec is the order in
   * which the Operator will receive them.
   */
  OpSpec &AddInput(std::string name, StorageDevice device, bool regular_input = true);

  /**
   * @brief Specifies the argument input to the op.
   * Argument inputs are named inputs that are treated as
   * per-iteration arguments. The input may be added only if
   * corresponding argument exists in the schema.
   */
  OpSpec &AddArgumentInput(std::string arg_name, std::string inp_name);

  /**
   * @brief Specifies the name and device (cpu or gpu) of an
   * output to the op. Intermediate data all have unique names,
   * so a tensor with name "cropped" will refer to the same
   * tensor regardless of whether device is CPU or GPU.
   * The ordering of outputs is also strict. The order in
   * which outputs are added to the OpSpec is the order in
   * which the Operator will receive them.
   */
  OpSpec &AddOutput(std::string name, StorageDevice device);

  int NumInput() const { return inputs_.size(); }

  int NumArgumentInput() const {
    return argument_inputs_.size();
  }

  int NumRegularInput() const {
    return NumInput() - NumArgumentInput();
  }

  int NumOutput() const { return outputs_.size(); }

  string Input(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return TensorName(inputs_[idx].name, inputs_[idx].device);
  }

  string InputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].name;
  }

  StorageDevice InputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].device;
  }

  bool IsArgumentInput(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return idx >= NumRegularInput();
  }

  std::string ArgumentInputName(int idx) const {
    DALI_ENFORCE(IsArgumentInput(idx),
        make_string("Index ", idx, " does not correspond to valid argument input."));
    return argument_inputs_[idx - NumRegularInput()].first;
  }

  int ArgumentInputIdx(std::string_view name) const {
    auto it = argument_input_idxs_.find(name);
    DALI_ENFORCE(it != argument_input_idxs_.end(),
                 make_string("No such argument input: \"", name, "\""));
    return it->second;
  }

  string Output(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return TensorName(outputs_[idx].name, outputs_[idx].device);
  }

  string OutputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].name;
  }

  StorageDevice OutputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].device;
  }

  int OutputIdxForName(std::string_view name, StorageDevice device) const {
    auto it = output_name_idx_.find(std::make_tuple(name, device));
    if (it == output_name_idx_.end())
      throw invalid_key(make_string("No such output: \"", TensorName(name, device), "\""));
    return it->second;
  }

  void RenameInput(int idx, std::string name) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    inputs_[idx].name = std::move(name);
  }

  void RenameOutput(int idx, std::string name) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    outputs_[idx].name = std::move(name);
  }

  auto &ArgumentInputs() const {
    return argument_inputs_;
  }

  const auto &Arguments() const {
    return arguments_;
  }

  /**
   * @brief Checks the spec to see if an argument has been specified
   *
   * @remark If user does not explicitly specify value for OptionalArgument,
   *         this will return false.
   */
  bool HasArgument(std::string_view name) const {
    return argument_idxs_.count(name);
  }

  /** Checks the spec to see if a tensor argument has been specified */
  bool HasTensorArgument(std::string_view name) const {
    return argument_input_idxs_.count(name);
  }

  /** Checks the spec to see if an argument has been specified by one of two possible ways */
  bool ArgumentDefined(std::string_view name) const {
    return HasArgument(name) || HasTensorArgument(name);
  }

  /** Lists all arguments specified in this spec. */
  auto ListArgumentNames() const {
    std::set<std::string_view, std::less<>> ret;
    for (auto &a : arguments_) {
      ret.insert(a->get_name());
    }
    for (auto &a : argument_inputs_) {
      ret.insert(a.first);
    }
    return ret;
  }

  /**
   * @brief Checks the Spec for an argument with the given name/type.
   * Returns the default if an argument with the given name/type does
   * not exist.
   */
  template <typename T>
  T GetArgument(
        std::string_view name,
        const ArgumentWorkspace *ws = nullptr,
        Index idx = 0) const {
    using S = argument_storage_t<T>;
    return GetArgumentImpl<T, S>(name, ws, idx);
  }

  template <typename T>
  bool TryGetArgument(
        T &result,
        std::string_view name,
        const ArgumentWorkspace *ws = nullptr,
        Index idx = 0) const {
    using S = argument_storage_t<T>;
    return TryGetArgumentImpl<T, S>(result, name, ws, idx);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   *
   * @remark On Python level the arguments marked with *_VEC type, convert a single value
   * of element type to a list of element types, so GetRepeatedArgument can be used.
   * When the argument is set through C++ there is no such conversion and GetSingleOrRepeatedArg()
   * should be used instead.
   */
  template <typename T>
  std::vector<T> GetRepeatedArgument(std::string_view name) const {
    using S = argument_storage_t<T>;
    return GetRepeatedArgumentImpl<T, S>(name);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename Collection>
  bool TryGetRepeatedArgument(Collection &result, std::string_view name) const {
    using T = typename Collection::value_type;
    using S = argument_storage_t<T>;
    return TryGetRepeatedArgumentImpl<S>(result, name);
  }

  InOutDeviceDesc& MutableInput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx];
  }

  InOutDeviceDesc& MutableOutput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx];
  }

  string ToString() const {
    std::stringstream ss;
    print(ss, "OpSpec for ", SchemaName(), ":\n  Inputs:\n");
    for (size_t i = 0; i < inputs_.size(); ++i) {
      print(ss, "    ", Input(i), "\n");
    }
    print(ss, "  Outputs:\n");
    for (size_t i = 0; i < outputs_.size(); ++i) {
      print(ss, "    ", Output(i), "\n");
    }
    print(ss, "  Arguments:\n");
    for (auto& a : arguments_) {
      print(ss, "    ", a->ToString(), "\n");
    }
    return ss.str();
  }

 private:
  template <typename T, typename S>
  T GetArgumentImpl(std::string_view name, const ArgumentWorkspace *ws, Index idx) const;

  /**
   * @brief Check if the ArgumentInput of given shape can be used with GetArgument(),
   *        representing a batch of scalars
   *
   * @argument should_throw whether this function should throw an error if the shape doesn't match
   * @return true iff the shape is allowed to be used as Argument
   */
  bool CheckScalarArgumentShape(const TensorListShape<> &shape, int batch_size,
                                std::string_view name, bool should_throw = false) const {
    DALI_ENFORCE(is_uniform(shape),
        make_string("Arguments should be passed as uniform TensorLists. Argument \"",
                    name, "\" is not uniform. To access non-uniform argument inputs use "
                    "ArgumentWorkspace::ArgumentInput method directly."));

    bool valid_shape = true;
    for (int i = 0; i < shape.num_samples() && valid_shape; i++) {
      valid_shape = volume(shape[i]) == 1 || shape[i].empty();
    }

    if (should_throw) {
      DALI_ENFORCE(
          valid_shape,
          make_string(
              "Unexpected shape of argument \"", name, "\". Expected batch of ", batch_size,
              " scalars or a batch of tensors containing one element per sample. Got:\n", shape));
    }
    return valid_shape;
  }

  template <typename T, typename S>
  bool TryGetArgumentImpl(T &result,
                                 std::string_view name,
                                 const ArgumentWorkspace *ws,
                                 Index idx) const;

  template <typename T, typename S>
  std::vector<T> GetRepeatedArgumentImpl(std::string_view name) const;

  template <typename S, typename C>
  bool TryGetRepeatedArgumentImpl(C &result, std::string_view name) const;

  string schema_name_;
  const OpSchema *schema_ = nullptr;

  // the list of arguments, in addition order
  std::vector<std::shared_ptr<Argument>> arguments_;
  // maps names to argument indices
  std::map<string, int, std::less<>> argument_idxs_;

  // argument input names and indices in addition order
  std::vector<std::pair<string, int>> argument_inputs_;
  // maps argument names to input indices
  std::map<std::string, int, std::less<>> argument_input_idxs_;

  // Regular arguments that were already set through renamed deprecated arguments
  // Maps regular_argument -> deprecated_argument
  std::map<std::string, std::string, std::less<>> set_through_deprecated_arguments_;

  vector<InOutDeviceDesc> inputs_, outputs_;
  std::map<InOutDeviceDesc, int, std::less<>> output_name_idx_;
};


template <typename T, typename S>
inline T OpSpec::GetArgumentImpl(
      std::string_view name,
      const ArgumentWorkspace *ws,
      Index idx) const {
  // Search for the argument in tensor arguments first
  if (this->HasTensorArgument(name)) {
    if (!ws)
      throw std::logic_error(make_string("Tensor value is unexpected for argument ",
                                         "\"", name, "\" but no workspace was provided."));
    const auto &value = ws->ArgumentInput(name);
    CheckScalarArgumentShape(value.shape(), GetArgument<int>("max_batch_size"), name, true);
    DALI_ENFORCE(IsType<T>(value.type()), make_string(
        "Unexpected type of argument \"", name, "\". Expected ",
        TypeTable::GetTypeName<T>(), " and got ", value.type()));
    return static_cast<T>(value.tensor<T>(idx)[0]);
  }
  // Search for the argument locally
  auto arg_it = argument_idxs_.find(name);
  if (arg_it != argument_idxs_.end()) {
    // Found locally - return
    Argument &arg = *arguments_[arg_it->second];
    return static_cast<T>(arg.Get<S>());
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchemaOrDefault();
    return static_cast<T>(schema.GetDefaultValueForArgument<S>(name));
  }
}

template <typename T, typename S>
inline bool OpSpec::TryGetArgumentImpl(
      T &result,
      std::string_view name,
      const ArgumentWorkspace *ws,
      Index idx) const {
  // Search for the argument in tensor arguments first
  if (this->HasTensorArgument(name)) {
    if (ws == nullptr)
      return false;
    const auto& value = ws->ArgumentInput(name);
    if (!CheckScalarArgumentShape(value.shape(), GetArgument<int>("max_batch_size"), name, false)) {
      return false;
    }
    if (!IsType<T>(value.type()))
      return false;
    result = value.tensor<T>(idx)[0];
    return true;
  }
  // Search for the argument locally
  auto arg_it = argument_idxs_.find(name);
  const OpSchema& schema = GetSchemaOrDefault();
  if (arg_it != argument_idxs_.end()) {
    // Found locally - return
    Argument &arg = *arguments_[arg_it->second];
    if (arg.IsType<S>()) {
      result = static_cast<T>(arg.Get<S>());
      return true;
    }
  } else if (schema.HasArgument(name, true) && schema.HasArgumentDefaultValue(name)) {
    // Argument wasn't present locally, get the default from the associated schema if any
    auto *val = schema.FindDefaultValue(name);
    using VT = const ValueInst<S>;
    if (VT *vt = dynamic_cast<VT *>(val)) {
      result = static_cast<T>(vt->Get());
      return true;
    }
  }
  return false;
}


template <typename T, typename S>
inline std::vector<T> OpSpec::GetRepeatedArgumentImpl(std::string_view name) const {
  using V = std::vector<S>;
  // Search for the argument locally
  auto arg_it = argument_idxs_.find(name);
  if (arg_it != argument_idxs_.end()) {
    // Found locally - return
    Argument &arg = *arguments_[arg_it->second];
    return detail::convert_vector<T>(arg.Get<V>());
  } else {
    // Argument wasn't present locally, get the default from the associated schema
    const OpSchema& schema = GetSchemaOrDefault();
    return detail::convert_vector<T>(schema.GetDefaultValueForArgument<V>(name));
  }
}

template <typename S, typename C>
inline bool OpSpec::TryGetRepeatedArgumentImpl(C &result, std::string_view name) const {
  using V = std::vector<S>;
  // Search for the argument locally
  auto arg_it = argument_idxs_.find(name);
  const OpSchema& schema = GetSchemaOrDefault();
  if (arg_it != argument_idxs_.end()) {
    // Found locally - return
    Argument &arg = *arguments_[arg_it->second];
    if (arg.IsType<V>()) {
      detail::copy_vector(result, arg.Get<V>());
      return true;
    }
  } else if (schema.HasArgument(name, true) && schema.HasArgumentDefaultValue(name)) {
    // Argument wasn't present locally, get the default from the associated schema if any
    auto *val = schema.FindDefaultValue(name);
    using VT = const ValueInst<V>;
    if (VT *vt = dynamic_cast<VT *>(val)) {
      detail::copy_vector(result, vt->Get());
      return true;
    }
  }
  return false;
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OP_SPEC_H_
