// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/dali.pb.h"
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
class OpSpec {
 public:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  using StrPair = std::pair<string, string>;

  inline OpSpec() {}

  /**
   * @brief Returns a full tensor name
   * given its name and device
   */
  static std::string TensorName(std::string name, std::string device) {
    return name + "_" + device;
  }

  /**
   * @brief Constructs a specification for an op with the given name.
   */
  explicit inline OpSpec(const string &name)
    : name_(name) {}

  explicit inline OpSpec(const dali_proto::OpDef& def) {
    name_ = def.name();

    // Extract all the arguments with correct types
    for (auto &arg : def.args()) {
      auto name = arg.name();

      this->AddInitializedArg(name, DeserializeProtobuf(arg));
    }

    for (int i = 0; i < def.input_size(); ++i) {
      if (!def.input(i).is_argument_input()) {
        this->AddInput(def.input(i).name(), def.input(i).device());
      }
    }

    for (int i = 0; i < def.input_size(); ++i) {
      if (def.input(i).is_argument_input()) {
        this->AddArgumentInput(def.input(i).arg_name(), def.input(i).name());
      }
    }

    for (int i = 0; i < def.output_size(); ++i) {
      this->AddOutput(def.output(i).name(), def.output(i).device());
    }
  }

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
    DALI_ENFORCE(arguments_.find(name) == arguments_.end(),
        "AddArg failed. Argument with name \"" + name +
        "\" already exists. ");
    arguments_[name] = arg;
    return *this;
  }

  /**
   * @brief Add an instantiated argument with given name
   */
  inline OpSpec& AddInitializedArg(const string& name, Argument* arg) {
    DALI_ENFORCE(arguments_.find(name) == arguments_.end(),
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
  OpSpec& AddInput(const string &name, const string &device, bool regular_input = true);

  /**
   * @brief Specifies the argument input to the op.
   * Argument inputs are named inputs that are treated as
   * per-iteration arguments. The input may be added only if
   * corresponding argument exists in the schema.
   */
  OpSpec& AddArgumentInput(const string &arg_name, const string &inp_name);

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

  inline int NumArgumentInput() const {
    return argument_inputs_indexes_.size();
  }

  inline int NumRegularInput() const {
    return NumInput() - NumArgumentInput();
  }

  inline int NumOutput() const { return outputs_.size(); }

  inline string Input(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return TensorName(inputs_[idx].first, inputs_[idx].second);
  }

  inline string InputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].first;
  }

  inline string InputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return inputs_[idx].second;
  }

  inline bool IsArgumentInput(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return argument_inputs_indexes_.find(idx) != argument_inputs_indexes_.end();
  }

  inline std::string ArgumentInputName(int idx) const {
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

  inline string Output(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return TensorName(outputs_[idx].first, outputs_[idx].second);
  }

  inline string OutputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].first;
  }

  inline string OutputDevice(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    return outputs_[idx].second;
  }

  inline const std::unordered_map<string, Index>& ArgumentInputs() const {
    return argument_inputs_;
  }

  inline int OutputIdxForName(const string &name, const string &device) {
    auto it = output_name_idx_.find(std::make_pair(name, device));
    DALI_ENFORCE(it != output_name_idx_.end(), "Output with name '" +
        name + "' and device '" + device + "' does not exist.");
    return it->second;
  }

  /**
   * @brief Checks the spec to see if an argument has been specified
   */
  bool HasArgument(const string &name) const {
    auto arg_it = arguments_.find(name);
    return arg_it != arguments_.end();
  }

  /**
   * @brief Checks the spec to see if a tensor argument has been specified
   */
  bool HasTensorArgument(const std::string &name) const {
    auto arg_it = argument_inputs_.find(name);
    return arg_it != argument_inputs_.end();
  }


  /**
   * @brief Lists all arguments specified in this spec.
   */
  std::vector<std::string> ListArguments() const {
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
  inline T GetArgument(const string &name,
                       const ArgumentWorkspace *ws = nullptr,
                       Index idx = 0) const {
    return GetArgument<T, T>(name, ws, idx);
  }

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  inline std::vector<T> GetRepeatedArgument(
      const string &name, const ArgumentWorkspace *ws = nullptr, Index idx = 0) const {
    DALI_ENFORCE(idx == 0, "Tensor arguments cannot be used for vector values");
    return GetArgument<T, std::vector<T>>(name, ws, idx);
  }

  inline StrPair* mutable_input(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    return &inputs_[idx];
  }

  inline StrPair* mutable_output(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
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

  /**
   * @brief Serialize spec to protobuf
   */
  void SerializeToProtobuf(dali_proto::OpDef *op, const string& inst_name) const {
    op->set_name(name());
    op->set_inst_name(inst_name);

    for (size_t i = 0; i < inputs_.size(); ++i) {
      dali_proto::InputOutput *in = op->add_input();
      in->set_name(inputs_[i].first);
      in->set_device(inputs_[i].second);
      if (this->IsArgumentInput(i)) {
         in->set_arg_name(this->ArgumentInputName(i));
      }
      in->set_is_argument_input(this->IsArgumentInput(i));
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      dali_proto::InputOutput *out = op->add_output();
      out->set_name(outputs_[i].first);
      out->set_device(outputs_[i].second);
      out->set_is_argument_input(false);
    }

    for (auto& a : arguments_) {
      // filter out args that need to be dealt with on
      // loading a serialized pipeline
      if (a.first == "batch_size" ||
          a.first == "num_threads" ||
          a.first == "bytes_per_sample_hint") {
        continue;
      }

      dali_proto::Argument *arg = op->add_args();

      a.second->SerializeToProtobuf(arg);
    }
  }

 private:
  template <typename T, typename S>
  inline S GetArgument(const string &name, const ArgumentWorkspace *ws, Index idx) const;

  string name_;
  std::unordered_map<string, Argument*> arguments_;
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
  inline OpSpec& OpSpec::AddArg(const string& name, const T& val) {                             \
    return this->AddArg<int64>(name, static_cast<int64>(val));                                  \
  }                                                                                             \
  template<>                                                                                    \
  inline OpSpec& OpSpec::AddArg(const string& name, const std::vector<T>& val) {                \
    vector<int64> tmp;                                                                          \
    for (auto t : val) {                                                                        \
      tmp.push_back(static_cast<int64>(t));                                                     \
    }                                                                                           \
    Argument * arg = Argument::Store(name, tmp);                                                \
    DALI_ENFORCE(arguments_.find(name) == arguments_.end(),                                     \
        "AddArg failed. Argument with name \"" + name +                                         \
        "\" already exists. ");                                                                 \
    arguments_[name] = arg;                                                                     \
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
INSTANTIATE_ARGUMENT_AS_INT64(DALIImageType);
INSTANTIATE_ARGUMENT_AS_INT64(DALIDataType);
INSTANTIATE_ARGUMENT_AS_INT64(DALIInterpType);
INSTANTIATE_ARGUMENT_AS_INT64(DALITensorLayout);
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OP_SPEC_H_
