#ifndef NDLL_PIPELINE_OP_SPEC_H_
#define NDLL_PIPELINE_OP_SPEC_H_

#include <map>
#include <utility>

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
  inline OpSpec(const string &name)
    : name_(name) {}

  /**
   * @brief Getter for the name of the Operator.
   */
  inline const string& name() const { return name_; }

  /**
   * @brief Add an argument with the given name and value.
   */
  template <typename T>
  OpSpec& AddArg(const string &name, const T &val);

  // Forward to string implementation
  template <unsigned N>
  OpSpec& AddArg(const string &name, const char (&c_str)[N]) {
    return this->AddArg<string>(name, c_str);
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
   * @brief Checks the Spec for an argument with the given name/type.
   * Returns the default if an argument with the given name/type does
   * not exist.
   */
  template <typename T>
  T GetArgument(const string &name, const T &default_value) const;

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  vector<T> GetRepeatedArgument(const string &name,
      const vector<T> &default_value = {}) const;

  inline StrPair* mutable_input(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, NumInput());
    return &inputs_[idx];
  }

  inline StrPair* mutable_output(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, NumOutput());
    return &outputs_[idx];
  }
  
private:
  // Helper function to handle argument types. Checks the correct type
  // field in the input argument. If it is not set, return the default
  template <typename T>
  T ArgumentTypeHelper(const Argument &arg, const T &default_value) const;
  
  string name_;
  std::unordered_map<string, Argument> arguments_;

  std::map<StrPair, int> input_name_idx_, output_name_idx_;
  vector<StrPair> inputs_, outputs_;
};

template <typename T>
T OpSpec::GetArgument(const string &name, const T &default_value) const {
  // Search for the argument by name
  auto arg_it = arguments_.find(name);
  
  if (arg_it == arguments_.end()) {
    return default_value;
  }

  return ArgumentTypeHelper<T>(arg_it->second, default_value);
}

template <typename T>
vector<T> OpSpec::GetRepeatedArgument(const string &name,
    const vector<T> &default_value) const {
  // Search for the argument by name
  auto arg_it = arguments_.find(name);

  if (arg_it == arguments_.end()) {
    return default_value;
  }

  return ArgumentTypeHelper<vector<T>>(arg_it->second, default_value);
}

} // namespace ndll

#endif // NDLL_PIPELINE_OP_SPEC_H_
