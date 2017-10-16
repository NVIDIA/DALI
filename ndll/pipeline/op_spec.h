#ifndef NDLL_PIPELINE_OP_SPEC_H_
#define NDLL_PIPELINE_OP_SPEC_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/argument.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

/**
 * @brief Defines all parameters needed to construct an Operator, including the op name,
 * any additional input and output tensors it may need, and any number of additional
 * arguments.
 */
class OpSpec {
public:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  
  OpSpec(const string &name, const string &stage = "Prefetch")
    : name_(name), stage_(stage) {}

  /**
   * @brief Getter for the name of the Operator.
   */
  inline const string& name() const { return name_; }
  
  /**
   * @brief Getter for the stage the Operator is to be run in.
   */
  inline const string &stage() const { return stage_; }
  
  // TODO(tgale): Make this a template and update instantiation macros
  OpSpec& AddArg(const string &name, const double &d);
  OpSpec& AddArg(const string &name, const float &f);
  OpSpec& AddArg(const string &name, const int64 &i);
  OpSpec& AddArg(const string &name, const int &i);
  OpSpec& AddArg(const string &name, const bool &b);
  OpSpec& AddArg(const string &name, const NDLLImageType &type);
  OpSpec& AddArg(const string &name, const NDLLInterpType &type);
  OpSpec& AddArg(const string &name, const NDLLDataType &type);
  OpSpec& AddArg(const string &name, const uint64 &i);
  OpSpec& AddArg(const string &name, const string &s);

  OpSpec& AddArg(const string &name, const vector<double> &d);
  OpSpec& AddArg(const string &name, const vector<float> &f);
  OpSpec& AddArg(const string &name, const vector<int64> &i);
  OpSpec& AddArg(const string &name, const vector<int> &i);
  OpSpec& AddArg(const string &name, const vector<bool> &b);
  OpSpec& AddArg(const string &name, const vector<NDLLImageType> &type);
  OpSpec& AddArg(const string &name, const vector<NDLLInterpType> &type);
  OpSpec& AddArg(const string &name, const vector<NDLLDataType> &type);
  OpSpec& AddArg(const string &name, const vector<uint64> &i);
  OpSpec& AddArg(const string &name, const vector<string> &s);

  OpSpec& AddExtraInput(const string &name);
  OpSpec& AddExtraOutput(const string &name);
  OpSpec& AddExtraGPUInput(const string &name);
  OpSpec& AddExtraGPUOutput(const string &name);

  // For accessing single arguments
  template <typename T>
  T GetSingleArgument(const string &name, const T &default_value) const;

  // For accessing repeated arguments
  template <typename T>
  vector<T> GetRepeatedArgument(const string &name,
      const vector<T> &default_value = {}) const;
  
  // For use by the Operators to get the extra Tensors
  TensorPtr<CPUBackend> ExtraInput(int index) const;
  TensorPtr<CPUBackend> ExtraOutput(int index) const;
  TensorPtr<GPUBackend> ExtraGPUInput(int index) const;
  TensorPtr<GPUBackend> ExtraGPUOutput(int index) const;
  
  // For use of the Pipeline to set the actual Tensors
  void AddExtraInputTensor(TensorPtr<CPUBackend> tensor);
  void AddExtraOutputTensor(TensorPtr<CPUBackend> tensor);
  void AddExtraInputTensor(TensorPtr<GPUBackend> tensor);
  void AddExtraOutputTensor(TensorPtr<GPUBackend> tensor);

  // For use of the Pipeline to get the names of the Tensors
  inline const vector<string>& ExtraInputNames() const {
    return extra_input_names_;
  }
  inline const vector<string>& ExtraOutputNames() const {
    return extra_output_names_;
  }
  inline const vector<string>& ExtraGPUInputNames() const {
    return gpu_extra_input_names_;
  }
  inline const vector<string>& ExtraGPUOutputNames() const {
    return gpu_extra_output_names_;
  }
  
private:
  // Helper function to handle argument types. Checks the correct type
  // field in the input argument. If it is not set, return the default
  template <typename T>
  T ArgumentTypeHelper(const Argument &arg, const T &default_value) const;
    
  string name_;
  string stage_;
  std::unordered_map<string, Argument> arguments_;

  // To store all added tensor names
  vector<string> extra_input_names_;
  vector<string> extra_output_names_;
  vector<string> gpu_extra_input_names_;
  vector<string> gpu_extra_output_names_;
  
  vector<TensorPtr<CPUBackend>> extra_inputs_;
  vector<TensorPtr<CPUBackend>> extra_outputs_;
  vector<TensorPtr<GPUBackend>> gpu_extra_inputs_;
  vector<TensorPtr<GPUBackend>> gpu_extra_outputs_;
};

template <typename T>
T OpSpec::GetSingleArgument(const string &name, const T &default_value) const {
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
