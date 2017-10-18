#ifndef NDLL_PIPELINE_OP_SPEC_H_
#define NDLL_PIPELINE_OP_SPEC_H_

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

  /**
   * @brief Constructs a specification for an op with the given name.
   */
  OpSpec(const string &name)
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
   * @brief Specifies the name of an extra (in addition to the 
   * batch of data being processed) input Tensor that the op
   * requires.
   */
  OpSpec& AddExtraInput(const string &name);

  /**
   * @brief Specified the name of an extra (in addition to the 
   * batch of data being processed) output Tensor that the op 
   * will produce
   */
  OpSpec& AddExtraOutput(const string &name);

  /**
   * @brief Specified the name of an extra (in addition to the 
   * batch of data being processed) input GPU Tensor that the 
   * op will produce
   */
  OpSpec& AddExtraGPUInput(const string &name);

  /**
   * @brief Specified the name of an extra (in addition to the 
   * batch of data being processed) output GPU Tensor that the 
   * op will produce
   */
  OpSpec& AddExtraGPUOutput(const string &name);

  /**
   * @brief Checks the Spec for an argument with the given name/type.
   * Returns the default if an argument with the given name/type does
   * not exist.
   */
  template <typename T>
  T GetSingleArgument(const string &name, const T &default_value) const;

  /**
   * @brief Checks the Spec for a repeated argument of the given name/type.
   * Returns the default if an argument with the given name does not exist.
   */
  template <typename T>
  vector<T> GetRepeatedArgument(const string &name,
      const vector<T> &default_value = {}) const;
  
  /**
   * @brief Returns the extra input with the given index. Tensors are
   * indexed in the order that they are added to the OpSpec. 
   *
   * This function is Used by the Operators to get their extra Tensor 
   * inputs that have been set by the Pipeline.
   */
  TensorPtr<CPUBackend> ExtraInput(int index) const;

  /**
   * @brief Returns the extra output with the given index. Tensors are
   * indexed in the order that they are added to the OpSpec. 
   *
   * This function is Used by the Operators to get their extra Tensor 
   * outputs that have been set by the Pipeline.
   */
  TensorPtr<CPUBackend> ExtraOutput(int index) const;

  /**
   * @brief Returns the extra GPU input with the given index. Tensors 
   * are indexed in the order that they are added to the OpSpec. 
   *
   * This function is Used by the Operators to get their extra Tensor 
   * GPU inputs that have been set by the Pipeline.
   */
  TensorPtr<GPUBackend> ExtraGPUInput(int index) const;

  /**
   * @brief Returns the extra GPU output with the given index. Tensors 
   * are indexed in the order that they are added to the OpSpec. 
   *
   * This function is Used by the Operators to get their extra Tensor 
   * GPU outputs that have been set by the Pipeline.
   */
  TensorPtr<GPUBackend> ExtraGPUOutput(int index) const;
  
  /**
   * @brief Used internally by the Pipeline to set pointers to the
   * constructed Tensor objects that are to be used by the op.
   */
  void AddExtraInputTensor(TensorPtr<CPUBackend> tensor);

  /**
   * @copydoc AddExtraInputTensor(TensorPtr<CPUBackend> tensor)
   */
  void AddExtraOutputTensor(TensorPtr<CPUBackend> tensor);

  /**
   * @copydoc AddExtraInputTensor(TensorPtr<CPUBackend> tensor)
   */
  void AddExtraInputTensor(TensorPtr<GPUBackend> tensor);

  /**
   * @copydoc AddExtraInputTensor(TensorPtr<CPUBackend> tensor)
   */
  void AddExtraOutputTensor(TensorPtr<GPUBackend> tensor);

  /**
   * @brief Returns a vector of names of the extra input Tensors that
   * have been added to the OpSpec.
   */
  inline const vector<string>& ExtraInputNames() const {
    return extra_input_names_;
  }

  /**
   * @brief Returns a vector of names of the extra output Tensors that
   * have been added to the OpSpec.
   */
  inline const vector<string>& ExtraOutputNames() const {
    return extra_output_names_;
  }

  /**
   * @brief Returns a vector of names of the extra gpu input Tensors 
   * that have been added to the OpSpec.
   */
  inline const vector<string>& ExtraGPUInputNames() const {
    return gpu_extra_input_names_;
  }

  /**
   * @brief Returns a vector of names of the extra gpu output Tensors 
   * that have been added to the OpSpec.
   */
  inline const vector<string>& ExtraGPUOutputNames() const {
    return gpu_extra_output_names_;
  }
  
private:
  // Helper function to handle argument types. Checks the correct type
  // field in the input argument. If it is not set, return the default
  template <typename T>
  T ArgumentTypeHelper(const Argument &arg, const T &default_value) const;
    
  string name_;
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
