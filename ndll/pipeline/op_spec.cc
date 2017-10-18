#include "ndll/pipeline/op_spec.h"

#include "ndll/common.h"
#include "ndll/pipeline/data/types.h"

namespace ndll {

#define INSTANTIATE_ADD_SINGLE_ARGUMENT(T, fieldname)                   \
  template <>                                                           \
  OpSpec& OpSpec::AddArg(const string &name, const T &val) {            \
    Argument arg;                                                       \
    arg.set_name(name);                                                 \
    arg.set_##fieldname(val);                                           \
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),             \
        "AddArg failed. Argument with name \"" + name +                 \
        "\" already exists.");                                          \
    arguments_[name] = arg;                                             \
    return *this;                                                       \
  }

INSTANTIATE_ADD_SINGLE_ARGUMENT(double, f);
INSTANTIATE_ADD_SINGLE_ARGUMENT(float, f);
INSTANTIATE_ADD_SINGLE_ARGUMENT(int64, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(int, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(bool, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(NDLLImageType, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(NDLLInterpType, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(NDLLDataType, i);
INSTANTIATE_ADD_SINGLE_ARGUMENT(uint64, ui);
INSTANTIATE_ADD_SINGLE_ARGUMENT(string, s);

#define INSTANTIATE_ADD_REPEATED_ARGUMENT(T, fieldname)                 \
  template <>                                                           \
  OpSpec& OpSpec::AddArg(const string &name, const vector<T> &v) {      \
    Argument arg;                                                       \
    arg.set_name(name);                                                 \
    for (const auto &val : v) {                                         \
      arg.add_##fieldname(val);                                         \
    }                                                                   \
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),             \
        "AddArg failed. Argument with name \"" + name +                 \
        "\" already exists. ");                                         \
    arguments_[name] = arg;                                             \
    return *this;                                                       \
  }

INSTANTIATE_ADD_REPEATED_ARGUMENT(double, rf);
INSTANTIATE_ADD_REPEATED_ARGUMENT(float, rf);
INSTANTIATE_ADD_REPEATED_ARGUMENT(int64, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(int, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(bool, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(NDLLImageType, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(NDLLInterpType, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(NDLLDataType, ri);
INSTANTIATE_ADD_REPEATED_ARGUMENT(uint64, rui);
INSTANTIATE_ADD_REPEATED_ARGUMENT(string, rs);

OpSpec& OpSpec::AddExtraInput(const string &name) {
  for (const auto &s : extra_input_names_) {
    NDLL_ENFORCE(s != name,
        "AddExtraInput failed. Input with name\"" + name + "\" already exists.");
  }
  extra_input_names_.push_back(name);
  return *this;
}

OpSpec& OpSpec::AddExtraOutput(const string &name) {
  for (const auto &s : extra_output_names_) {
    NDLL_ENFORCE(s != name,
        "AddExtraOutput failed. Output with name\"" + name + "\" already exists.");
  }
  extra_output_names_.push_back(name);
  return *this;
}

OpSpec& OpSpec::AddExtraGPUInput(const string &name) {
  for (const auto &s : gpu_extra_input_names_) {
    NDLL_ENFORCE(s != name,
        "AddExtraGPUInput failed. Input with name\"" + name + "\" already exists.");
  }
  gpu_extra_input_names_.push_back(name);
  return *this;
}

OpSpec& OpSpec::AddExtraGPUOutput(const string &name) {
  for (const auto &s : gpu_extra_output_names_) {
    NDLL_ENFORCE(s != name,
        "AddExtraGPUOutput failed. Output with name\"" + name + "\" already exists.");
  }
  gpu_extra_output_names_.push_back(name);
  return *this;
}

void OpSpec::AddExtraInputTensor(TensorPtr<CPUBackend> tensor) {
  extra_inputs_.push_back(tensor);
}

void OpSpec::AddExtraOutputTensor(TensorPtr<CPUBackend> tensor) {
  extra_outputs_.push_back(tensor);
}

void OpSpec::AddExtraInputTensor(TensorPtr<GPUBackend> tensor) {
  gpu_extra_inputs_.push_back(tensor);
}

void OpSpec::AddExtraOutputTensor(TensorPtr<GPUBackend> tensor) {
  gpu_extra_outputs_.push_back(tensor);
}

OpSpec::TensorPtr<CPUBackend> OpSpec::ExtraInput(int index) const {
  NDLL_ENFORCE((index >= 0) && ((size_t)index < extra_inputs_.size()),
      "Input out of range. Did you add the operator with the "
      "correct number of extra input tenors?");
  return extra_inputs_[index];
}

OpSpec::TensorPtr<CPUBackend> OpSpec::ExtraOutput(int index) const {
  NDLL_ENFORCE((index >= 0) && ((size_t)index < extra_outputs_.size()),
      "Output out of range. Did you add the operator with the "
      "correct number of extra output tenors?");
  return extra_outputs_[index];
}

OpSpec::TensorPtr<GPUBackend> OpSpec::ExtraGPUInput(int index) const {
  NDLL_ENFORCE((index >= 0) && ((size_t)index < gpu_extra_inputs_.size()),
      "Input out of range. Did you add the operator with the "
      "correct number of extra input tenors?");
  return gpu_extra_inputs_[index];
}

OpSpec::TensorPtr<GPUBackend> OpSpec::ExtraGPUOutput(int index) const {
  NDLL_ENFORCE((index >= 0) && ((size_t)index < gpu_extra_outputs_.size()),
      "Output out of range. Did you add the operator with the "
      "correct number of extra output tenors?");
  return gpu_extra_outputs_[index];
}

#define INSTANTIATE_SINGLE_ARGUMENT_HELPER(T, fieldname)                \
  template <>                                                           \
  T OpSpec::ArgumentTypeHelper(const Argument &arg,                     \
      const T &default_value) const {                                   \
    if (arg.has_##fieldname()) return (T)arg.get_##fieldname();         \
    return default_value;                                               \
  }

INSTANTIATE_SINGLE_ARGUMENT_HELPER(double, f);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(float, f);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(int64, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(int, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(bool, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(NDLLImageType, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(NDLLInterpType, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(NDLLDataType, i);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(uint64, ui);
INSTANTIATE_SINGLE_ARGUMENT_HELPER(string, s);

// For repeated arguments, we do not return the default value unless the
// argument name is not set. If the argument name exists, but the repeated
// field has not been set by the user, we still return the repeated field.
// Thus, if the user sets an arg with the right name but the wrong type,
// The operator will receive "" when it queries for the arg with the type
// if was expecting.
#define INSTANTIATE_REPEATED_ARGUMENT_HELPER(T, fieldname)          \
  template <>                                                       \
  vector<T> OpSpec::ArgumentTypeHelper(const Argument &arg,         \
      const vector<T> &default_value) const {                       \
    vector<T> tmp;                                                  \
    for (const auto &val : arg.fieldname()) tmp.push_back((T)val);  \
    return tmp;                                                     \
  }

INSTANTIATE_REPEATED_ARGUMENT_HELPER(double, rf);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(float, rf);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(int64, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(int, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(bool, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(NDLLImageType, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(NDLLInterpType, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(NDLLDataType, ri);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(uint64, rui);
INSTANTIATE_REPEATED_ARGUMENT_HELPER(string, rs);
  
} // namespace ndll
