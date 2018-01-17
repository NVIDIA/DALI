// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/op_spec.h"

#include "ndll/pipeline/data/types.h"

namespace ndll {

#define INSTANTIATE_ARGUMENT(T)                                            \
  template<>                                                               \
  OpSpec& OpSpec::AddArg(const string& name, const T& val) {               \
    Argument * arg = Argument::Store(name, val);                           \
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),                \
        "AddArg failed. Argument with name \"" + name +                    \
        "\" already exists. ");                                            \
    arguments_[name] = arg;                                                \
    return *this;                                                          \
  }                                                                        \
  template<>                                                               \
  OpSpec& OpSpec::AddArg(const string& name, const std::vector<T>& val) {  \
    Argument * arg = Argument::Store(name, val);                           \
    NDLL_ENFORCE(arguments_.find(name) == arguments_.end(),                \
        "AddArg failed. Argument with name \"" + name +                    \
        "\" already exists. ");                                            \
    arguments_[name] = arg;                                                \
    return *this;                                                          \
  }                                                                        \
  template<>                                                               \
  T OpSpec::GetArgument(const string& name, const T& default_value) const; \
  template<>                                                               \
  std::vector<T> OpSpec::GetArgument(const string& name,                   \
      const std::vector<T>& default_value) const;

INSTANTIATE_ARGUMENT(float);
INSTANTIATE_ARGUMENT(int64);
INSTANTIATE_ARGUMENT(bool);
INSTANTIATE_ARGUMENT(NDLLImageType);
INSTANTIATE_ARGUMENT(NDLLInterpType);
INSTANTIATE_ARGUMENT(NDLLDataType);
INSTANTIATE_ARGUMENT(std::string);

OpSpec& OpSpec::AddArg(const string& name, int val) {
  return this->AddArg<int64>(name, val);
}
OpSpec& OpSpec::AddArg(const string& name, unsigned int val) {
  return this->AddArg<int64>(name, val);
}
OpSpec& OpSpec::AddArg(const string& name, long val) {
  return this->AddArg<int64>(name, val);
}
OpSpec& OpSpec::AddArg(const string& name, unsigned long val) {
  return this->AddArg<int64>(name, val);
}

OpSpec& OpSpec::AddInput(const string &name, const string &device) {
  NDLL_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for input \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  StrPair name_device_pair = std::make_pair(name, device);
  NDLL_ENFORCE(input_name_idx_.count(name_device_pair) == 0,
      "Input '" + name + "' with device '" + device + "' "
      "already added to OpSpec");

  inputs_.push_back(std::make_pair(name, device));
  auto ret = input_name_idx_.insert({name_device_pair, inputs_.size()-1});
  NDLL_ENFORCE(ret.second, "Input name/device insertion failed.");
  return *this;
}

OpSpec& OpSpec::AddOutput(const string &name, const string &device) {
  NDLL_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for output \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  StrPair name_device_pair = std::make_pair(name, device);
  NDLL_ENFORCE(output_name_idx_.count(name_device_pair) == 0,
      "Output '" + name + "' with device '" + device + "' "
      "already added to OpSpec");

  outputs_.push_back(std::make_pair(name, device));
  auto ret = output_name_idx_.insert({name_device_pair, outputs_.size()-1});
  NDLL_ENFORCE(ret.second, "Output name/device insertion failed.");
  return *this;
}

}  // namespace ndll
