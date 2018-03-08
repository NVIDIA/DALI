// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/op_spec.h"

#include "ndll/pipeline/data/types.h"

namespace ndll {

OpSpec& OpSpec::AddInput(const string &name, const string &device) {
  NDLL_ENFORCE(device == "gpu" || device == "cpu", "Invalid device "
      "specifier \"" + device + "\" for input \"" + name + "\". "
      "Valid options are \"cpu\" or \"gpu\"");
  StrPair name_device_pair = std::make_pair(name, device);

  inputs_.push_back(std::make_pair(name, device));
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
