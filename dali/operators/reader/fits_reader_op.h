// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_FITS_READER_OP_H_
#define DALI_OPERATORS_READER_FITS_READER_OP_H_

#include <string>
#include <utility>
#include <vector>

#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/reader/loader/fits_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/util/crop_window.h"

namespace dali {

template <typename Backend, typename Target>
class FitsReader : public DataReader<Backend, Target> {
 public:
  explicit FitsReader(const OpSpec& spec)
      : DataReader<Backend, Target>(spec), slice_attr_(spec, nullptr) {}

  bool CanInferOutputs() const override {
    return true;
  }

  USE_READER_OPERATOR_MEMBERS(Backend, Target);

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    // here should go  checking if all images have same dims
    DataReader<Backend, Target>::SetupImpl(output_desc, ws);
    return true;
  }

  NamedSliceAttr slice_attr_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;
  float fill_value_ = 0;

  std::vector<bool> need_transpose_;
  std::vector<bool> need_slice_;
};

class FitsReaderCPU : public FitsReader<CPUBackend, FitsFileWrapper> {
 public:
  explicit FitsReaderCPU(const OpSpec& spec) : FitsReader<CPUBackend, FitsFileWrapper>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FitsLoader>(spec, shuffle_after_epoch);
  }

 protected:
  void RunImpl(Workspace& ws) override;
  using Operator<CPUBackend>::RunImpl;

 private:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, FitsFileWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_OP_H_
