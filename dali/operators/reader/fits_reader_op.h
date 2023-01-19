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

#include "dali/operators/reader/loader/fits_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/util/crop_window.h"

namespace dali {

template <typename Backend, typename Target>
class FitsReader : public DataReader<Backend, Target> {
 public:
  explicit FitsReader(const OpSpec& spec) : DataReader<Backend, Target>(spec) {}

  bool CanInferOutputs() const override {
    return true;
  }

  USE_READER_OPERATOR_MEMBERS(Backend, Target);
  using DataReader<Backend, Target>::GetCurrBatchSize;
  using DataReader<Backend, Target>::GetSample;
  using Operator<Backend>::spec_;


  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    DataReader<Backend, Target>::SetupImpl(output_desc, ws);

    int batch_size = GetCurrBatchSize();
    const auto& file_0 = GetSample(0);
    DALIDataType output_type = file_0.get_type();
    int ndim = file_0.get_shape().sample_dim();
    TensorListShape<> sh(batch_size, ndim);

    // TODO: implement checking that all images have same dimensions
    // also all general calculations for all images such as roi

    output_desc.resize(1);
    output_desc[0].shape = std::move(sh);
    output_desc[0].type = output_type;
    return true;
  }
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
