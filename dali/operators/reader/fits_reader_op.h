// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
class FitsReader : public DataReader<Backend, Target, Target, true> {
 public:
  explicit FitsReader(const OpSpec& spec) : DataReader<Backend, Target, Target, true>(spec) {}

  bool CanInferOutputs() const override {
    return true;
  }

  // TODO(skarpinski) Debug fits reader and add checkpointing support
  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    DALI_FAIL("Fits reader does not support checkpointing.");
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    DALI_FAIL("Fits reader does not support checkpointing.");
  }

  USE_READER_OPERATOR_MEMBERS(Backend, Target, Target, true);
  using DataReader<Backend, Target, Target, true>::GetCurrBatchSize;
  using DataReader<Backend, Target, Target, true>::GetSample;
  using Operator<Backend>::spec_;

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    DataReader<Backend, Target, Target, true>::SetupImpl(output_desc, ws);

    int num_outputs = ws.NumOutput();
    int num_samples = GetCurrBatchSize();             // samples here are synonymous with files
    vector<int> ndims(num_outputs);                   // to store dimensions of each output
    vector<DALIDataType> output_dtypes(num_outputs);  // to store dtype of each output

    output_desc.resize(num_outputs);
    const auto& sample_0 = GetSample(0);

    for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
      ndims[output_idx] = sample_0.header[output_idx].shape.sample_dim();
      output_dtypes[output_idx] = sample_0.header[output_idx].type();
      output_desc[output_idx].shape = TensorListShape<>(num_samples, ndims[output_idx]);
    }

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& sample = GetSample(sample_idx);
      // here we don't validate current data dims and type, only the final dimensions and type
      // declared by a header, since in case of GPU decompression changes to dimensions will occur
      // on in runImpl
      for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
        DALI_ENFORCE(
            sample.header[output_idx].shape.sample_dim() == ndims[output_idx],
            make_string("Inconsistent data: All samples in the batch must have the same number of "
                        "dimensions for outputs with the same idx. "
                        "Got \"",
                        sample_0.filename, "\" with ", ndims[output_idx], " dimensions and \"",
                        sample.filename, "\" with ", sample.data[output_idx].shape().sample_dim(),
                        " dimensions on output with idx = ", output_idx));

        DALI_ENFORCE(
            sample.header[output_idx].type() == output_dtypes[output_idx],
            make_string("Inconsistent data: All samples in the batch must have the same data type "
                        "for outputs with the same idx. "
                        "Got \"",
                        sample_0.filename, "\" with data type ", output_dtypes[output_idx],
                        " and \"", sample.filename, "\" with data type ",
                        sample.header[output_idx].type()));

        output_desc[output_idx].shape.set_tensor_shape(sample_idx, sample.header[output_idx].shape);
        output_desc[output_idx].type = sample.header[output_idx].type();
      }
    }

    return true;
  }
};

class FitsReaderCPU : public FitsReader<CPUBackend, FitsFileWrapper> {
 public:
  explicit FitsReaderCPU(const OpSpec& spec) : FitsReader<CPUBackend, FitsFileWrapper>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FitsLoaderCPU>(spec, shuffle_after_epoch);
  }

 protected:
  void RunImpl(Workspace& ws) override;
  using Operator<CPUBackend>::RunImpl;

 private:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, FitsFileWrapper, FitsFileWrapper, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_FITS_READER_OP_H_
