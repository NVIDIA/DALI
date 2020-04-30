// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_NUMPY_READER_OP_H_
#define DALI_OPERATORS_READER_NUMPY_READER_OP_H_

#include <utility>
#include <string>
#include <vector>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/numpy_loader.h"

namespace dali {

#define NUMPY_ALLOWED_DIMS (1, 2, 3, 4, 5)

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)


class NumpyReader : public DataReader<CPUBackend, ImageFileWrapper > {
 public:
  explicit NumpyReader(const OpSpec& spec)
    : DataReader< CPUBackend, ImageFileWrapper >(spec) {
    bool shuffle_after_epoch = spec_.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<NumpyLoader>(spec, std::vector<string>(),
                                      shuffle_after_epoch);
    // check if there are static args
    GetStaticSliceArg(slice_anchors_, "anchor");
    GetStaticSliceArg(slice_shapes_, "shape");
    SanitizeSliceArgs(slice_anchors_, slice_shapes_);
  }

  // we need to override this, because we want to allow for sliced reads
  void Prefetch() override;

  // read the sample
  void Run(HostWorkspace &ws) override;

  // actual read implementation for a single sample
  void RunImpl(SampleWorkspace &ws) override;

 protected:
  // helper functions for extracting the slices statically or dynamically
  // those are defined so that it is easier to deal with combined static + dynamic
  // arguments. For example, a typical use case would be to define a static slice_shape
  // but a random anchor point.
  void GetStaticSliceArg(TensorListShape<>& tls, const char *name);
  void GetDynamicSliceArg(TensorListShape<>& tls, ArgumentWorkspace &ws, const char *name);
  void SanitizeSliceArgs(TensorListShape<>& anchor, const TensorListShape<>& shape);

  // used for copying the image to output buffer
  void CopyHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input);

  // used for transposing output in case of column-major order
  void TransposeHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input);

  // slicing helpers
  TensorListShape<> slice_anchors_;
  TensorListShape<> slice_shapes_;

  // other parameters
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageFileWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_OP_H_
