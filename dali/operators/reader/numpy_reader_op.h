// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/util/crop_window.h"


namespace dali {

class NumpyReader : public DataReader<CPUBackend, ImageFileWrapper > {
 public:
  explicit NumpyReader(const OpSpec& spec)
      : DataReader<CPUBackend, ImageFileWrapper>(spec),
        slice_attr_(spec, "roi_start", "rel_roi_start", "roi_end", "rel_roi_end", "roi_shape",
                    "rel_roi_shape", "roi_axes", nullptr) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<NumpyLoader>(spec, shuffle_after_epoch);
    out_of_bounds_policy_ = GetOutOfBoundsPolicy(spec);
    if (out_of_bounds_policy_ == OutOfBoundsPolicy::Pad) {
      fill_value_ = spec.GetArgument<float>("fill_value");
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(HostWorkspace &ws) override;

 protected:
  void TransposeHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input);
  void SliceHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input,
                   const CropWindow& roi, float fill_value = 0);
  void SlicePermuteHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input,
                          const CropWindow& roi, float fill_value = 0);
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageFileWrapper);

 private:
  NamedSliceAttr slice_attr_;
  std::vector<CropWindow> rois_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;
  float fill_value_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_OP_H_
