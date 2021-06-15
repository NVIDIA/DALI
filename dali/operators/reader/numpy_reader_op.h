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

template <typename Backend, typename Target>
class NumpyReader : public DataReader<Backend, Target> {
 public:
  explicit NumpyReader(const OpSpec& spec)
      : DataReader<Backend, Target>(spec),
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

  USE_READER_OPERATOR_MEMBERS(Backend, Target);
  using DataReader<Backend, Target>::GetCurrBatchSize;
  using DataReader<Backend, Target>::GetSample;

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    DataReader<Backend, Target>::SetupImpl(output_desc, ws);

    int batch_size = GetCurrBatchSize();
    const auto& file_0 = GetSample(0);
    TypeInfo output_type = file_0.data.type();
    int ndim = file_0.data.shape().sample_dim();
    TensorListShape<> sh(batch_size, ndim);

    bool need_slice = slice_attr_.template ProcessArguments<Backend>(ws, batch_size, ndim);

    rois_.clear();
    if (need_slice)
      rois_.resize(batch_size);

    for (int i = 0; i < batch_size; i++) {
      const auto& file_i = GetSample(i);
      const auto& file_sh = file_i.data.shape();
      auto sample_sh = sh.tensor_shape_span(i);

      DALI_ENFORCE(
          file_i.data.shape().sample_dim() == ndim,
          make_string("Inconsistent data: All samples in the batch must have the same number of "
                      "dimensions. "
                      "Got \"",
                      file_0.filename, "\" with ", ndim, " dimensions and \"", file_i.filename,
                      "\" with ", file_i.data.shape().sample_dim(), " dimensions"));
      DALI_ENFORCE(
          file_i.data.type().id() == output_type.id(),
          make_string("Inconsistent data: All samples in the batch must have the same data type. "
                      "Got \"",
                      file_0.filename, "\" with data type ", output_type.id(), " and \"",
                      file_i.filename, "\" with data type ", file_i.data.type().id()));

      bool is_transposed = file_i.fortan_order;
      // Calculate the full transposed shape first
      if (is_transposed) {
        for (int d = 0; d < ndim; d++)
          sample_sh[d] = file_sh[ndim - 1 - d];
      } else {
        for (int d = 0; d < ndim; d++)
          sample_sh[d] = file_sh[d];
      }

      if (need_slice) {
        // Calculate the cropping window, based on the final layout (user provides axes in that
        // layout)
        auto full_sample_sh = sh.tensor_shape(i);  // already permuted dims
        auto tmp_roi = slice_attr_.GetCropWindowGenerator(i)(full_sample_sh, {});
        ApplySliceBoundsPolicy(out_of_bounds_policy_, full_sample_sh, tmp_roi.anchor,
                               tmp_roi.shape);
        sh.set_tensor_shape(i, tmp_roi.shape);  // set the final shape

        // Reverse the cropping window arguments if needed, as we provide slice arguments in the
        // original layout
        auto& roi = rois_[i];
        if (is_transposed) {
          for (int d = 0; d < ndim; d++) {
            roi.anchor[d] = tmp_roi.anchor[ndim - 1 - d];
            roi.shape[d] = tmp_roi.shape[ndim - 1 - d];
          }
        } else {
          roi = std::move(tmp_roi);
        }
      }
    }
    output_desc.resize(1);
    output_desc[0].shape = std::move(sh);
    output_desc[0].type = output_type;
    return true;
  }

 protected:
  NamedSliceAttr slice_attr_;
  std::vector<CropWindow> rois_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;
  float fill_value_ = 0;
};

class NumpyReaderCPU : public NumpyReader<CPUBackend, NumpyFileWrapper> {
 public:
  explicit NumpyReaderCPU(const OpSpec& spec) : NumpyReader<CPUBackend, NumpyFileWrapper>(spec) {}

  void RunImpl(HostWorkspace &ws) override;

 protected:
  void TransposeHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input);
  void SliceHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input,
                   const CropWindow& roi, float fill_value = 0);
  void SlicePermuteHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input,
                          const CropWindow& roi, float fill_value = 0);

  USE_READER_OPERATOR_MEMBERS(CPUBackend, NumpyFileWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_OP_H_
