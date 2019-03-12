// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_H_

#include <utility>
#include <vector>
#include <tuple>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/crop/crop_attr.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Crop : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline Crop(const OpSpec &spec)
    : Operator<Backend>(spec)
    , CropAttr(spec)
    , C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});
    output_offsets_.Resize({batch_size_});
    Init(batch_size_);
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  // TODO(klecki) do not mix returning by pointer and return type
  inline Dims GetOutShape(DALITensorLayout inputLayout, DALITensorLayout *pOutLayout, int dataIdx) {
    *pOutLayout = output_layout_ == DALI_SAME ? inputLayout : output_layout_;
    if (*pOutLayout == DALI_NCHW)
      return {C_, crop_height_[dataIdx], crop_width_[dataIdx]};
    else
      return {crop_height_[dataIdx], crop_width_[dataIdx], C_};
  }

  template <typename Out>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  // Invoke CalcOutputSize from CropKernel
  template <typename Kernel>
  void AllocateOutput(const Tensor<CPUBackend> &input, typename Kernel::KernelAttributes args,
                      Tensor<CPUBackend> &output) {
    auto in_shape = detail::ToStaticShape<Kernel::input_dim>(input.shape());
    auto out_shape = Kernel::CalcOutputSize(in_shape, args);
    output.Resize(detail::ToDynamicShape(out_shape));
  }

  // Invoke Run from CropKernel
  template <typename Kernel>
  void RunKernel(const Tensor<CPUBackend> &input, typename Kernel::KernelAttributes args,
                 Tensor<CPUBackend> &output) {
    // TODO(klecki) - Input and output allocations should already be hanlded at this stage.

    const auto *in = input.template data<typename Kernel::InputType>();
    auto in_shape = detail::ToStaticShape<Kernel::input_dim>(input.shape());
    auto *out = output.template mutable_data<typename Kernel::OutputType>();
    auto out_shape = detail::ToStaticShape<Kernel::output_dim>(output.shape());
    Kernel::Run(in, in_shape, args, out, out_shape);
  }

  template <typename Kernel>
  void AllocateAndRunKernel(Workspace<CPUBackend> *ws, const int idx) {
    const auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);
    const int dataIdx = ws->data_idx();
    const int threadIdx = ws->thread_idx();
    const int h_start = per_sample_crop_[threadIdx].first;
    const int w_start = per_sample_crop_[threadIdx].second;
    const int crop_height = crop_height_[dataIdx];
    const int crop_width = crop_width_[dataIdx];

    typename Kernel::KernelAttributes args{h_start, w_start, crop_height, crop_width};
    AllocateOutput<Kernel>(input, args, output);
    RunKernel<Kernel>(input, args, output);
  }

  Tensor<CPUBackend> input_ptrs_, input_strides_, output_offsets_;
  Tensor<GPUBackend> input_ptrs_gpu_, input_strides_gpu_, output_offsets_gpu_;
  Tensor<GPUBackend> crop_width_gpu_, crop_height_gpu_;
  vector<int> crop_offsets_;

  // Crop starting position (in input)
  std::vector<std::pair<int, int>> per_sample_crop_;
  // Input dims
  std::vector<std::pair<int, int>> per_sample_dimensions_;

  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;

  const int C_;

  USE_OPERATOR_MEMBERS();

 private:
  void DataDependentSetup(Workspace<Backend> *ws, int idx);

  inline static std::tuple<Index, Index, Index> GetHWC(const vector<Index> &shape) {
    if (shape.size() == 3) {
      return std::make_tuple(shape[0], shape[1], shape[2]);
    } else if (shape.size() == 4) {
      return std::make_tuple(shape[1], shape[2], shape[3]);
    }
    DALI_FAIL("Expected 3-dimensional or 4-dimensional input");
  }

  void SetupSharedSampleParams(const ArgumentWorkspace *ws,
                               const vector<Index> &inputShape, int threadIdx,
                               int dataIdx) {
    Index H, W, C;
    std::tie(H, W, C) = GetHWC(inputShape);

    per_sample_dimensions_[threadIdx] = std::make_pair(H, W);

    DALI_ENFORCE(C == C_,
      "Input channel dimension does not match "
      "the output image type. Expected input with " +
      to_string(C_) + " channels, got " + to_string(C) + ".");

    DALI_ENFORCE(H >= crop_height_[dataIdx] && W >= crop_width_[dataIdx],
      "Image dimensions for sample " + std::to_string(dataIdx)
      + " (" + std::to_string(H)
      + ", " + std::to_string(W) + ")"
      + " are smaller than the cropping window"
      + " (" + std::to_string(crop_height_[dataIdx])
      + ", " + std::to_string(crop_width_[dataIdx]) + ")");

    per_sample_crop_[threadIdx] = CalculateCropYX(
      crop_y_norm_[dataIdx],
      crop_x_norm_[dataIdx],
      crop_height_[dataIdx],
      crop_width_[dataIdx],
      H, W);
  }

  void Init(int size) {
    per_sample_crop_.resize(size);
    per_sample_dimensions_.resize(size);
    output_type_ = DALI_NO_TYPE;
    output_layout_ = DALI_SAME;
  }

  /**
   * @brief Enforce that all shapes match
   *
   * @param ws
   * @return const vector<Index> One matching shape for all inputs
   */
  virtual const std::vector<Index> CheckShapes(const SampleWorkspace *ws) {
    const auto &input = ws->Input<CPUBackend>(0);
    // enforce that all shapes match
    for (int i = 1; i < ws->NumInput(); ++i) {
      DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }
    return input.shape();
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
