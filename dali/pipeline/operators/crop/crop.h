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

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class CropAttr {
 protected:
  explicit inline CropAttr(const OpSpec &spec)
      : image_type_(spec.GetArgument<DALIImageType>("image_type")),
        C_(IsColor(image_type_) ? 3 : 1),
        batch_size_{spec.GetArgument<int>("batch_size")} {
    if (spec.name() != "Resize") {
      vector<float> cropArgs = spec.GetRepeatedArgument<float>("crop");

      DALI_ENFORCE(cropArgs[0] >= 0, "Crop height must be greater than zero. Received: " +
                                         std::to_string(cropArgs[0]));
      DALI_ENFORCE(cropArgs[1] >= 0, "Crop width must be greater than zero. Received: " +
                                         std::to_string(cropArgs[1]));

      crop_height_ = std::vector<int>(batch_size_, static_cast<int>(cropArgs[0]));
      crop_width_ = std::vector<int>(batch_size_, static_cast<int>(cropArgs[1]));
    }
  }

  /**
   * @brief Calculate coordinate where the crop starts in pixels.
   *
   * TODO(klecki) this should produce (crop_[0] * W, crop_[1] * H) and is broken, (FIXME)
   *
   * @param spec
   * @param ws
   * @param imgIdx
   * @param H
   * @param W
   * @return std::pair<int, int>
   */
  std::pair<int, int> SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
                                const Index dataIdx, int H, int W);

  /**
   * @brief Enforce that all shapes match
   *
   * @param ws
   * @return const vector<Index> One matching shape for all inputs
   */
  virtual const vector<Index> CheckShapes(const SampleWorkspace *ws);

  vector<int> crop_height_;
  vector<int> crop_width_;

  const DALIImageType image_type_;
  const int C_;
  const int batch_size_;
};

template <typename Backend>
class Crop : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline Crop(const OpSpec &spec) : Operator<Backend>(spec), CropAttr(spec) {
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
                      Tensor<CPUBackend> *output) {
    auto in_shape = detail::ToStaticShape<Kernel::input_dim>(input.shape());
    auto out_shape = Kernel::CalcOutputSize(in_shape, args);
    output->Resize(detail::ToDynamicShape(out_shape));
  }

  // Invoke Run from CropKernel
  template <typename Kernel>
  void RunKernel(const Tensor<CPUBackend> &input, typename Kernel::KernelAttributes args,
                 Tensor<CPUBackend> *output) {
    // ValidateHelper not needed - TensorWrapper ensures that ptr != nullptr.
    // TODO(klecki) - Input and output allocations should already be hanlded at this stage.

    const auto *in = input.template data<typename Kernel::InputType>();
    auto in_shape = detail::ToStaticShape<Kernel::input_dim>(input.shape());
    auto *out = output->template mutable_data<typename Kernel::OutputType>();
    auto out_shape = detail::ToStaticShape<Kernel::output_dim>(output->shape());
    Kernel::Run(in, in_shape, args, out, out_shape);
  }

  template <typename Kernel>
  void AllocateAndRunKernel(Workspace<CPUBackend> *ws, const int idx) {
    const auto &input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);
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

  USE_OPERATOR_MEMBERS();

 private:
  void DataDependentSetup(Workspace<Backend> *ws, int idx);
  template <typename Out>
  void ValidateHelper(TensorList<Backend> *output);

  void SetupSharedSampleParams(const ArgumentWorkspace *ws,
                               const vector<Index> &inputShape, int threadIdx,
                               int dataIdx) {
    DALI_ENFORCE(inputShape.size() == 3, "Expects 3-dimensional image input.");

    const int H = inputShape[0];
    const int W = inputShape[1];

    per_sample_dimensions_[threadIdx] = std::make_pair(H, W);

    int C = inputShape[2];
    DALI_ENFORCE(C == C_,
                 "Input channel dimension does not match "
                 "the output image type. Expected input with " +
                     to_string(C_) + " channels, got " + to_string(C) + ".");

    per_sample_crop_[threadIdx] = SetCropXY(spec_, ws, dataIdx, H, W);
  }

  void Init(int size) {
    per_sample_crop_.resize(size);
    per_sample_dimensions_.resize(size);
    output_type_ = DALI_NO_TYPE;
    output_layout_ = DALI_SAME;
  }

  void CallRunHelper(Workspace<Backend> *ws, int idx) {
    if (output_type_ == DALI_FLOAT) {
      RunHelper<float>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      RunHelper<unsigned char>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      RunHelper<int16>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      RunHelper<int>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      RunHelper<int64>(ws, idx);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
