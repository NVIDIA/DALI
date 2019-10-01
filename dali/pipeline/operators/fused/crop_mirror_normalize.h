// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/crop_attr.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

namespace detail {

inline size_t horizontal_dim_idx(const TensorLayout &layout) {
  return layout.find('W');
}



inline int channels_dim(TensorLayout in_layout) {
  return ImageLayoutInfo::ChannelDimIndex(in_layout);
}

// Rewrite Operator data as arguments for kernel
// TODO(klecki): It probably could be written directly in that format
template <size_t Dims>
kernels::SliceFlipNormalizePermutePadArgs<Dims> GetKernelArgs(
    TensorLayout input_layout, TensorLayout output_layout,
    const std::vector<int64_t> &slice_anchor, const std::vector<int64_t> &slice_shape,
    bool horizontal_flip, bool pad_output, const std::vector<float> &mean,
    const std::vector<float> &inv_std_dev) {
  kernels::SliceFlipNormalizePermutePadArgs<Dims> args(slice_shape);

  for (std::size_t d = 0; d < Dims; d++) {
    args.anchor[d] = slice_anchor[d];
  }

  if (pad_output) {
    args.padded_shape[channels_dim(input_layout)] = 4;
  }

  if (horizontal_flip) {
    args.flip[horizontal_dim_idx(input_layout)] = true;
  }

  // Check if permutation is needed
  args.permuted_dims = GetLayoutMapping<Dims>(input_layout, output_layout);

  const bool should_normalize =
      !std::all_of(mean.begin(), mean.end(), [](float x) { return x == 0.0f; }) ||
      !std::all_of(inv_std_dev.begin(), inv_std_dev.end(), [](float x) { return x == 1.0f; });
  if (should_normalize) {
    args.mean = mean;
    args.inv_stddev = inv_std_dev;
    args.normalization_dim = channels_dim(input_layout);
  }

  return args;
}

}  // namespace detail


template <typename Backend>
class CropMirrorNormalize : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec)
      : Operator<Backend>(spec),
        CropAttr(spec),
        output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
        output_layout_(spec.GetArgument<TensorLayout>("output_layout")),
        pad_output_(spec.GetArgument<bool>("pad_output")),
        slice_anchors_(batch_size_),
        slice_shapes_(batch_size_),
        mirror_(batch_size_) {
    if (!spec.TryGetRepeatedArgument(mean_vec_, "mean")) {
      mean_vec_ = { spec.GetArgument<float>("mean") };
    }

    if (!spec.TryGetRepeatedArgument(inv_std_vec_, "std")) {
      inv_std_vec_ = { spec.GetArgument<float>("std") };
    }

    // Inverse the std-deviation
    for (auto &element : inv_std_vec_) {
      element = 1.f / element;
    }
    if (std::is_same<Backend, GPUBackend>::value) {
      kmgr_.Resize(1, 1);
    } else {
      kmgr_.Resize(num_threads_, batch_size_);
    }
  }

  inline ~CropMirrorNormalize() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(Workspace<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  // Propagate input -> output type and layout
  // Gather the CropAttr (obtain arguments from Spec/ArgumentWorkspace)
  void SetupAndInitialize(const workspace_t<Backend> &ws) {
    const auto &input = ws.template InputRef<Backend>(0);
    input_type_ = input.type().id();
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = input_type_;

    const auto &in_shape = input.shape();  // This can be a copy
    input_layout_ = this->InputLayout(ws, 0);
    DALI_ENFORCE(ImageLayoutInfo::IsImage(input_layout_),
      ("Unsupported layout: '" + input_layout_.str() + "' for input 0 '" +
      this->spec_.InputName(0) + "'"));
    DALI_ENFORCE(input_layout_.ndim() == in_shape.sample_dim(),
      "Number of dimension in layout description does not match the number"
      " of dimensions in the input.");
    if (output_layout_.empty())
      output_layout_ = input_layout_;
    else
      DALI_ENFORCE(output_layout_.is_permutation_of(input_layout_),
        "The requested output layout is not a permutation of input layout.");

    CropAttr::ProcessArguments(ws);

    std::size_t number_of_dims = in_shape.sample_dim();

    VALUE_SWITCH(number_of_dims, Dims, (3, 4),
    (
      using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
      // We won't change the underlying type after the first allocation
      if (!kernel_sample_args_.has_value()) {
        kernel_sample_args_ = std::vector<Args>(batch_size_);
      }
      auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);

      // Set internal info for each sample based on CropAttr
      for (int data_idx = 0; data_idx < batch_size_; data_idx++) {
        mirror_[data_idx] = this->spec_.template GetArgument<int>("mirror", &ws, data_idx);
        SetupSample(data_idx, input_layout_, in_shape.tensor_shape(data_idx));
        // convert the Operator representation to Kernel parameter representation
        kernel_sample_args[data_idx] = detail::GetKernelArgs<Dims>(
          input_layout_, output_layout_, slice_anchors_[data_idx], slice_shapes_[data_idx],
          mirror_[data_idx], pad_output_, mean_vec_, inv_std_vec_);
      }

      // NOLINTNEXTLINE(whitespace/parens)
    ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims)););



    auto &output = ws.template OutputRef<Backend>(0);
    output.SetLayout(output_layout_);
  }

  // Calculate slice window and anchor for given data_idx
  void SetupSample(int data_idx, TensorLayout layout, const kernels::TensorShape<> &shape) {
    Index F = 1, H, W, C;
    DALI_ENFORCE(shape.size() == 3 || shape.size() == 4,
      "Unexpected number of dimensions: " + std::to_string(shape.size()));
    DALI_ENFORCE(ImageLayoutInfo::NumSpatialDims(layout) == 2,
      "Only 2D images and sequences of images are supported");
    DALI_ENFORCE(ImageLayoutInfo::HasChannel(layout),
      "This operator expects an explicit channel dimesnion, even for monochrome images");

    int h_dim = layout.find('H');
    int w_dim = layout.find('W');
    int c_dim = layout.find('C');
    int f_dim = layout.find('F');

    DALI_ENFORCE(h_dim >= 0 && w_dim >= 0 && c_dim >= 0,
      "Height, Width and Channel must be present in the layout. Got: " + layout.str());

    H = shape[h_dim];
    W = shape[w_dim];
    C = shape[c_dim];
    if (f_dim >= 0) {
      F = shape[layout.find('F')];
    }

    const bool is_whole_image = IsWholeImage();
    int crop_h = is_whole_image ? H : crop_height_[data_idx];
    int crop_w = is_whole_image ? W : crop_width_[data_idx];

    float anchor_norm[2] = {crop_y_norm_[data_idx], crop_x_norm_[data_idx]};
    auto anchor = CalculateAnchor(make_span(anchor_norm), {crop_h, crop_w}, {H, W});
    int64_t crop_y = anchor[0], crop_x = anchor[1];

    int ndim = shape.sample_dim();
    slice_anchors_[data_idx].resize(ndim);
    slice_shapes_[data_idx].resize(ndim);

    slice_anchors_[data_idx][h_dim] = crop_y;
    slice_shapes_[data_idx][h_dim] = crop_h;
    slice_anchors_[data_idx][w_dim] = crop_x;
    slice_shapes_[data_idx][w_dim] = crop_w;

    slice_anchors_[data_idx][c_dim] = 0;
    slice_shapes_[data_idx][c_dim] = C;
    if (f_dim >= 0) {
      slice_anchors_[data_idx][f_dim] = 0;
      slice_shapes_[data_idx][f_dim] = F;
    }
  }

  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  TensorLayout input_layout_ = "HWC";
  TensorLayout output_layout_;

  // Whether to pad output to 4 channels
  bool pad_output_;

  std::vector<std::vector<int64_t>> slice_anchors_, slice_shapes_;

  std::vector<float> mean_vec_, inv_std_vec_;
  std::vector<int> mirror_;

  kernels::KernelManager kmgr_;
  any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};



}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
