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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/pipeline/operator/operator.h"

#define CMN_IN_TYPES (uint8_t, int16_t, uint16_t, int32_t, float, float16)
#define CMN_OUT_TYPES (float, float16)
#define CMN_NDIMS (3, 4, 5)

namespace dali {

namespace detail {

template <typename T>
T NextPowerOfTwo(T value) {
  T power = 1;
  while (power < value)
    power <<= 1;
  return power;
}

// Rewrite Operator data as arguments for kernel
// TODO(klecki): It probably could be written directly in that format
template <int Dims>
kernels::SliceFlipNormalizePermutePadArgs<Dims> GetKernelArgs(
    TensorLayout input_layout, TensorLayout output_layout,
    const std::vector<int64_t> &slice_anchor, const std::vector<int64_t> &slice_shape,
    bool horizontal_flip, bool pad_output, const std::vector<float> &mean,
    const std::vector<float> &inv_std_dev) {
  kernels::SliceFlipNormalizePermutePadArgs<Dims> args(slice_shape);

  for (int d = 0; d < Dims; d++) {
    args.anchor[d] = slice_anchor[d];
  }

  int channel_dim_idx = ImageLayoutInfo::ChannelDimIndex(input_layout);
  assert(channel_dim_idx >= 0);
  if (pad_output) {
    args.padded_shape[channel_dim_idx] = NextPowerOfTwo(args.shape[channel_dim_idx]);
  }

  if (horizontal_flip) {
    int horizontal_dim_idx = input_layout.find('W');
    assert(horizontal_dim_idx >= 0);
    args.flip[horizontal_dim_idx] = true;
  }

  // Check if permutation is needed
  args.permuted_dims = GetLayoutMapping<Dims>(input_layout, output_layout);

  const bool should_normalize =
      !std::all_of(mean.begin(), mean.end(), [](float x) { return x == 0.0f; }) ||
      !std::all_of(inv_std_dev.begin(), inv_std_dev.end(), [](float x) { return x == 1.0f; });
  if (should_normalize) {
    args.mean = mean;
    args.inv_stddev = inv_std_dev;
    args.normalization_dim = channel_dim_idx;
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

    DALI_ENFORCE(!mean_vec_.empty() && !inv_std_vec_.empty(),
      "mean and standard deviation can't be empty");

    DALI_ENFORCE(
      mean_vec_.size() == inv_std_vec_.size() || mean_vec_.size() == 1 || inv_std_vec_.size() == 1,
      "`mean` and `stddev` must either be of the same size, be scalars, or one of them can be a "
      "vector and the other a scalar.");

    // Inverse the std-deviation
    for (auto &element : inv_std_vec_) {
      element = 1.f / element;
    }

    // Handle irregular mean/std argument lengths
    auto args_size = std::max(mean_vec_.size(), inv_std_vec_.size());
    if (mean_vec_.size() != inv_std_vec_.size()) {
      if (mean_vec_.size() == 1)
        mean_vec_.resize(args_size, mean_vec_[0]);

      if (inv_std_vec_.size() == 1)
        inv_std_vec_.resize(args_size, inv_std_vec_[0]);
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

    int number_of_dims = in_shape.sample_dim();
    VALUE_SWITCH(number_of_dims, Dims, CMN_NDIMS,
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
  void SetupSample(int data_idx, TensorLayout layout, const TensorShape<> &shape) {
    Index F = 1, D = 1, H, W, C;
    DALI_ENFORCE(shape.size() >= 3 || shape.size() <= 5,
      "Unexpected number of dimensions: " + std::to_string(shape.size()));
    DALI_ENFORCE(layout.ndim() == shape.size());
    int spatial_ndim = ImageLayoutInfo::NumSpatialDims(layout);
    DALI_ENFORCE(spatial_ndim == 2 || spatial_ndim == 3,
      "Only 2D or 3D images and sequences of images are supported");
    DALI_ENFORCE(ImageLayoutInfo::HasChannel(layout),
      "This operator expects an explicit channel dimension, even for monochrome images");

    int h_dim = layout.find('H');
    int w_dim = layout.find('W');
    int c_dim = layout.find('C');
    int f_dim = layout.find('F');
    int d_dim = layout.find('D');

    DALI_ENFORCE(h_dim >= 0 && w_dim >= 0 && c_dim >= 0,
      "Height, Width and Channel must be present in the layout. Got: " + layout.str());

    H = shape[h_dim];
    W = shape[w_dim];
    C = shape[c_dim];
    if (f_dim >= 0)
      F = shape[f_dim];
    if (d_dim >= 0)
      D = shape[d_dim];

    // Special case.
    // This allows using crop_d to crop on the sequence dimension,
    // by treating video inputs as a volume instead of a sequence
    if (has_crop_d_ && F > 1 && D == 1) {
      std::swap(d_dim, f_dim);
      std::swap(D, F);
      spatial_ndim++;
    }

    auto crop_window_gen = GetCropWindowGenerator(data_idx);
    auto win = spatial_ndim == 3 ?
      crop_window_gen({D, H, W}, "DHW") : crop_window_gen({H, W}, "HW");

    int ndim = shape.sample_dim();
    slice_anchors_[data_idx].resize(ndim);
    slice_shapes_[data_idx].resize(ndim);

    if (d_dim >= 0) {
      slice_anchors_[data_idx][d_dim] = win.anchor[spatial_ndim - 3];
      slice_shapes_[data_idx][d_dim] = win.shape[spatial_ndim - 3];
    }

    slice_anchors_[data_idx][h_dim] = win.anchor[spatial_ndim - 2];
    slice_shapes_[data_idx][h_dim] = win.shape[spatial_ndim - 2];

    slice_anchors_[data_idx][w_dim] = win.anchor[spatial_ndim - 1];
    slice_shapes_[data_idx][w_dim] = win.shape[spatial_ndim - 1];

    slice_anchors_[data_idx][c_dim] = 0;
    slice_shapes_[data_idx][c_dim] = C;

    if (f_dim >= 0) {
      slice_anchors_[data_idx][f_dim] = 0;
      slice_shapes_[data_idx][f_dim] = F;
    }
  }

  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  TensorLayout input_layout_;
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

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_
