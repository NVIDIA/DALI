// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_EXPERIMENTAL_RESIZE_OP_IMPL_CVCUDA_H_
#define DALI_OPERATORS_IMAGE_RESIZE_EXPERIMENTAL_RESIZE_OP_IMPL_CVCUDA_H_

#include <cvcuda/OpHQResize.hpp>

#include <vector>

#include "dali/kernels/imgproc/resample/params.h"
#include "dali/operators/image/resize/resize_op_impl.h"
#include "dali/operators/nvcvop/nvcvop.h"

namespace dali {

template <int spatial_ndim>
class ResizeOpImplCvCuda : public ResizeBase<GPUBackend>::Impl {
 public:
  explicit ResizeOpImplCvCuda(int minibatch_size) : minibatch_size_(minibatch_size) {}

  static_assert(spatial_ndim == 2 || spatial_ndim == 3, "Only 2D and 3D resizing is supported");


  /// Dimensionality of each separate frame. If input contains no channel dimension, one is added
  static constexpr int frame_ndim = spatial_ndim + 1;

  void Setup(TensorListShape<> &out_shape, const TensorListShape<> &in_shape, int first_spatial_dim,
             span<const kernels::ResamplingParams> params) override {
    // Calculate output shape of the input, as supplied (sequences, planar images, etc)
    GetResizedShape(out_shape, in_shape, params, spatial_ndim, first_spatial_dim);

    // Create "frames" from outer dimensions and "channels" from inner dimensions.
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params, first_spatial_dim);

    // Now that we have per-frame parameters, we can calculate the output shape of the
    // effective frames (from videos, channel planes, etc).
    GetResizedShape(out_shape_, in_shape_, make_cspan(params_), 0);

    curr_minibatch_size_ = minibatch_size_;
    // CV-CUDA tensors can't represent empty samples,
    // so curr_minibatch_size_ is set to 1 to skip each empty sample one by one
    if (HasEmptySamples(in_shape_)) {
      curr_minibatch_size_ = 1;
    }

    // Now that we know how many logical frames there are, calculate batch subdivision.
    CalculateMinibatchPartition(in_shape_.num_samples(), curr_minibatch_size_);

    SetupKernel();
  }

  void SetupKernel() {
    kernels::KernelContext ctx;
    rois_.clear();
    rois_.reserve(minibatch_size_ * minibatches_.size());
    workspace_reqs_ = {};
    std::vector<HQResizeTensorShapeI> mb_input_shapes(curr_minibatch_size_);
    std::vector<HQResizeTensorShapeI> mb_output_shapes(curr_minibatch_size_);
    for (int mb_idx = 0, num_mb = minibatches_.size(); mb_idx < num_mb; mb_idx++) {
      auto &mb = minibatches_[mb_idx];

      int end = mb.start + mb.count;
      auto param_slice = make_span(&params_[mb.start], mb.count);
      for (int i = mb.start, j = 0; i < end; i++, j++) {
        rois_.push_back(GetRoi(param_slice[j]));
        for (int d = 0; d < spatial_ndim; ++d) {
          mb_input_shapes[j].extent[d] = static_cast<int32_t>(in_shape_.tensor_shape_span(i)[d]);
          mb_output_shapes[j].extent[d] = static_cast<int32_t>(out_shape_.tensor_shape_span(i)[d]);
        }
      }
      int num_channels = in_shape_[0][frame_ndim - 1];
      HQResizeTensorShapesI mb_input_shape{mb_input_shapes.data(), mb.count, spatial_ndim,
                                           num_channels};
      HQResizeTensorShapesI mb_output_shape{mb_output_shapes.data(), mb.count, spatial_ndim,
                                            num_channels};
      mb.rois = HQResizeRoisF{mb.count, spatial_ndim, &rois_.data()[mb.start]};

      auto param = params_[mb.start][0];
      mb.min_interpolation = GetInterpolationType(param.min_filter);
      mb.mag_interpolation = GetInterpolationType(param.mag_filter);
      mb.antialias = param.min_filter.antialias || param.mag_filter.antialias;
      auto ws_req = resize_op_.getWorkspaceRequirements(mb.count, mb_input_shape, mb_output_shape,
                                                        mb.min_interpolation, mb.mag_interpolation,
                                                        mb.antialias, mb.rois);
      workspace_reqs_ = nvcvop::MaxWorkspaceRequirements(workspace_reqs_, ws_req);
    }
  }

  static bool HasEmptySamples(const TensorListShape<> &in_shape) {
    for (int s = 0; s < in_shape.num_samples(); s++) {
      if (volume(in_shape.tensor_shape_span(s)) == 0) {
        return true;
      }
    }
    return false;
  }

  HQResizeRoiF GetRoi(const ResamplingParamsND<spatial_ndim> &params) {
    HQResizeRoiF roi;
    for (int d = 0; d < spatial_ndim; d++) {
      roi.lo[d] = params[d].roi.start;
      roi.hi[d] = params[d].roi.end;
    }
    return roi;
  }

  NVCVInterpolationType GetInterpolationType(kernels::FilterDesc filter_desc) {
    using kernels::ResamplingFilterType;
    switch (filter_desc.type) {
      case ResamplingFilterType::Nearest:
        return NVCVInterpolationType::NVCV_INTERP_NEAREST;
      case ResamplingFilterType::Linear:
        return NVCVInterpolationType::NVCV_INTERP_LINEAR;
      case ResamplingFilterType::Triangular:
        return NVCVInterpolationType::NVCV_INTERP_LINEAR;
      case ResamplingFilterType::Cubic:
        return NVCVInterpolationType::NVCV_INTERP_CUBIC;
      case ResamplingFilterType::Lanczos3:
        return NVCVInterpolationType::NVCV_INTERP_LANCZOS;
      case ResamplingFilterType::Gaussian:
        return NVCVInterpolationType::NVCV_INTERP_GAUSSIAN;
      default:
        DALI_FAIL("Unsupported filter type");
    }
  }

  void RunResize(Workspace &ws, TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input) override {
    TensorList<GPUBackend> in_frames;
    in_frames.ShareData(input);
    in_frames.Resize(in_shape_);
    PrepareInput(in_frames);

    TensorList<GPUBackend> out_frames;
    out_frames.ShareData(output);
    out_frames.Resize(out_shape_);
    PrepareOutput(out_frames);


    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));

    auto workspace_mem = op_workspace_.Allocate(workspace_reqs_, scratchpad);

    for (size_t b = 0; b < minibatches_.size(); b++) {
      MiniBatch &mb = minibatches_[b];
      if (mb.input.numTensors() == 0 || mb.output.numTensors() == 0) {
        assert(mb.count == 1);
        // we skip empty samples
        continue;
      }
      resize_op_(ws.stream(), workspace_mem, mb.input, mb.output, mb.min_interpolation,
                 mb.mag_interpolation, mb.antialias, mb.rois);
    }
  }

  int CalculateMinibatchPartition(int total_frames, int minibatch_size) {
    int num_minibatches = div_ceil(total_frames, minibatch_size);

    minibatches_.resize(num_minibatches);
    int start = 0;
    for (int i = 0; i < num_minibatches; i++) {
      int end = (i + 1) * total_frames / num_minibatches;
      auto &mb = minibatches_[i];
      mb.start = start;
      mb.count = end - start;
      start = end;
    }
    return num_minibatches;
  }

  TensorListShape<frame_ndim> in_shape_, out_shape_;
  std::vector<ResamplingParamsND<spatial_ndim>> params_;

  cvcuda::HQResize resize_op_{};
  nvcvop::NVCVOpWorkspace op_workspace_;
  cvcuda::WorkspaceRequirements workspace_reqs_{};
  std::vector<HQResizeRoiF> rois_;
  const TensorLayout sample_layout_ = (spatial_ndim == 2) ? "HWC" : "DHWC";

  struct MiniBatch {
    int start, count;
    nvcv::TensorBatch input;
    nvcv::TensorBatch output;
    NVCVInterpolationType min_interpolation;
    NVCVInterpolationType mag_interpolation;
    bool antialias;
    HQResizeRoisF rois;
  };

  std::vector<MiniBatch> minibatches_;

  void PrepareInput(const TensorList<GPUBackend> &input) {
    for (auto &mb : minibatches_) {
      int curr_capacity = mb.output ? mb.output.capacity() : 0;
      if (mb.count > curr_capacity) {
        int new_capacity = std::max(mb.count, curr_capacity * 2);
        auto reqs = nvcv::TensorBatch::CalcRequirements(new_capacity);
        mb.input = nvcv::TensorBatch(reqs);
      } else {
        mb.input.clear();
      }
      for (int i = mb.start; i < mb.start + mb.count; ++i) {
        if (volume(in_shape_.tensor_shape_span(i)) != 0) {
          mb.input.pushBack(nvcvop::AsTensor(input[i], sample_layout_));
        }
      }
    }
  }

  void PrepareOutput(const TensorList<GPUBackend> &out) {
    for (auto &mb : minibatches_) {
      int curr_capacity = mb.output ? mb.output.capacity() : 0;
      if (mb.count > curr_capacity) {
        int new_capacity = std::max(mb.count, curr_capacity * 2);
        auto reqs = nvcv::TensorBatch::CalcRequirements(new_capacity);
        mb.output = nvcv::TensorBatch(reqs);
      } else {
        mb.output.clear();
      }
      for (int i = mb.start; i < mb.start + mb.count; ++i) {
        if (volume(out_shape_.tensor_shape_span(i)) != 0) {
          mb.output.pushBack(nvcvop::AsTensor(out[i], sample_layout_));
        }
      }
    }
  }

  int minibatch_size_;
  int curr_minibatch_size_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_EXPERIMENTAL_RESIZE_OP_IMPL_CVCUDA_H_
