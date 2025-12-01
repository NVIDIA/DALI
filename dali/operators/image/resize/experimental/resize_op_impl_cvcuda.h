// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include <unordered_map>

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
    first_spatial_dim_ = first_spatial_dim;

    // Calculate output shape of the input, as supplied (sequences, planar images, etc)
    GetResizedShape(out_shape, in_shape, params, spatial_ndim, first_spatial_dim);

    // Create "frames" from outer dimensions and "channels" from inner dimensions.
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params, first_spatial_dim);

    ValidateShapes(in_shape_);

    // Now that we have per-frame parameters, we can calculate the output shape of the
    // effective frames (from videos, channel planes, etc).
    GetResizedShape(out_shape_, in_shape_, make_cspan(params_), 0);

    // Now that we know how many logical frames there are, calculate batch subdivision.
    CalculateMinibatchPartition(minibatch_size_);

    CalculateSourceSamples(in_shape, first_spatial_dim);

    SetupKernel();
  }

  void ValidateShapes(const TensorListShape<> &in_shape) {
    // Ensure all frames have the same number of channels
    if (in_shape_.num_samples() == 0)
      return;
    int num_channels = in_shape_.tensor_shape_span(0)[frame_ndim - 1];
    for (int i = 1; i < in_shape_.num_samples(); ++i) {
      int ch = in_shape_.tensor_shape_span(i)[frame_ndim - 1];
      DALI_ENFORCE(ch == num_channels, "All frames must have the same number of channels");
    }
  }

  // Assign each minibatch a range of frames in the original input/output TensorLists
  void CalculateSourceSamples(const TensorListShape<> &original_shape, int first_spatial_dim) {
    int64_t sample_id = 0;
    int64_t starting_frame_idx = 0;
    int64_t sample_volume = 0;
    int64_t sample_num_frames = 0;
    for (auto &mb : minibatches_) {
      assert(sample_id < original_shape.num_samples());
      sample_volume = original_shape[sample_id].num_elements();
      while (sample_volume == 0) {
        sample_id++;
        assert(sample_id < original_shape.num_samples());
        sample_volume = original_shape[sample_id].num_elements();
      }
      mb.sample_offset = sample_id;
      mb.starting_frame_idx = starting_frame_idx;
      starting_frame_idx += mb.count;
      sample_num_frames = num_frames(original_shape[sample_id], first_spatial_dim);
      while (starting_frame_idx >= sample_num_frames) {
        starting_frame_idx -= sample_num_frames;
        if (++sample_id >= original_shape.num_samples()) {
          break;
        }
        sample_num_frames = num_frames(original_shape[sample_id], first_spatial_dim);
      }
    }
  }

  int64_t num_frames(const TensorShape<> &shape, int first_spatial_dim) const {
    return volume(&shape[0], &shape[first_spatial_dim]);
  }

  void SetupKernel() {
    rois_.resize(total_frames_);
    workspace_reqs_[0] = {};
    workspace_reqs_[1] = {};
    std::vector<HQResizeTensorShapeI> mb_input_shapes(minibatch_size_);
    std::vector<HQResizeTensorShapeI> mb_output_shapes(minibatch_size_);
    auto *rois_ptr = rois_.data();
    for (int mb_idx = 0, num_mb = minibatches_.size(); mb_idx < num_mb; mb_idx++) {
      auto &mb = minibatches_[mb_idx];

      int end = mb.start + mb.count;
      for (int i = mb.start, j = 0; i < end; i++, j++) {
        rois_ptr[j] = GetRoi(params_[i]);
        for (int d = 0; d < spatial_ndim; ++d) {
          mb_input_shapes[j].extent[d] = static_cast<int32_t>(in_shape_.tensor_shape_span(i)[d]);
          mb_output_shapes[j].extent[d] =
              static_cast<int32_t>(out_shape_.tensor_shape_span(i)[d]);
        }
      }
      int num_channels = in_shape_[0][frame_ndim - 1];
      HQResizeTensorShapesI mb_input_shape{mb_input_shapes.data(), mb.count, spatial_ndim,
                                           num_channels};
      HQResizeTensorShapesI mb_output_shape{mb_output_shapes.data(), mb.count, spatial_ndim,
                                            num_channels};
      mb.rois = HQResizeRoisF{mb.count, spatial_ndim, rois_ptr};
      rois_ptr += mb.count;

      const auto &param = params_[mb.start][0];
      mb.min_interpolation = GetInterpolationType(param.min_filter);
      mb.mag_interpolation = GetInterpolationType(param.mag_filter);
      mb.antialias = param.min_filter.antialias || param.mag_filter.antialias;
      auto ws_req = resize_op_.getWorkspaceRequirements(mb.count, mb_input_shape, mb_output_shape,
                                                        mb.min_interpolation, mb.mag_interpolation,
                                                        mb.antialias, mb.rois);
      workspace_reqs_[mb_idx % 2] = cvcuda::MaxWorkspaceReq(workspace_reqs_[mb_idx % 2], ws_req);
    }
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
    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));

    auto workspace_mem = AllocateWorkspaces(scratchpad);

    std::vector<CachedBatches> batches;

    for (size_t b = 0; b < minibatches_.size(); b++) {
      MiniBatch &mb = minibatches_[b];
      batches.push_back(GetBatches(mb.count));
      auto &mb_input = batches.back().input;
      auto &mb_output = batches.back().output;
      mb_input.clear();
      mb_output.clear();

      nvcvop::PushFramesToBatch(mb_input, input, first_spatial_dim_, mb.sample_offset,
                                mb.starting_frame_idx, mb.count, sample_layout_);
      nvcvop::PushFramesToBatch(mb_output, output, first_spatial_dim_, mb.sample_offset,
                                mb.starting_frame_idx, mb.count, sample_layout_);
      resize_op_(ws.stream(), workspace_mem[b % 2], mb_input, mb_output, mb.min_interpolation,
                 mb.mag_interpolation, mb.antialias, mb.rois);
    }

    for (auto &batch : batches) {
      ReturnBatches(std::move(batch));
    }
  }

  std::array<cvcuda::Workspace, 2> AllocateWorkspaces(kernels::Scratchpad &scratchpad) {
    std::array<cvcuda::Workspace, 2> result;
    result[0] = op_workspace_.Allocate(workspace_reqs_[0], scratchpad);
    if (minibatches_.size() > 1) {
      result[1] = op_workspace_.Allocate(workspace_reqs_[1], scratchpad);
    }
    return result;
  }

  void CalculateMinibatchPartition(int minibatch_size) {
    total_frames_ = in_shape_.num_samples();
    std::vector<std::pair<int, int>> continuous_ranges;
    kernels::FilterDesc min_filter_desc = params_[0][0].min_filter;
    kernels::FilterDesc mag_filter_desc = params_[0][0].mag_filter;
    int start_id = 0;
    for (int i = 0; i < total_frames_; i++) {
      if (params_[i][0].min_filter != min_filter_desc ||
          params_[i][0].mag_filter != mag_filter_desc) {
        // we break the range if different filter types are used
        continuous_ranges.emplace_back(start_id, i);
        start_id = i;
      }
    }
    if (start_id < total_frames_) {
      continuous_ranges.emplace_back(start_id, total_frames_);
    }

    minibatches_.clear();
    int mb_idx = 0;
    for (auto &range : continuous_ranges) {
      int range_count = range.second - range.first;
      int num_minibatches = div_ceil(range_count, minibatch_size);

      minibatches_.resize(minibatches_.size() + num_minibatches);
      int start = 0;
      for (int i = 0; i < num_minibatches; ++i, ++mb_idx) {
        int end = (i + 1) * range_count / num_minibatches;
        auto &mb = minibatches_[mb_idx];
        mb.start = start + range.first;
        mb.count = end - start;
        start = end;
      }
    }
  }

  struct CachedBatches {
    nvcv::TensorBatch input;
    nvcv::TensorBatch output;
  };

  using BatchCache = std::vector<CachedBatches>;

  CachedBatches GetBatches(int batch_size) {
    auto &cache = batch_cache_[batch_size];
    if (!cache.empty()) {
      auto result = std::move(cache.back());
      cache.pop_back();
      return result;
    }
    auto reqs = nvcv::TensorBatch::CalcRequirements(batch_size);
    return CachedBatches{nvcv::TensorBatch(reqs), nvcv::TensorBatch(reqs)};
  }

  void ReturnBatches(CachedBatches &&batches) {
    batch_cache_[batches.input.numTensors()].push_back(std::move(batches));
  }

  TensorListShape<frame_ndim> in_shape_, out_shape_;
  int total_frames_ = 0;  // number of non-empty frames
  std::vector<ResamplingParamsND<spatial_ndim>> params_;
  int first_spatial_dim_ = 0;

  cvcuda::HQResize resize_op_{};
  nvcvop::NVCVOpWorkspace op_workspace_;
  std::array<cvcuda::WorkspaceRequirements, 2> workspace_reqs_{};
  std::vector<HQResizeRoiF> rois_;
  const TensorLayout sample_layout_ = (spatial_ndim == 2) ? "HWC" : "DHWC";

  std::vector<const void*> in_frames_;
  std::vector<const void*> out_frames_;

  struct MiniBatch {
    int start, count;
    NVCVInterpolationType min_interpolation;
    NVCVInterpolationType mag_interpolation;
    bool antialias;
    HQResizeRoisF rois;
    int64_t sample_offset;  // id of a starting sample in the original IOs
    int64_t starting_frame_idx;  // id of a starting frame in the starting sample
  };

  std::vector<MiniBatch> minibatches_;
  int minibatch_size_;
  // cache of batches for different batch sizes
  std::unordered_map<int, BatchCache> batch_cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_EXPERIMENTAL_RESIZE_OP_IMPL_CVCUDA_H_
