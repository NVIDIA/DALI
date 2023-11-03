// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_H_
#define DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_H_

#include <memory>
#include <utility>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/operators/sequence/optical_flow/optical_flow_adapter/optical_flow_stub.h"
#include "dali/operators/sequence/optical_flow/optical_flow_impl/optical_flow_impl.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/nvml.h"

namespace dali {

namespace detail {

template<typename Backend>
struct backend_to_compute {
  using type = ComputeCPU;
};

template<>
struct backend_to_compute<GPUBackend> {
  using type = ComputeGPU;
};

static const std::string kPresetArgName = "preset";                               // NOLINT
static const std::string kOutputGridArgName = "output_grid";                      // NOLINT
static const std::string kHintGridArgName = "hint_grid";                          // NOLINT
static const std::string kEnableTemporalHintsArgName = "enable_temporal_hints";   // NOLINT
static const std::string kEnableExternalHintsArgName = "enable_external_hints";   // NOLINT
static const std::string kImageTypeArgName = "image_type";                        // NOLINT

}  // namespace detail

template<typename Backend>
class OpticalFlow : public StatelessOperator<Backend> {
  using ComputeBackend = typename detail::backend_to_compute<Backend>::type;

 public:
  explicit OpticalFlow(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        quality_factor_(spec.GetArgument<float>(detail::kPresetArgName)),
        out_grid_size_(spec.GetArgument<int>(detail::kOutputGridArgName)),
        hint_grid_size_(spec.GetArgument<int>(detail::kHintGridArgName)),
        enable_temporal_hints_(spec.GetArgument<bool>(detail::kEnableTemporalHintsArgName)),
        enable_external_hints_(spec.GetArgument<bool>(detail::kEnableExternalHintsArgName)),
        of_params_({quality_factor_, out_grid_size_, hint_grid_size_, enable_temporal_hints_,
                    enable_external_hints_}),
        optical_flow_(std::unique_ptr<optical_flow::OpticalFlowAdapter<ComputeBackend>>(
            new optical_flow::OpticalFlowStub<ComputeBackend>(of_params_))),
        image_type_(spec.GetArgument<DALIImageType>(detail::kImageTypeArgName)),
        device_id_(spec.GetArgument<int>("device_id")) {
    // In case external hints are enabled, we need 2 inputs
    DALI_ENFORCE((enable_external_hints_ && spec.NumInput() == 2) || !enable_external_hints_,
                 "Incorrect number of inputs. Expected: 2, Obtained: " +
                 std::to_string(spec.NumInput()));
    sync_ = CUDAEvent::Create(device_id_);
#if NVML_ENABLED
    nvml::Init();
#endif
  }

  ~OpticalFlow();
  DISABLE_COPY_MOVE_ASSIGN(OpticalFlow);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    if (enable_external_hints_) {
      const auto &hints = ws.Input<Backend>(1);
      // Extract calculation params
      ExtractParams(input, hints);
    } else {
      ExtractParams(input);
    }

    cudaStream_t of_stream = ws.stream();
    #if NVML_ENABLED
      {
        static float driver_version = nvml::GetDriverVersion();
        if (driver_version > 460 && driver_version < 470.21)
          of_stream = 0;
      }
    #else
      {
        int driver_cuda_version = 0;
        CUDA_CALL(cuDriverGetVersion(&driver_cuda_version));
        if (driver_cuda_version >= 11030 && driver_cuda_version < 11040)
          of_stream = 0;
      }
    #endif

    auto input_sh = input.shape();

    // sort all sequences by size, to make sure that samples are processed grouped by shape to avoid
    // NV OF reconfiguration as  much as possible
    processing_order_.resize(nsequences_);
    for (int sequence_idx = 0; sequence_idx < nsequences_; sequence_idx++) {
      processing_order_[sequence_idx] = {{input_sh[sequence_idx][2], input_sh[sequence_idx][1]},
                                          sequence_idx};
    }
    std::sort(processing_order_.begin(), processing_order_.end());

    of_lazy_init(input_sh[0][2], input_sh[0][1], depth_, image_type_, device_id_, of_stream);

    TensorListShape<> new_sizes(nsequences_, 4);
    for (int i = 0; i < nsequences_; i++) {
      auto out_shape = optical_flow_->CalcOutputShape(input_sh[i][1], input_sh[i][2]);
      auto shape = shape_cat(sequence_sizes_[i] - 1, out_shape);
      new_sizes.set_tensor_shape(i, shape);
    }
    output_desc.resize(1);
    output_desc[0] = {new_sizes, DALI_FLOAT};
    return true;
  }

  void RunImpl(Workspace &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:
  /**
   * Optical flow lazy initialization
   */
  void of_lazy_init(size_t width, size_t height, size_t channels, DALIImageType image_type,
                    int device_id, cudaStream_t stream) {
    std::call_once(of_initialized_,
                   [&]() {
                       optical_flow_.reset(
                               new optical_flow::OpticalFlowImpl(of_params_,
                                                                 width,
                                                                 height,
                                                                 channels,
                                                                 image_type,
                                                                 device_id,
                                                                 stream));
                        optical_flow_->Init(of_params_);
                   });
  }

  /**
   * Use input TensorList to extract calculation params
   * Only NFHWC layout is supported
   */
  void ExtractParams(const TensorList<Backend> &tl) {
    auto shape = tl.shape();
    nsequences_ = shape.size();
    DALI_ENFORCE(shape.sample_dim() == 4, "Input for Optical Flow must be a sequence of frames.");
    depth_ = shape[0][3];
    sequence_sizes_.reserve(nsequences_);
    for (int i = 0; i < nsequences_; i++) {
      sequence_sizes_[i] = shape[i][0];
    }

    for (auto sz : sequence_sizes_) {
      DALI_ENFORCE(sz >= 2, (sz == 1
                             ? "One-frame sequence encountered. Make sure that all input sequences "
                               "for Optical Flow have at least 2 frames."
                             : "Empty sequence encountered. Make sure that all input sequences"
                               " for Optical Flow have at least 2 frames."));
    }
  }


  /**
   * Overload for operator that takes also hints as input
   */
  void ExtractParams(const TensorList<Backend> &input, const TensorList<Backend> &hints) {
    ExtractParams(input);

    auto hints_shape = hints.shape();
    DALI_ENFORCE(hints_shape.sample_dim() == 3, "Hint should have a sample dim equal 3.");
    DALI_ENFORCE(hints_shape.size() == nsequences_,
                 "Number of input sequences and hints must match");
    hints_height_ = hints_shape[0][1];
    hints_width_ = hints_shape[0][2];
    hints_depth_ = hints_shape[0][3];
    DALI_ENFORCE(hints_depth_ == 2, "Hints shall have depth of 2: flow_x and flow_y");
    DALI_ENFORCE(
            hints_height_ == (frames_height_ + 3) / 4 && hints_width_ == (frames_width_ + 3) / 4,
            "Hints resolution has to be 4 times smaller in each dimension (4x4 grid)");
    DALI_ENFORCE(is_uniform(hints_shape),
                 "Width, height and depth must be equal for all hints");
  }

  struct DimsOrder {
    std::pair<int, int> dims;
    int idx;
    bool operator<(const DimsOrder &rhs) const {
      return dims < rhs.dims;
    }
  };

  const float quality_factor_;
  const int out_grid_size_;
  const int hint_grid_size_;
  const bool enable_temporal_hints_;
  const bool enable_external_hints_;
  std::once_flag of_initialized_;
  optical_flow::OpticalFlowParams of_params_;
  std::unique_ptr<optical_flow::OpticalFlowAdapter<ComputeBackend>> optical_flow_;
  DALIImageType image_type_;
  int device_id_;
  int frames_width_ = -1, frames_height_ = -1, depth_ = -1, nsequences_ = -1;
  int hints_width_ = -1, hints_height_ = -1, hints_depth_ = -1;
  std::vector<int> sequence_sizes_;
  std::vector<DimsOrder> processing_order_;
  CUDAEvent sync_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_H_
