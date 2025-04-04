// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_DECODER_VIDEO_DECODER_BASE_H_
#define DALI_OPERATORS_VIDEO_DECODER_VIDEO_DECODER_BASE_H_

#include <cuda_runtime.h>
#include <cstring>  // For std::memcpy and std::memset
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/boundary.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/common/memset.h"
#include "dali/operators/video/video_utils.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend, typename FramesDecoderImpl>
class DLL_PUBLIC VideoDecoderBase : public Operator<Backend> {
 public:
  USE_OPERATOR_MEMBERS();
  using InBackend = CPUBackend;
  using OutBackend =
      std::conditional_t<std::is_same_v<Backend, CPUBackend>, CPUBackend, GPUBackend>;
  using OutStorage = detail::storage_tag_map_t<OutBackend>;

  explicit VideoDecoderBase(const OpSpec &spec) : Operator<Backend>(spec) {
    if (spec_.HasArgument("device")) {
      auto device_str = spec_.template GetArgument<std::string>("device");
      if (device_str == "mixed") {
        thread_pool_ = std::make_unique<ThreadPool>(
            spec.GetArgument<int>("num_threads"), spec.GetArgument<int>("device_id"),
            spec.GetArgument<bool>("affine"), "VideoDecoder");
      }
    }

    DALI_ENFORCE(!(frames_.HasValue() && (start_frame_.HasValue() || stride_.HasValue() ||
                                          sequence_length_.HasValue() || end_frame_.HasValue())),
                 "Cannot specify both `frames` and any of `start_frame`, `sequence_length`, "
                 "`stride`, `end_frame` arguments");

    DALI_ENFORCE(!(sequence_length_.HasValue() && end_frame_.HasValue()),
                 "Cannot specify both `sequence_length` and `end_frame` arguments");

    boundary_type_ = GetBoundaryType(spec_);
    build_index_ = spec_.template GetArgument<bool>("build_index");

    if (boundary_type_ == boundary::BoundaryType::CONSTANT) {
      auto tmp = spec_.template GetRepeatedArgument<int>("fill_value");
      fill_value_.clear();
      for (auto value : tmp) {
        DALI_ENFORCE(value >= 0 && value <= 255, "fill_value must be in range [0, 255]");
        fill_value_.push_back(static_cast<uint8_t>(value));
      }
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(VideoDecoderBase);

 protected:
  struct WorkerContext {
    FramesDecoderImpl* frames_decoder = nullptr;
    cudaStream_t stream = 0;
    Tensor<OutBackend> constant_frame;

    WorkerContext() {
      if (std::is_same_v<Backend, CPUBackend>) {
        constant_frame.set_pinned(false);
      }
    }
  };

  std::unique_ptr<FramesDecoderImpl> CreateDecoder(const char *data, size_t size, bool build_index,
                                                   std::string_view source_info,
                                                   cudaStream_t stream = 0) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      return std::make_unique<FramesDecoderImpl>(data, size, source_info);
    } else {
      return std::make_unique<FramesDecoderImpl>(data, size, source_info, stream);
    }
  }

  void AcquireArguments(const Workspace &ws, int batch_size) {
    if (start_frame_.HasValue())
      start_frame_.Acquire(spec_, ws, batch_size);
    if (stride_.HasValue())
      stride_.Acquire(spec_, ws, batch_size);
    if (sequence_length_.HasValue())
      sequence_length_.Acquire(spec_, ws, batch_size);
    if (end_frame_.HasValue())
      end_frame_.Acquire(spec_, ws, batch_size);
    if (frames_.HasValue())
      frames_.Acquire(spec_, ws, batch_size);
  }

  /**
   * @brief Get the total number of frames for a given sample
   */
  int GetTotalNumFrames(int sample_idx) const {
    int num_frames = frames_decoders_[sample_idx]->NumFrames();
    return num_frames;
  }

  /**
   * @brief Get the stride for a given sample (default is 1)
   */
  int GetStride(int sample_idx) const {
    int stride = 1;
    if (stride_.HasValue()) {
      stride = stride_[sample_idx].data[0];
    }
    return stride;
  }

  /**
   * @brief Get the start frame for a given sample (default is 0)
   */
  int GetStartFrame(int sample_idx) const {
    int start = 0;
    if (start_frame_.HasValue()) {
      start = start_frame_[sample_idx].data[0];
    }
    return start;
  }

  /**
   * @brief Get the end frame for a given sample (default is the total number of frames)
   */
  int GetEndFrame(int sample_idx) const {
    int end = GetTotalNumFrames(sample_idx);
    if (end_frame_.HasValue()) {
      end = end_frame_[sample_idx].data[0];
    }
    return end;
  }

  /**
   * @brief Returns the number of frames that can be extracted from this video
   *
   * Calculates how many frames can be obtained when starting at the specified start frame
   * and advancing by the stride value until reaching the end of the video. For example,
   * with start_frame=2, stride=3, and a 10 frame video, this would return 3 since we can
   * extract frames [2,5,8].
   */
  int GetAvailableNumFrames(int sample_idx) const {
    int total_num_frames = GetTotalNumFrames(sample_idx);
    int start_frame = GetStartFrame(sample_idx);
    int stride = GetStride(sample_idx);
    int available_frames = std::max(0, total_num_frames - start_frame);
    int available_strided_frames = (available_frames + stride - 1) / stride;
    return available_strided_frames;
  }

  /**
   * @brief Returns the number of frames to be decoded for a given sample
   *
   * The sequence length is determined by the following priority:
   * 1. If `frames` argument exists, use its length
   * 2. If `end_frame` exists, calculate length from start_frame, end_frame and stride, capped by
   *    available frames if boundary type is ISOLATED
   * 3. If `sequence_length` argument exists, use its value, capped by available frames if boundary
   *    type is ISOLATED, and taking into account the start frame and stride
   * 4. Otherwise, use the total number of available frames (taking into account the start frame and
   *    stride)
   *
   * @remark When using `end_frame`, the sequence length is calculated as ceil((end - start) /
   * stride) to ensure all frames in [start, end) are included. For example, with start=0, end=10,
   * stride=3, the frames [0,3,6,9] will be extracted, resulting in a sequence length of 4.
   */
  int GetSequenceLength(int sample_idx) const {
    int sequence_len;
    if (frames_.HasValue()) {
      sequence_len = frames_[sample_idx].shape[0];
    } else if (sequence_length_.HasValue()) {
      sequence_len = sequence_length_[sample_idx].data[0];
      if (boundary_type_ == boundary::BoundaryType::ISOLATED) {
        sequence_len = std::min(sequence_len, GetAvailableNumFrames(sample_idx));
      }
    } else if (end_frame_.HasValue()) {
      int start = GetStartFrame(sample_idx);
      int end = GetEndFrame(sample_idx);
      if (boundary_type_ == boundary::BoundaryType::ISOLATED) {
        int end_of_video = GetTotalNumFrames(sample_idx);
        end = std::min(end, end_of_video);
      }
      int stride = GetStride(sample_idx);
      DALI_ENFORCE(end > start,
                   make_string("end_frame (", end, ") must be greater than start_frame (", start,
                               "), for sample #", sample_idx));
      sequence_len =
          (end - start + stride - 1) / stride;  // Round up to include all frames in [start,end)
    } else {
      sequence_len = GetAvailableNumFrames(sample_idx);
    }
    return sequence_len;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<InBackend>(0);

    // Validate input
    DALI_ENFORCE(input.type() == DALI_UINT8, "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1, "Input buffer must be 1-dimensional.");
    for (int64_t i = 0; i < input.num_samples(); ++i) {
      DALI_ENFORCE(input[i].shape().num_elements() > 0,
                   make_string("Incorrect sample at position: ", i, ". ",
                               "Video decoder does not support empty input samples."));
    }

    int batch_size = input.num_samples();
    AcquireArguments(ws, batch_size);
    frames_decoders_.resize(batch_size);

    // Create decoders in parallel
    ThreadPool &thread_pool = GetThreadPool(ws);
    ctx_.resize(thread_pool.NumThreads());
    TensorListShape<4> out_shape(batch_size);
    for (int s = 0; s < batch_size; ++s) {
      thread_pool.AddWork(
          [&, s](int tid) {
            const char *data = reinterpret_cast<const char *>(input[s].data<uint8_t>());
            size_t size = input[s].shape().num_elements();
            auto source_info = input.GetMeta(s).GetSourceInfo();
            frames_decoders_[s] = CreateDecoder(data, size, build_index_, source_info,
                                                ws.has_stream() ? ws.stream() : 0);
            DALI_ENFORCE(frames_decoders_[s]->IsValid(),
                         make_string("Failed to create video decoder for \"",
                                     frames_decoders_[s]->Filename(), "\""));
            auto sample_shape = out_shape.tensor_shape_span(s);
            int num_frames;
            if (GetStartFrame(s) >= GetTotalNumFrames(s)) {
              throw std::runtime_error("Start frame " + std::to_string(GetStartFrame(s)) +
                                       " is out of bounds for sample #" + std::to_string(s) +
                                       ", expected range [0, " +
                                       std::to_string(GetTotalNumFrames(s)) + ")");
            }

            if (frames_.HasValue()) {
              num_frames = frames_[s].shape[0];
            } else {
              num_frames = GetSequenceLength(s);
            }
            sample_shape[0] = num_frames;
            sample_shape[1] = frames_decoders_[s]->Height();
            sample_shape[2] = frames_decoders_[s]->Width();
            sample_shape[3] = frames_decoders_[s]->Channels();
          },
          -s);
    }
    thread_pool.RunAll();

    output_desc.resize(1);
    output_desc[0].shape = std::move(out_shape);
    output_desc[0].type = DALI_UINT8;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &output = ws.Output<OutBackend>(0);
    const auto &input = ws.Input<InBackend>(0);
    int batch_size = input.num_samples();

    // Decode samples in parallel
    ThreadPool &thread_pool = GetThreadPool(ws);
    for (int s = 0; s < batch_size; ++s) {
      thread_pool.AddWork(
          [&, s](int tid) {
            int64_t num_frames = output[s].shape()[0];  // It was calculated in SetupImpl
            int orig_num_frames = frames_decoders_[s]->NumFrames();
            auto &ctx = ctx_[tid];
            ctx.frames_decoder = frames_decoders_[s].get();
            ctx.stream = ws.has_stream() ? ws.stream() : 0;

            const uint8_t *constant_frame = boundary_type_ == boundary::BoundaryType::CONSTANT ?
                ConstantFrame(ctx.constant_frame, ctx.frames_decoder->FrameShape(),
                             make_cspan(fill_value_), ctx.stream, true) : nullptr;

            if (frames_.HasValue()) {
              frames_decoders_[s]->DecodeFrames(output[s].template mutable_data<uint8_t>(),
                                                make_cspan(frames_[s].data, frames_[s].shape[0]),
                                                boundary_type_, constant_frame);
            } else {
              auto start_frame = GetStartFrame(s);
              auto stride = GetStride(s);
              auto end_frame = start_frame + num_frames * stride;  // it can go beyond the video length
              frames_decoders_[s]->DecodeFrames(output[s].template mutable_data<uint8_t>(),
                                                start_frame, end_frame, stride,
                                                boundary_type_, constant_frame);
            }
            frames_decoders_[s].reset();
          },
          output[s].shape().num_elements());
    }
    thread_pool.RunAll();
  }

  ThreadPool &GetThreadPool(const Workspace &ws) {
    return std::is_same<MixedBackend, Backend>::value ? *thread_pool_ : ws.GetThreadPool();
  }

  std::vector<std::unique_ptr<FramesDecoderImpl>> frames_decoders_;

  std::unique_ptr<ThreadPool> thread_pool_;  // Used only for mixed backend
  boundary::BoundaryType boundary_type_ = boundary::BoundaryType::CONSTANT;
  SmallVector<uint8_t, 16> fill_value_;
  ArgValue<int> start_frame_{"start_frame", spec_};
  ArgValue<int> stride_{"stride", spec_};
  ArgValue<int> sequence_length_{"sequence_length", spec_};
  ArgValue<int> end_frame_{"end_frame", spec_};
  ArgValue<int, 1> frames_{"frames", spec_};
  bool build_index_;

  std::vector<WorkerContext> ctx_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_DECODER_VIDEO_DECODER_BASE_H_
