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

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_

#include <cuda_runtime.h>
#include <cstring>  // For std::memcpy and std::memset
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/boundary.h"
#include "dali/core/span.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/common/memset.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend, typename FramesDecoderImpl>
class DLL_PUBLIC VideoDecoderBase : public Operator<Backend> {
 private:
  struct FrameInfo {
    FrameInfo() : frame_num(-1), frame_idx(-1), constant_value(0) {}

    static FrameInfo FrameIndex(int frame_idx, int frame_num) {
      FrameInfo info;
      info.frame_num = frame_num;
      info.frame_idx = frame_idx;
      info.constant_value = 0;
      return info;
    }

    static FrameInfo Constant(uint8_t value, int frame_num) {
      FrameInfo info;
      info.frame_num = frame_num;
      info.frame_idx = -1;
      info.constant_value = value;
      return info;
    }

    bool operator<(const FrameInfo &other) const {
      return frame_idx < other.frame_idx;  // sort by frame index
    }

    bool is_constant() const {
      return frame_idx == -1;
    }

    int frame_num;
    int frame_idx;
    uint8_t constant_value;
  };

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

    DALI_ENFORCE(!(frames_.HasValue() &&
                   (start_frame_.HasValue() || stride_.HasValue() || sequence_length_.HasValue())),
                 "Cannot specify both `frames` and any of `start_frame`, `sequence_length`, "
                 "`stride` arguments");

    if (frames_.HasValue() && (spec_.HasArgument("pad_mode") || spec_.HasArgument("pad_value"))) {
      DALI_WARN(
          "Padding options (`pad_mode`, `pad_value`) are ignored when `frames` argument is "
          "provided");
    }

    auto pad_mode_str = spec_.template GetArgument<std::string>("pad_mode");
    if (pad_mode_str == "none" || pad_mode_str == "") {
      boundary_type_ = boundary::BoundaryType::ISOLATED;
    } else if (pad_mode_str == "constant") {
      boundary_type_ = boundary::BoundaryType::CONSTANT;
    } else if (pad_mode_str == "edge" || pad_mode_str == "repeat") {
      boundary_type_ = boundary::BoundaryType::CLAMP;
    } else if (pad_mode_str == "reflect_1001" || pad_mode_str == "symmetric") {
      boundary_type_ = boundary::BoundaryType::REFLECT_1001;
    } else if (pad_mode_str == "reflect_101" || pad_mode_str == "reflect") {
      boundary_type_ = boundary::BoundaryType::REFLECT_101;
    } else {
      DALI_FAIL(make_string("Invalid pad_mode: ", pad_mode_str, "\n",
                            "Valid options are: none, constant, edge, reflect_1001, reflect_101"));
    }

    if (boundary_type_ == boundary::BoundaryType::CONSTANT) {
      pad_value_ = spec_.template GetArgument<int>("pad_value");
      DALI_ENFORCE(pad_value_ >= 0 && pad_value_ <= 255, "pad_value must be in range [0, 255]");
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(VideoDecoderBase);

 protected:
  std::unique_ptr<FramesDecoderImpl> CreateDecoder(const char *data, size_t size, bool build_index,
                                                   std::string_view source_info,
                                                   cudaStream_t stream = 0) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      return std::make_unique<FramesDecoderImpl>(data, size, build_index, true, -1, source_info);
    } else {
      return std::make_unique<FramesDecoderImpl>(data, size, stream, build_index, -1, source_info);
    }
  }

  void AcquireArguments(const Workspace &ws, int batch_size) {
    if (start_frame_.HasValue())
      start_frame_.Acquire(spec_, ws, batch_size);
    if (stride_.HasValue())
      stride_.Acquire(spec_, ws, batch_size);
    if (sequence_length_.HasValue())
      sequence_length_.Acquire(spec_, ws, batch_size);
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
   * 2. If `sequence_length` argument exists, use its value, capped by available frames if boundary
   * type is ISOLATED, and taking into account the start frame and stride,
   * 3. Otherwise, use the total number of available frames (taking into account the start frame and
   * stride)
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
    } else {
      sequence_len = GetAvailableNumFrames(sample_idx);
    }
    return sequence_len;
  }

  bool ShouldBuildIndex(int sample_idx) const {
    int first_frame = GetStartFrame(sample_idx);
    if (frames_.HasValue()) {
      first_frame = frames_[sample_idx].data[0];
      for (int f = 0; f < frames_[sample_idx].shape[0]; f++) {
        first_frame = std::min(first_frame, frames_[sample_idx].data[f]);
      }
    }
    return first_frame > 0;
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
            frames_decoders_[s] = CreateDecoder(data, size, ShouldBuildIndex(s), source_info,
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
            ctx.frame_infos.clear();
            ctx.frame_infos.reserve(num_frames);

            if (frames_.HasValue()) {
              for (int64_t frame_num = 0; frame_num < num_frames; frame_num++) {
                if (frames_[s].data[frame_num] >= orig_num_frames) {
                  throw std::runtime_error("Unexpected out-of-bounds frame index " +
                                           std::to_string(frames_[s].data[frame_num]) +
                                           " for sample #" + std::to_string(s) +
                                           ", containing only " + std::to_string(orig_num_frames) +
                                           " frames");
                }
                ctx.frame_infos.push_back(
                    FrameInfo::FrameIndex(frames_[s].data[frame_num], frame_num));
              }
            } else {
              int stride = GetStride(s);
              int start_frame = GetStartFrame(s);
              for (int64_t frame_num = 0, frame_idx = start_frame; frame_num < num_frames;
                   frame_num++, frame_idx += stride) {
                if (frame_idx >= orig_num_frames) {
                  switch (boundary_type_) {
                    case boundary::BoundaryType::CLAMP:
                      ctx.frame_infos.push_back(
                          FrameInfo::FrameIndex(orig_num_frames - 1, frame_num));
                      break;
                    case boundary::BoundaryType::REFLECT_1001:
                      ctx.frame_infos.push_back(
                          FrameInfo::FrameIndex(boundary::idx_reflect_1001<int>(
                                                    static_cast<int>(frame_idx), orig_num_frames),
                                                frame_num));
                      break;
                    case boundary::BoundaryType::REFLECT_101:
                      ctx.frame_infos.push_back(
                          FrameInfo::FrameIndex(boundary::idx_reflect_101<int>(
                                                    static_cast<int>(frame_idx), orig_num_frames),
                                                frame_num));
                      break;
                    case boundary::BoundaryType::CONSTANT:
                      ctx.frame_infos.push_back(FrameInfo::Constant(pad_value_, frame_num));
                      break;
                    default:
                    case boundary::BoundaryType::ISOLATED:
                      break;  // will result in a shorter sequence than requested
                  }
                } else {
                  ctx.frame_infos.push_back(FrameInfo::FrameIndex(frame_idx, frame_num));
                }
              }
            }
            // Sort the frame indices so that we can decode them in natural order
            // to avoid unnecessary seeking when possible
            std::sort(ctx.frame_infos.begin(), ctx.frame_infos.end());
            DecodeFrames(make_cspan(ctx.frame_infos), output[s], s,
                         ws.has_stream() ? ws.stream() : 0);
            frames_decoders_[s].reset();
          },
          output[s].shape().num_elements());
    }
    thread_pool.RunAll();
  }

  ThreadPool &GetThreadPool(const Workspace &ws) {
    return std::is_same<MixedBackend, Backend>::value ? *thread_pool_ : ws.GetThreadPool();
  }

  void DecodeFrames(span<const FrameInfo> frame_infos, SampleView<OutBackend> output,
                    int64_t sample_idx, cudaStream_t stream = 0) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();
    uint8_t *output_data = output.template mutable_data<uint8_t>();

    FrameInfo last_frame;
    for (auto &frame : frame_infos) {
      if (frame.is_constant()) {
        kernels::memset<OutStorage>(output_data + frame.frame_num * frame_size,
                                    frame.constant_value, frame_size, stream);
        continue;
      } else if (frame.frame_idx == last_frame.frame_idx) {
        kernels::copy<OutStorage, OutStorage>(output_data + frame.frame_num * frame_size,
                                              output_data + last_frame.frame_num * frame_size,
                                              frame_size, stream);
        continue;
      }

      if (frame.frame_idx != frames_decoder.NextFrameIdx()) {
        frames_decoder.SeekFrame(frame.frame_idx);
        assert(frames_decoder.NextFrameIdx() == frame.frame_idx);
      }
      frames_decoder.ReadNextFrame(output_data + frame.frame_num * frame_size);
      last_frame = frame;
    }
  }

  std::vector<std::unique_ptr<FramesDecoderImpl>> frames_decoders_;

  std::unique_ptr<ThreadPool> thread_pool_;  // Used only for mixed backend
  boundary::BoundaryType boundary_type_ = boundary::BoundaryType::CONSTANT;
  uint8_t pad_value_ = 0;

  ArgValue<int> start_frame_{"start_frame", spec_};
  ArgValue<int> stride_{"stride", spec_};
  ArgValue<int> sequence_length_{"sequence_length", spec_};
  ArgValue<int, 1> frames_{"frames", spec_};

  struct WorkerContext {
    std::vector<FrameInfo> frame_infos;
  };
  std::vector<WorkerContext> ctx_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
