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
#include <string>
#include <utility>
#include <cstring>  // For std::memcpy and std::memset
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include "dali/core/boundary.h"
#include "dali/core/span.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

// If the next frame is within this threshold, we skip samples instead of seeking
#define SEEK_THRESHOLD 10

namespace dali {

namespace impl {
template <typename Backend>
void memcpy(uint8_t *dst, const uint8_t *src, size_t size, cudaStream_t stream) {
  if constexpr (std::is_same_v<Backend, CPUBackend>) {
    std::memcpy(dst, src, size);
  } else {
    CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
  }
}

template <typename Backend>
void memset(uint8_t *dst, uint8_t value, size_t size, cudaStream_t stream) {
  if constexpr (std::is_same_v<Backend, CPUBackend>) {
    std::memset(dst, value, size);
  } else {
    CUDA_CALL(cudaMemsetAsync(dst, value, size, stream));
  }
}
}  // namespace impl

template <typename Backend, typename FramesDecoderImpl>
class DLL_PUBLIC VideoDecoderBase : public Operator<Backend> {
 private:
  struct FrameInfo {
    FrameInfo(int frame_idx, int frame_num, bool is_constant)
        : frame_idx(frame_idx), frame_num(frame_num), is_constant(is_constant) {}

    bool operator<(const FrameInfo &other) const {
      return frame_idx < other.frame_idx;
    }

    int frame_idx;
    int frame_num;
    bool is_constant = false;
  };

 public:
  USE_OPERATOR_MEMBERS();
  using InBackend = CPUBackend;
  using OutBackend =
      std::conditional_t<std::is_same_v<Backend, CPUBackend>, CPUBackend, GPUBackend>;

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
    } else if (pad_mode_str == "edge") {
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

    // Get frame selection parameters
    if (start_frame_.HasValue())
      start_frame_.Acquire(spec_, ws, batch_size);
    if (stride_.HasValue())
      stride_.Acquire(spec_, ws, batch_size);
    if (sequence_length_.HasValue())
      sequence_length_.Acquire(spec_, ws, batch_size);
    if (frames_.HasValue())
      frames_.Acquire(spec_, ws, batch_size);

    frames_decoders_.resize(batch_size);

    bool build_index = start_frame_.HasValue() || stride_.HasValue() ||
                       sequence_length_.HasValue() || frames_.HasValue();

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
            frames_decoders_[s] = CreateDecoder(data, size, build_index, source_info,
                                                ws.has_stream() ? ws.stream() : 0);
            DALI_ENFORCE(frames_decoders_[s]->IsValid(),
                         make_string("Failed to create video decoder for \"",
                                     frames_decoders_[s]->Filename(), "\""));
            auto sample_shape = out_shape.tensor_shape_span(s);
            int num_frames = frames_.HasValue()          ? frames_[s].shape[0] :
                             sequence_length_.HasValue() ? sequence_length_[s].data[0] :
                                                           frames_decoders_[s]->NumFrames();
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

            auto &ctx = ctx_[tid];
            ctx.frame_infos.clear();
            ctx.frame_infos.reserve(num_frames);

            if (frames_.HasValue()) {
              for (int64_t frame_num = 0; frame_num < num_frames; frame_num++) {
                ctx.frame_infos.emplace_back(frames_[s].data[frame_num], frame_num, false);
              }
            } else {
              int stride = stride_.HasValue() ? stride_[s].data[0] : 1;
              int start_frame = start_frame_.HasValue() ? start_frame_[s].data[0] : 0;
              int orig_num_frames = frames_decoders_[s]->NumFrames();
              for (int64_t frame_num = 0, frame_idx = start_frame; frame_num < num_frames;
                   frame_num++, frame_idx += stride) {
                if (frame_idx >= orig_num_frames) {
                  switch (boundary_type_) {
                    case boundary::BoundaryType::CLAMP:
                      ctx.frame_infos.emplace_back(orig_num_frames - 1, frame_num, false);
                      break;
                    case boundary::BoundaryType::REFLECT_1001:
                      ctx.frame_infos.emplace_back(
                          boundary::idx_reflect_1001<int>(static_cast<int>(frame_idx),
                                                          orig_num_frames),
                          frame_num, false);
                      break;
                    case boundary::BoundaryType::REFLECT_101:
                      ctx.frame_infos.emplace_back(
                          boundary::idx_reflect_101<int>(static_cast<int>(frame_idx),
                                                         orig_num_frames),
                          frame_num, false);
                      break;
                    case boundary::BoundaryType::CONSTANT:
                      ctx.frame_infos.emplace_back(-1, frame_num, true);
                      break;
                    default:
                    case boundary::BoundaryType::ISOLATED:
                      throw std::runtime_error("Unexpected out-of-bounds frame index: " +
                                               std::to_string(frame_idx));
                      break;
                  }
                } else {
                  ctx.frame_infos.emplace_back(frame_idx, frame_num, false);
                }
              }
            }
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

  bool DecodeFrames(span<const FrameInfo> frame_infos, SampleView<OutBackend> output,
                    int64_t sample_idx, cudaStream_t stream = 0) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();
    uint8_t *output_data = output.template mutable_data<uint8_t>();

    // Sort the frame indices so that we can decode them in natural order
    // to avoid unnecessary seeking when possible
    FrameInfo last_frame(-1, -1, false);
    for (auto &frame : frame_infos) {
      if (frame.is_constant) {
        impl::memset<OutBackend>(output_data + frame.frame_num * frame_size, pad_value_, frame_size,
                                 stream);
        continue;
      } else if (frame.frame_idx == last_frame.frame_idx) {
        impl::memcpy<OutBackend>(output_data + frame.frame_num * frame_size,
                                 output_data + last_frame.frame_num * frame_size, frame_size,
                                 stream);
        continue;
      }
      // Only seek if frame is not close to current position
      int current_frame = frames_decoder.NextFrameIdx();
      int frames_to_skip = frame.frame_idx - current_frame;

      if (frames_to_skip < 0 || frames_to_skip > SEEK_THRESHOLD) {
        // Frame is too far ahead or behind, need to seek
        frames_decoder.SeekFrame(frame.frame_idx);
      } else {
        for (int skip = 0; skip < frames_to_skip; skip++) {
          frames_decoder.ReadNextFrame(nullptr, false);
        }
      }

      if (frames_decoder.NextFrameIdx() == frame.frame_idx) {
        frames_decoder.ReadNextFrame(output_data + frame.frame_num * frame_size);
      } else {
        return false;
      }
      last_frame = frame;
    }
    return true;
  }

  std::vector<std::unique_ptr<FramesDecoderImpl>> frames_decoders_;

  std::unique_ptr<ThreadPool> thread_pool_;  // Used only for mixed backend
  boundary::BoundaryType boundary_type_ = boundary::BoundaryType::ISOLATED;
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
