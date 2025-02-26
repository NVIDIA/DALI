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
#include "dali/core/small_vector.h"

namespace dali {

template <typename Backend, typename FramesDecoderImpl>
class DLL_PUBLIC VideoDecoderBase : public Operator<Backend> {
 private:
  struct FrameInfo {
    static FrameInfo FrameIndex(int frame_idx, int frame_num) {
      FrameInfo info;
      info.frame_num = frame_num;
      info.frame_idx = frame_idx;
      return info;
    }

    static FrameInfo Constant(int frame_num) {
      FrameInfo info;
      info.frame_num = frame_num;
      return info;
    }

    bool operator<(const FrameInfo &other) const {
      return frame_idx < other.frame_idx;  // sort by frame index
    }

    bool is_constant() const {
      return frame_idx == -1;
    }

    int frame_num = 0;
    int frame_idx = -1;
  };

  struct WorkerContext {
    std::vector<FrameInfo> frame_infos;
    SmallVector<uint8_t, 16> fill_value;
    mm::uptr<uint8_t> fill_value_frame_cpu_buffer;
    mm::uptr<uint8_t> fill_value_frame_gpu_buffer;
    int64_t fill_value_frame_size = 0;
    FramesDecoderImpl* frames_decoder = nullptr;
    cudaStream_t stream = 0;
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

    DALI_ENFORCE(!(frames_.HasValue() && (start_frame_.HasValue() || stride_.HasValue() ||
                                          sequence_length_.HasValue() || end_frame_.HasValue())),
                 "Cannot specify both `frames` and any of `start_frame`, `sequence_length`, "
                 "`stride`, `end_frame` arguments");

    DALI_ENFORCE(!(sequence_length_.HasValue() && end_frame_.HasValue()),
                 "Cannot specify both `sequence_length` and `end_frame` arguments");

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
  std::unique_ptr<FramesDecoderImpl> CreateDecoder(const char *data, size_t size, bool build_index,
                                                   std::string_view source_info,
                                                   cudaStream_t stream = 0) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      return std::make_unique<FramesDecoderImpl>(data, size, build_index, -1, source_info);
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

  void AddFrame(int frame_num, int frame_idx, int orig_num_frames, WorkerContext &ctx) const {
    if (frame_idx >= orig_num_frames) {
      switch (boundary_type_) {
        case boundary::BoundaryType::CLAMP:
          ctx.frame_infos.push_back(FrameInfo::FrameIndex(orig_num_frames - 1, frame_num));
          break;
        case boundary::BoundaryType::REFLECT_1001:
          ctx.frame_infos.push_back(FrameInfo::FrameIndex(
              boundary::idx_reflect_1001<int>(static_cast<int>(frame_idx), orig_num_frames),
              frame_num));
          break;
        case boundary::BoundaryType::REFLECT_101:
          ctx.frame_infos.push_back(FrameInfo::FrameIndex(
              boundary::idx_reflect_101<int>(static_cast<int>(frame_idx), orig_num_frames),
              frame_num));
          break;
        case boundary::BoundaryType::CONSTANT:
          ctx.frame_infos.push_back(FrameInfo::Constant(frame_num));
          break;
        default:
        case boundary::BoundaryType::ISOLATED:
          break;  // will result in a shorter sequence than requested
      }
    } else {
      ctx.frame_infos.push_back(FrameInfo::FrameIndex(frame_idx, frame_num));
    }
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
            ctx.fill_value = fill_value_;
            ctx.frame_infos.clear();

            if (boundary_type_ == boundary::BoundaryType::CONSTANT) {
              int nfill_values = fill_value_.size();
              DALI_ENFORCE(nfill_values == 1 || nfill_values == ctx.frames_decoder->Channels(),
                           make_string("Constant padding requires either a single fill value or a "
                                       "fill value per channel. "
                                       "Got ",
                                       nfill_values, " fill values for a video with ",
                                       ctx.frames_decoder->Channels(), " channels."));
            }

            if (frames_.HasValue()) {
              for (int64_t frame_num = 0; frame_num < num_frames; frame_num++) {
                DALI_ENFORCE(
                    boundary_type_ != boundary::BoundaryType::ISOLATED ||
                        frames_[s].data[frame_num] < orig_num_frames,
                    make_string("Unexpected out-of-bounds frame index ", frames_[s].data[frame_num],
                                " for pad_mode = 'none' and sample #", s, ", containing only ",
                                orig_num_frames, " frames. Change `pad_mode` to fill the missing "
                                "frames."));
                AddFrame(frame_num, frames_[s].data[frame_num], orig_num_frames, ctx);
              }
            } else {
              int stride = GetStride(s);
              int start_frame = GetStartFrame(s);
              for (int64_t frame_num = 0, frame_idx = start_frame; frame_num < num_frames;
                   frame_num++, frame_idx += stride) {
                AddFrame(frame_num, frame_idx, orig_num_frames, ctx);
              }
            }
            // Sort the frame indices so that we can decode them in natural order
            // to avoid unnecessary seeking when possible
            std::sort(ctx.frame_infos.begin(), ctx.frame_infos.end());
            DecodeFrames(output[s], ctx);
            frames_decoders_[s].reset();
          },
          output[s].shape().num_elements());
    }
    thread_pool.RunAll();
  }

  ThreadPool &GetThreadPool(const Workspace &ws) {
    return std::is_same<MixedBackend, Backend>::value ? *thread_pool_ : ws.GetThreadPool();
  }

  template <typename Storage>
  void SetConstantPadding(uint8_t *output, int64_t output_size, WorkerContext &ctx) {
    if (ctx.fill_value.size() == 1) {
      kernels::memset<Storage>(output, ctx.fill_value[0], output_size, ctx.stream);
    } else if (std::is_same_v<Storage, StorageCPU>) {
      for (int64_t i = 0; i < output_size; i++) {
        output[i] = ctx.fill_value[i % ctx.fill_value.size()];
      }
    } else if (std::is_same_v<Storage, StorageGPU>) {
      bool is_same_fill_value = true;
      if (ctx.fill_value_frame_size >= static_cast<int64_t>(ctx.fill_value.size())) {
        for (size_t i = 0; i < ctx.fill_value.size(); i++) {
          if (ctx.fill_value_frame_cpu_buffer.get()[i] != ctx.fill_value[i]) {
            is_same_fill_value = false;
            break;
          }
        }
      }
      if (ctx.fill_value_frame_size < ctx.frames_decoder->FrameSize() || !is_same_fill_value) {
        ctx.fill_value_frame_size = ctx.frames_decoder->FrameSize();
        ctx.fill_value_frame_cpu_buffer =
            mm::alloc_raw_unique<uint8_t, mm::memory_kind::pinned>(ctx.fill_value_frame_size);
        ctx.fill_value_frame_gpu_buffer =
            mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(ctx.fill_value_frame_size);
        SetConstantPadding<StorageCPU>(ctx.fill_value_frame_cpu_buffer.get(),
                                       ctx.fill_value_frame_size, ctx);
        kernels::copy<StorageGPU, StorageCPU>(ctx.fill_value_frame_gpu_buffer.get(),
                                              ctx.fill_value_frame_cpu_buffer.get(),
                                              ctx.fill_value_frame_size, ctx.stream);
      }
      for (int64_t offset = 0; offset < output_size; offset += ctx.frames_decoder->FrameSize()) {
        assert(offset + ctx.frames_decoder->FrameSize() <= output_size);
        kernels::copy<StorageGPU, StorageGPU>(output + offset,
                                              ctx.fill_value_frame_gpu_buffer.get(),
                                              ctx.frames_decoder->FrameSize(), ctx.stream);
      }
    }
  }

  void DecodeFrames(SampleView<OutBackend> output, WorkerContext &ctx) {
    auto &frames_decoder = *ctx.frames_decoder;
    int64_t frame_size = frames_decoder.FrameSize();
    int num_fill_values = ctx.fill_value.size();
    uint8_t *output_data = output.template mutable_data<uint8_t>();

    FrameInfo last_frame{};
    for (auto &frame : ctx.frame_infos) {
      if (frame.is_constant()) {
        SetConstantPadding<OutStorage>(output_data + frame.frame_num * frame_size, frame_size, ctx);
      } else if (frame.frame_idx == last_frame.frame_idx) {
        kernels::copy<OutStorage, OutStorage>(output_data + frame.frame_num * frame_size,
                                              output_data + last_frame.frame_num * frame_size,
                                              frame_size, ctx.stream);
      } else {
        if (frame.frame_idx != frames_decoder.NextFrameIdx()) {
          frames_decoder.SeekFrame(frame.frame_idx);
          assert(frames_decoder.NextFrameIdx() == frame.frame_idx);
        }
        frames_decoder.ReadNextFrame(output_data + frame.frame_num * frame_size);
        last_frame = frame;
      }
    }
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

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
