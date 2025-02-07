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

#include <vector>
#include <memory>
#include <optional>
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

/**
 * Padding mode for handling missing frames in video sequences
 */
enum class PaddingMode {
  kNone,      // Return shorter sequences if not enough frames
  kConstant,  // Pad with constant value (default 0)
  kEdge       // Repeat last valid frame
};

namespace impl {
  template<typename Backend>
  void memcpy(uint8_t* dst, const uint8_t* src, size_t size, cudaStream_t stream) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      std::memcpy(dst, src, size);
    } else {
      CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    }
  }

  template<typename Backend>
  void memset(uint8_t* dst, uint8_t value, size_t size, cudaStream_t stream) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      std::memset(dst, value, size);
    } else {
      CUDA_CALL(cudaMemsetAsync(dst, value, size, stream));
    }
  }
}

template <typename Backend, typename FramesDecoderImpl>
class DLL_PUBLIC VideoDecoderBase : public Operator<Backend> {
 public:
  using InBackend = CPUBackend;
  using OutBackend = std::conditional_t<std::is_same_v<Backend, CPUBackend>,
                                        CPUBackend,
                                        GPUBackend>;

  VideoDecoderBase(const OpSpec &spec) : Operator<Backend>(spec) {
    auto pad_mode_str = spec_.template GetArgument<std::string>("pad_mode");
    if (pad_mode_str == "none") {
      padding_mode_ = PaddingMode::kNone;
    } else if (pad_mode_str == "constant") {
      padding_mode_ = PaddingMode::kConstant;
    } else if (pad_mode_str == "edge") {
      padding_mode_ = PaddingMode::kEdge;
    } else {
      DALI_FAIL(make_string("Invalid pad_mode: ", pad_mode_str));
    }

    if (padding_mode_ == PaddingMode::kConstant) {
      pad_value_ = spec_.template GetArgument<int>("pad_value");
      DALI_ENFORCE(pad_value_ >= 0 && pad_value_ <= 255,
                   "pad_value must be in range [0, 255]");
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(VideoDecoderBase);

 protected:
  std::unique_ptr<FramesDecoderImpl> CreateDecoder(const char* data, size_t size, 
                                                   bool build_index,
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
    DALI_ENFORCE(input.type() == DALI_UINT8,
                 "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1,
                 "Input buffer must be 1-dimensional.");
    for (int64_t i = 0; i < input.num_samples(); ++i) {
      DALI_ENFORCE(input[i].shape().num_elements() > 0,
                   make_string("Incorrect sample at position: ", i, ". ",
                             "Video decoder does not support empty input samples."));
    }

    int batch_size = input.num_samples();

    // Get frame selection parameters
    start_frame_.Acquire(spec_, ws, batch_size);
    stride_.Acquire(spec_, ws, batch_size);
    sequence_length_.Acquire(spec_, ws, batch_size);

    frames_decoders_.resize(batch_size);

    bool build_index = start_frame_.HasExplicitValue() ||
                      stride_.HasExplicitValue() ||
                      sequence_length_.HasValue();

    // Create decoders in parallel
    ThreadPool& thread_pool = GetThreadPool(ws);
    for (int s = 0; s < batch_size; ++s) {
      thread_pool.AddWork([&, s](int tid) {
        const char* data = reinterpret_cast<const char *>(input[s].data<uint8_t>());
        size_t size = input[s].shape().num_elements();
        auto source_info = input.GetMeta(s).GetSourceInfo();
        frames_decoders_[s] = CreateDecoder(data, size, build_index, source_info, ws.stream());
      });
    }
    thread_pool.RunAll();

    // Set output shape for each sample
    TensorListShape<4> out_shape(batch_size);
    for (int s = 0; s < batch_size; s++) {
      auto& decoder = frames_decoders_[s];
      DALI_ENFORCE(decoder->IsValid(), make_string("Failed to create video decoder for \"",
                                   decoder->Filename(), "\""));
      auto sample_shape = out_shape.tensor_shape_span(s);
      sample_shape[0] = sequence_length_.HasValue() ? sequence_length_[s].data[0] : frames_decoders_[s]->NumFrames();
      sample_shape[1] = frames_decoders_[s]->Height();
      sample_shape[2] = frames_decoders_[s]->Width();
      sample_shape[3] = frames_decoders_[s]->Channels();
    }

    output_desc.resize(1);
    output_desc[0].shape = out_shape;
    output_desc[0].type = DALI_UINT8;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &output = ws.Output<OutBackend>(0);
    const auto &input = ws.Input<InBackend>(0);
    int batch_size = input.num_samples();
    
    // Decode samples in parallel
    ThreadPool& thread_pool = GetThreadPool(ws);
    for (int s = 0; s < batch_size; ++s) {
      thread_pool.AddWork([&, s](int) {
        int64_t num_frames = output[s].shape()[0]; // It was calculated in SetupImpl
        DecodeFrames(output[s], s,
                     start_frame_[s].data[0],
                     num_frames,
                     stride_[s].data[0],
                     ws.stream());
        frames_decoders_[s].reset();
      }, input[s].shape().num_elements());
    }
    thread_pool.RunAll();
  }

  ThreadPool &GetThreadPool(const Workspace &ws) {
    return std::is_same<MixedBackend, Backend>::value ? *thread_pool_ : ws.GetThreadPool();
  }

  bool DecodeFrames(SampleView<OutBackend> output, int64_t sample_idx,
                    int64_t start_frame, int64_t sequence_length, int64_t stride,
                    cudaStream_t stream = 0) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();
    uint8_t *output_data = output.template mutable_data<uint8_t>();

    int64_t f = 0;
    if (start_frame > 0) {
      frames_decoder.SeekFrame(start_frame);
    }

    // Read frames with specified stride
    for (; f < sequence_length && frames_decoder.NextFrameIdx() != -1; f++) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
      for (int i = 1; i < stride && frames_decoder.NextFrameIdx() != -1; i++) {
        frames_decoder.ReadNextFrame(nullptr, false);
      }
    }

    // Handle padding if needed
    if (f < sequence_length) {
      switch (padding_mode_) {
        case PaddingMode::kNone:
          return false;
        case PaddingMode::kEdge:
          if (f > 0) {
            uint8_t* last_frame = output_data + (f-1) * frame_size;
            for (int64_t i = f; i < sequence_length; i++) {
              impl::memcpy<OutBackend>(output_data + i * frame_size, 
                                     last_frame, frame_size, stream);
            }
          }
          break;
        case PaddingMode::kConstant:
        default:
          impl::memset<OutBackend>(output_data + f * frame_size, 
                                 pad_value_, frame_size * (sequence_length - f), stream);
          break;
      }
    }
    return true;
  }

  std::vector<std::unique_ptr<FramesDecoderImpl>> frames_decoders_;

  std::unique_ptr<ThreadPool> thread_pool_;  // Used only for mixed backend
  PaddingMode padding_mode_ = PaddingMode::kConstant;
  uint8_t pad_value_ = 0;

  USE_OPERATOR_MEMBERS();
  ArgValue<int, 1> start_frame_{"start_frame", spec_};
  ArgValue<int, 1> stride_{"stride", spec_};
  ArgValue<int, 1> sequence_length_{"sequence_length", spec_};
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
