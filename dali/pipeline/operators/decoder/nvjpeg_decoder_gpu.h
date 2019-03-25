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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_GPU_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_GPU_H_

#include <functional>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/decoder/nvjpeg_helper.h"
#include "dali/util/ocv.h"

namespace dali {

using ImageInfo = EncodedImageInfo<unsigned int>;

class nvJPEGDecoderGPUStage : public Operator<MixedBackend> {
 public:
  explicit nvJPEGDecoderGPUStage(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HYBRID, &decoder_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_hybrid_));

    NVJPEG_CALL(nvjpegBufferDeviceCreate(handle_, nullptr, &device_buffer_));

    // Do we really need both in both stages ops?
    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));
  }

  ~nvJPEGDecoderGPUStage() noexcept(false) {
    NVJPEG_CALL(nvjpegBufferDeviceDestroy(device_buffer_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_host_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_hybrid_));
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace *ws) override {
    std::vector<Dims> output_shape(batch_size_);
    // Creating output shape and setting the order of images so the largest are processed first
    // (for load balancing)
    std::vector<std::pair<size_t, size_t>> image_order(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      const auto& info_tensor = ws->Input<CPUBackend>(0, i);
      const ImageInfo* info =
          reinterpret_cast<const ImageInfo*>(info_tensor.raw_data());
      int c = NumberOfChannels(output_image_type_);
      output_shape[i] = Dims({info->heights[0], info->widths[0], c});
      image_order[i] = std::make_pair(volume(output_shape[i]), i);
    }
    std::sort(image_order.begin(), image_order.end(),
              std::greater<std::pair<size_t, size_t>>());

    auto& output = ws->Output<GPUBackend>(0);
    output.Resize(output_shape);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output.set_type(type);

    for (auto& size_idx : image_order) {
      const int sample_idx = size_idx.second;

      const auto& info_tensor = ws->Input<CPUBackend>(0, sample_idx);
      const auto& state_tensor = ws->Input<CPUBackend>(1, sample_idx);
      const ImageInfo* info;
      const StateNvJPEG* nvjpeg_state;
      std::tie(info, nvjpeg_state) = GetInfoState(info_tensor, state_tensor);

      const auto file_name = info_tensor.GetSourceInfo();

      auto *output_data = output.mutable_tensor<uint8_t>(sample_idx);
      if (info->nvjpeg_support) {
        nvjpegImage_t nvjpeg_image;
        nvjpeg_image.channel[0] = output_data;
        nvjpeg_image.pitch[0] = NumberOfChannels(output_image_type_) * info->widths[0];

        nvjpegJpegState_t state = GetNvjpegState(*nvjpeg_state);

        NVJPEG_CALL(nvjpegStateAttachDeviceBuffer(state,
                              device_buffer_));

        nvjpegJpegDecoder_t decoder = GetDecoder(nvjpeg_state->nvjpeg_backend);
        NVJPEG_CALL_EX(nvjpegDecodeJpegTransferToDevice(
            handle_,
            decoder,
            state,
            nvjpeg_state->jpeg_stream,
            ws->stream()), file_name);

        NVJPEG_CALL_EX(nvjpegDecodeJpegDevice(
            handle_,
            decoder,
            state,
            &nvjpeg_image,
            ws->stream()), file_name);
      } else {
        // Fallback was handled by CPU op and wrote OpenCV ouput in Input #2
        // we just need to copy to device
        auto& in = ws->Input<CPUBackend>(2, sample_idx);
        const auto *input_data = in.data<uint8_t>();
        auto *output_data = output.mutable_tensor<uint8_t>(sample_idx);
        CUDA_CALL(cudaMemcpyAsync(output_data,
                    input_data,
                    info->heights[0] * info->widths[0] * NumberOfChannels(output_image_type_),
                    cudaMemcpyHostToDevice, ws->stream()));
      }
    }
  }

 protected:
  USE_OPERATOR_MEMBERS();

  inline std::pair<const ImageInfo*, const StateNvJPEG*>
  GetInfoState(const Tensor<CPUBackend>& info_tensor, const Tensor<CPUBackend>& state_tensor) {
    const ImageInfo* info =
          reinterpret_cast<const ImageInfo*>(info_tensor.raw_data());
    const StateNvJPEG* nvjpeg_state =
          reinterpret_cast<const StateNvJPEG*>(state_tensor.raw_data());
    return std::make_pair(info, nvjpeg_state);
  }

  inline nvjpegJpegDecoder_t GetDecoder(nvjpegBackend_t backend) const {
    switch (backend) {
      case NVJPEG_BACKEND_HYBRID:
        return decoder_host_;
      case NVJPEG_BACKEND_GPU_HYBRID:
        return decoder_hybrid_;
      default:
        DALI_FAIL("Unknown nvjpegBackend_t " + std::to_string(backend));
    }
  }

  // output colour format
  DALIImageType output_image_type_;

  // Common handles
  nvjpegHandle_t handle_;
  nvjpegJpegDecoder_t decoder_host_;
  nvjpegJpegDecoder_t decoder_hybrid_;

  nvjpegBufferDevice_t device_buffer_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_GPU_H_
