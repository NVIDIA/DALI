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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_CPU_H_
#define DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_CPU_H_

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dali/operators/decoder/nvjpeg/decoupled_api/nvjpeg_helper.h"
#include "dali/operators/decoder/nvjpeg/decoupled_api/nvjpeg_allocator.h"

#include "dali/image/image_factory.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"

namespace dali {


using ImageInfo = EncodedImageInfo<unsigned int>;

using PinnedAllocator = memory::ChunkPinnedAllocator;

class nvJPEGDecoderCPUStage : public Operator<CPUBackend> {
 public:
  explicit nvJPEGDecoderCPUStage(const OpSpec& spec) :
    Operator<CPUBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    use_fast_idct_(spec.GetArgument<bool>("use_fast_idct")),
    decode_params_(batch_size_),
    use_chunk_allocator_(spec.GetArgument<bool>("use_chunk_allocator")) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    // Do we really need both in both stages ops?
    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HYBRID, &decoder_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_hybrid_));

    for (int i = 0; i < batch_size_; i++) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle_, &decode_params_[i]));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(decode_params_[i],
                                                    GetFormat(output_image_type_)));
      NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(decode_params_[i], true));
    }

    if (use_chunk_allocator_) {
      int nbuffers = spec.GetArgument<int>("cpu_prefetch_queue_depth") * batch_size_;
      PinnedAllocator::PreallocateBuffers(host_memory_padding, nbuffers);
      pinned_allocator_.pinned_malloc = &PinnedAllocator::Alloc;
      pinned_allocator_.pinned_free = &PinnedAllocator::Free;
    }
  }

  virtual ~nvJPEGDecoderCPUStage() {
    try {
      NVJPEG_CALL(nvjpegDecoderDestroy(decoder_host_));
      NVJPEG_CALL(nvjpegDecoderDestroy(decoder_hybrid_));
      for (int i = 0; i < batch_size_; i++) {
        NVJPEG_CALL(nvjpegDecodeParamsDestroy(decode_params_[i]));
      }
      NVJPEG_CALL(nvjpegDestroy(handle_));
      PinnedAllocator::FreeBuffers();
    } catch (const std::exception &e) {
      // If destroying nvJPEG resources failed we are leaking something so terminate
      std::cerr << "Fatal error: exception in ~nvJPEGDecoderCPUStage():\n" << e.what() << std::endl;
      std::terminate();
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    return false;
  }

  void RunImpl(SampleWorkspace &ws) override {
    const int data_idx = ws.data_idx();
    const auto& in = ws.Input<CPUBackend>(0);
    const auto *input_data = in.data<uint8_t>();
    const auto in_size = in.size();
    const auto file_name = in.GetSourceInfo();

    // We init and store the images and nvJPEG state in the CPU operator outputs
    // Since this CPU operator is instatiated on the GPU/Mixed boundary, it allows us to use
    // the Executor CPU output multiple-buffering

    // Output #0 contains general infomation about the sample
    // Output #1 contains nvJPEG state, buffer and information
    // Output #2 optionally contains OpenCV if nvJPEG cannot decode the sample

    ImageInfo* info;
    StateNvJPEG* state_nvjpeg;
    std::tie(info, state_nvjpeg) = InitAndGet(ws.Output<CPUBackend>(0),
                                              ws.Output<CPUBackend>(1));

    ws.Output<CPUBackend>(0).SetSourceInfo(file_name);

    nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                                static_cast<const unsigned char*>(input_data),
                                                in_size,
                                                false,
                                                false,
                                                state_nvjpeg->jpeg_stream);
    info->nvjpeg_support = ret == NVJPEG_STATUS_SUCCESS;
    auto crop_generator = GetCropWindowGenerator(data_idx);
    int64_t nchannels = NumberOfChannels(output_image_type_);
    if (!info->nvjpeg_support) {
      try {
        const auto image = ImageFactory::CreateImage(static_cast<const uint8 *>(input_data),
                                                     in_size, output_image_type_);
        const auto shape = image->PeekShape();
        info->heights[0] = shape[0];
        info->widths[0] = shape[1];
        if (output_image_type_ == DALI_ANY_DATA)
          nchannels = shape[2];

        if (crop_generator) {
          TensorShape<> shape{info->heights[0], info->widths[0]};
          info->crop_window = crop_generator(shape, "HW");
          DALI_ENFORCE(info->crop_window.IsInRange(shape));
          info->heights[0] = info->crop_window.shape[0];
          info->widths[0] = info->crop_window.shape[1];
        }
        auto& out = ws.Output<CPUBackend>(2);
        out.set_type(TypeInfo::Create<uint8_t>());
        out.SetLayout("HWC");
        out.Resize({info->heights[0], info->widths[0], nchannels});
        auto *output_data = out.mutable_data<uint8_t>();

        HostFallback<StorageCPU>(input_data, in_size, output_image_type_, output_data, 0,
                                          file_name, info->crop_window, use_fast_idct_);
      } catch (const std::runtime_error& e) {
        DALI_FAIL(e.what() + "File: " + file_name);
      }
    } else {
      NVJPEG_CALL(nvjpegJpegStreamGetFrameDimensions(state_nvjpeg->jpeg_stream,
                                                     info->widths,
                                                     info->heights));
      NVJPEG_CALL(nvjpegJpegStreamGetComponentsNum(state_nvjpeg->jpeg_stream,
                                                   &info->c));

      if (crop_generator) {
        TensorShape<> shape{info->heights[0], info->widths[0]};
        info->crop_window = crop_generator(shape, "HW");
        auto &crop_window = info->crop_window;
        DALI_ENFORCE(crop_window.IsInRange(shape));
        nvjpegDecodeParamsSetROI(decode_params_[data_idx],
          crop_window.anchor[1], crop_window.anchor[0], crop_window.shape[1], crop_window.shape[0]);
        info->widths[0] = crop_window.shape[1];
        info->heights[0] = crop_window.shape[0];
      }

      state_nvjpeg->nvjpeg_backend =
                    ShouldUseHybridHuffman(*info, input_data, in_size, hybrid_huffman_threshold_)
                    ? NVJPEG_BACKEND_GPU_HYBRID : NVJPEG_BACKEND_HYBRID;

      nvjpegJpegState_t state = GetNvjpegState(*state_nvjpeg);
      NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(state,
                                                state_nvjpeg->pinned_buffer));
      nvjpegStatus_t ret = nvjpegDecodeJpegHost(
          handle_,
          GetDecoder(state_nvjpeg->nvjpeg_backend),
          state,
          decode_params_[data_idx],
          state_nvjpeg->jpeg_stream);
      if (ret != NVJPEG_STATUS_SUCCESS) {
        if (ret == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || ret == NVJPEG_STATUS_BAD_JPEG) {
          info->nvjpeg_support = false;
        } else {
          NVJPEG_CALL_EX(ret, file_name);
        }
      } else {
        // TODO(spanev): free Output #2
      }
    }
  }

 protected:
  USE_OPERATOR_MEMBERS();

  virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  inline std::pair<ImageInfo*, StateNvJPEG*>
  InitAndGet(Tensor<CPUBackend>& info_tensor, Tensor<CPUBackend>& state_tensor) {
    if (info_tensor.size() == 0) {
      TypeInfo type;
      // we need to set a arbitrary to be able to access to call raw_data()
      type.SetType<uint8_t>();

      std::shared_ptr<ImageInfo> info_p(new ImageInfo());
      info_tensor.ShareData(info_p, sizeof(ImageInfo), {1});
      info_tensor.set_type(TypeInfo::Create<ImageInfo>());

      std::shared_ptr<StateNvJPEG> state_p(new StateNvJPEG(),
        [](StateNvJPEG* s) {
          NVJPEG_CALL(nvjpegJpegStreamDestroy(s->jpeg_stream));
          NVJPEG_CALL(nvjpegBufferPinnedDestroy(s->pinned_buffer));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_host_state));
          NVJPEG_CALL(nvjpegJpegStateDestroy(s->decoder_hybrid_state));
          delete s;
      });

      // We want to use nvJPEG default pinned allocator
      auto* allocator = use_chunk_allocator_ ? &pinned_allocator_ : nullptr;
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, allocator, &state_p->pinned_buffer));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_host_,
                                        &state_p->decoder_host_state));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_hybrid_,
                                        &state_p->decoder_hybrid_state));
      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &state_p->jpeg_stream));

      state_tensor.ShareData(state_p, sizeof(StateNvJPEG), {1});
      state_tensor.set_type(TypeInfo::Create<StateNvJPEG>());
    }

    ImageInfo* info = info_tensor.mutable_data<ImageInfo>();
    StateNvJPEG* nvjpeg_state = state_tensor.mutable_data<StateNvJPEG>();
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
        return nullptr;
    }
  }

  // output colour format
  DALIImageType output_image_type_;

  unsigned int hybrid_huffman_threshold_;
  // Used in host fallback
  bool use_fast_idct_;

  // Common handles
  nvjpegHandle_t handle_;
  nvjpegJpegDecoder_t decoder_host_;
  nvjpegJpegDecoder_t decoder_hybrid_;

  // TODO(spanev): add huffman hybrid decode
  std::vector<nvjpegDecodeParams_t> decode_params_;

  bool use_chunk_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_ = {};
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_CPU_H_
