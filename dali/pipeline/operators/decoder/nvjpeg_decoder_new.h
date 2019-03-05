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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_

#include <nvjpeg.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/decoder/nvjpeg_helper.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/util/thread_pool.h"

#define PINNED_DOUBLE_BUFFERING 1
namespace dali {

class nvJPEGDecoderNew : public Operator<MixedBackend> {
 public:
  explicit nvJPEGDecoderNew(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    output_info_(batch_size_),
    image_decoders_(batch_size_),
    image_states_(batch_size_),
    jpeg_streams_(batch_size_),
    decode_params_(batch_size_),
    decoder_host_state_(batch_size_),
    decoder_huff_hybrid_state_(batch_size_),
    output_shape_(batch_size_),
    #if PINNED_DOUBLE_BUFFERING
    pinned_buffer_(num_threads_ * 2),
    curr_pinned_buff_(num_threads_, 0),
    #else
    pinned_buffer_(num_threads_),
    #endif
    device_buffer_(num_threads_),
    streams_(num_threads_),
    events_(num_threads_),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 true /* pin threads */) {
    NVJPEG_CALL(nvjpegCreateSimple(&handle_));

    size_t device_memory_padding = spec.GetArgument<Index>("device_memory_padding");
    size_t host_memory_padding = spec.GetArgument<Index>("host_memory_padding");
    NVJPEG_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding, handle_));
    NVJPEG_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding, handle_));

    // to create also in GPU Op
    NVJPEG_CALL(nvjpegDecoderCreate(
                      handle_, NVJPEG_BACKEND_HYBRID, &decoder_huff_host_));
    NVJPEG_CALL(nvjpegDecoderCreate(
                      handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_huff_hybrid_));


    for (int i = 0; i < batch_size_; i++) {
      NVJPEG_CALL(nvjpegDecodeParamsCreate(handle_, &decode_params_[i]));
      NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(decode_params_[i],
                                                    GetFormat(output_image_type_)));
      NVJPEG_CALL(nvjpegDecodeParamsSetAllowCMYK(decode_params_[i], true));

      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &jpeg_streams_[i]));
      // We want to use nvJPEG default pinned allocator

      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_huff_host_,
                                        &decoder_host_state_[i]));
      NVJPEG_CALL(nvjpegDecoderStateCreate(handle_,
                                        decoder_huff_hybrid_,
                                        &decoder_huff_hybrid_state_[i]));
    }

    // GPU
    // create the handles, streams and events we'll use
    // We want to use nvJPEG default device allocator
#if PINNED_DOUBLE_BUFFERING
    for (int i = 0; i < num_threads_ * 2; ++i) {
#else
    for (int i = 0; i < num_threads_; ++i) {
#endif
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, nullptr, &pinned_buffer_[i]));
    }
    for (int i = 0; i < num_threads_; ++i) {
      NVJPEG_CALL(nvjpegBufferDeviceCreate(handle_, nullptr, &device_buffer_[i]));
      CUDA_CALL(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
      CUDA_CALL(cudaEventCreate(&events_[i]));
    }
    CUDA_CALL(cudaEventCreate(&master_event_));
  }

  virtual ~nvJPEGDecoderNew() noexcept(false) {
    try {
      thread_pool_.WaitForWork();
    } catch (...) {
      // As other images being process might fail, but we don't care
    }
    for (int i = 0; i < batch_size_; ++i) {
      NVJPEG_CALL(nvjpegJpegStreamDestroy(jpeg_streams_[i]));
      NVJPEG_CALL(nvjpegJpegStateDestroy(decoder_host_state_[i]));
      NVJPEG_CALL(nvjpegJpegStateDestroy(decoder_huff_hybrid_state_[i]));
    }
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_host_));
    NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_hybrid_));
#if PINNED_DOUBLE_BUFFERING
    for (int i = 0; i < num_threads_ * 2; ++i) {
#else
    for (int i = 0; i < num_threads_; ++i) {
#endif
      NVJPEG_CALL(nvjpegBufferPinnedDestroy(pinned_buffer_[i]));
    }

    for (int i = 0; i < num_threads_; ++i) {
      NVJPEG_CALL(nvjpegBufferDeviceDestroy(device_buffer_[i]));
      CUDA_CALL(cudaEventDestroy(events_[i]));
      CUDA_CALL(cudaStreamDestroy(streams_[i]));
    }
    CUDA_CALL(cudaEventDestroy(master_event_));
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace *ws) override {
    ParseImagesInfo(ws);
    ProcessImages(ws);
  }

 protected:
  void ParseImagesInfo(MixedWorkspace *ws) {
    // Parsing and preparing metadata
    for (int i = 0; i < batch_size_; i++) {
      const auto &in = ws->Input<CPUBackend>(0, i);
      const auto *input_data = in.data<uint8_t>();
      const auto in_size = in.size();
      const auto file_name = in.GetSourceInfo();

      nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                                 static_cast<const unsigned char*>(input_data),
                                                 in_size,
                                                 false,
                                                 false,
                                                 jpeg_streams_[i]);

      EncodedImageInfo info;
      info.nvjpeg_support = ret == NVJPEG_STATUS_SUCCESS;
      if (!info.nvjpeg_support) {
        try {
          const auto image = ImageFactory::CreateImage(
                                           static_cast<const uint8 *>(input_data), in_size);
          const auto dims = image->GetImageDims();
          info.heights[0] = std::get<0>(dims);
          info.widths[0] = std::get<1>(dims);
          info.nvjpeg_support = false;
          output_info_[i] = info;
        } catch (const std::runtime_error &e) {
          DALI_FAIL(e.what() + "File: " + file_name);
        }
      } else {
        NVJPEG_CALL(nvjpegJpegStreamGetFrameDimensions(jpeg_streams_[i],
                                                       info.widths,
                                                       info.heights));
        NVJPEG_CALL(nvjpegJpegStreamGetComponentsNum(jpeg_streams_[i],
                                                     &info.c));
        if (ShouldBeHybrid(info)) {
          image_decoders_[i] = decoder_huff_hybrid_;
          image_states_[i] = decoder_huff_hybrid_state_[i];
        } else {
          image_decoders_[i] = decoder_huff_host_;
          image_states_[i] = decoder_host_state_[i];
        }

        /*
        // TODO(spanev): add this function when integrating Crop
        nvjpegnvjpegDecodeParamsSetROI(decode_params_[pos], offset_x, offset_y, roi_w, roi_h);
        */
      }
      const auto c = static_cast<Index>(NumberOfChannels(output_image_type_));
      output_shape_[i] = Dims({info.heights[0], info.widths[0], c});
      output_info_[i] = info;
    }
  }

  void ProcessImages(MixedWorkspace* ws) {
    // Creating output shape and setting the order of images so the largest are processed first
    // (for load balancing)
    std::vector<std::pair<size_t, size_t>> image_order(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      image_order[i] = std::make_pair(volume(output_shape_[i]), i);
    }
    std::sort(image_order.begin(), image_order.end(),
              std::greater<std::pair<size_t, size_t>>());

    auto& output = ws->Output<GPUBackend>(0);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output.set_type(type);
    output.Resize(output_shape_);
    output.SetLayout(DALI_NHWC);

    CUDA_CALL(cudaEventRecord(master_event_, ws->stream()));
    for (int i = 0; i < num_threads_; ++i) {
      CUDA_CALL(cudaStreamWaitEvent(streams_[i], master_event_, 0));
    }

    for (int idx = 0; idx < batch_size_; ++idx) {
      const int i = image_order[idx].second;

      const auto &in = ws->Input<CPUBackend>(0, i);
      const auto file_name = in.GetSourceInfo();
      cudaStream_t stream = ws->stream();
      thread_pool_.DoWorkWithID(std::bind(
            [this, i, file_name, stream, &in, &output](int idx, int tid) {
              SampleWorker(i, file_name, in.size(), tid,
                              in.data<uint8_t>(), output.mutable_tensor<uint8_t>(i));
            }, i, std::placeholders::_1));
    }

    thread_pool_.WaitForWork();
    for (int i = 0; i < num_threads_; ++i) {
      CUDA_CALL(cudaEventRecord(events_[i], streams_[i]));
      CUDA_CALL(cudaStreamWaitEvent(ws->stream(), events_[i], 0));
    }
  }

  void SampleWorker(int i, string file_name, int in_size, int tid,
                    const uint8_t* input_data, uint8_t* output_data) {
    EncodedImageInfo& info = output_info_[i];

    if (!info.nvjpeg_support) {
      OCVFallback(input_data, in_size, output_data, streams_[tid], file_name);
      return;
    }

#if PINNED_DOUBLE_BUFFERING
    curr_pinned_buff_[tid] = (curr_pinned_buff_[tid] + 1) % 2;
    const int buff_idx = tid + (curr_pinned_buff_[tid] * num_threads_);
    NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(image_states_[i],
                                              pinned_buffer_[buff_idx]));
#else
    NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(image_states_[i],
                                              pinned_buffer_[tid]));

#endif


#if !PINNED_DOUBLE_BUFFERING
    CUDA_CALL(cudaEventSynchronize(events_[tid]));
#endif
    nvjpegStatus_t ret = nvjpegDecodeJpegHost(handle_,
                                              image_decoders_[i],
                                              image_states_[i],
                                              decode_params_[i],
                                              jpeg_streams_[i]);
    // If image is somehow not supported try hostdecoder
    if (ret != NVJPEG_STATUS_SUCCESS) {
      if (ret == NVJPEG_STATUS_JPEG_NOT_SUPPORTED || ret == NVJPEG_STATUS_BAD_JPEG) {
        info.nvjpeg_support = false;
      } else {
        NVJPEG_CALL_EX(ret, file_name);
      }
    }

    if (info.nvjpeg_support) {
      nvjpegImage_t nvjpeg_image;
      nvjpeg_image.channel[0] = output_data;
      nvjpeg_image.pitch[0] = NumberOfChannels(output_image_type_) * info.widths[0];

      NVJPEG_CALL(nvjpegStateAttachDeviceBuffer(image_states_[i], device_buffer_[tid]));

      NVJPEG_CALL(nvjpegDecodeJpegTransferToDevice(
          handle_,
          image_decoders_[i],
          image_states_[i],
          jpeg_streams_[i],
          streams_[tid]));

#if !PINNED_DOUBLE_BUFFERING
      // Next sample processed in this thread has to know when H2D finished
      CUDA_CALL(cudaEventRecord(events_[tid], streams_[tid]));
#endif


      NVJPEG_CALL(nvjpegDecodeJpegDevice(
          handle_,
          image_decoders_[i],
          image_states_[i],
          &nvjpeg_image,
          streams_[tid]));
    } else {
      OCVFallback(input_data, in_size, output_data, streams_[tid], file_name);
    }
  }

  bool ShouldBeHybrid(EncodedImageInfo& info) {
    return info.widths[0] * info.heights[0] > hybrid_huffman_threshold_;
  }
  /**
   // Running the CPU stage
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(const uint8_t *data, int size,
                   uint8_t *decoded_device_data, cudaStream_t s, string file_name) {
    const int c = (output_image_type_ == DALI_GRAY) ? 1 : 3;
    auto decode_type = (output_image_type_ == DALI_GRAY) ? cv::IMREAD_GRAYSCALE
                                                         : cv::IMREAD_COLOR;
    cv::Mat input(1,
                  size,
                  CV_8UC1,
                  reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(data)));
    cv::Mat tmp = cv::imdecode(input, decode_type);

    if (tmp.data == nullptr) {
      DALI_FAIL("Unsupported image type: " + file_name);
    }

    // Transpose BGR -> output_image_type_ if needed
    if (IsColor(output_image_type_) && output_image_type_ != DALI_BGR) {
      OpenCvColorConversion(DALI_BGR, tmp, output_image_type_, tmp);
    }

    CUDA_CALL(cudaMemcpyAsync(decoded_device_data,
                              tmp.ptr(),
                              tmp.rows * tmp.cols * c,
                              cudaMemcpyHostToDevice, s));
  }

  USE_OPERATOR_MEMBERS();
  nvjpegHandle_t handle_;

  // output colour format
  DALIImageType output_image_type_;

  unsigned hybrid_huffman_threshold_;

  // Common
  // Storage for per-image info
  std::vector<EncodedImageInfo> output_info_;
  nvjpegJpegDecoder_t decoder_huff_host_;
  nvjpegJpegDecoder_t decoder_huff_hybrid_;

  // CPU
  // Per sample: lightweight
  std::vector<nvjpegJpegDecoder_t> image_decoders_;
  std::vector<nvjpegJpegState_t> image_states_;
  std::vector<nvjpegJpegStream_t> jpeg_streams_;
  std::vector<nvjpegDecodeParams_t> decode_params_;
  std::vector<nvjpegJpegState_t> decoder_host_state_;
  std::vector<nvjpegJpegState_t> decoder_huff_hybrid_state_;
  std::vector<Dims> output_shape_;
  // Per thread
  std::vector<nvjpegBufferPinned_t> pinned_buffer_;
#if PINNED_DOUBLE_BUFFERING
  std::vector<uint8_t> curr_pinned_buff_;
#endif

  // GPU
  // Per thread
  std::vector<nvjpegBufferDevice_t> device_buffer_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;

  cudaEvent_t master_event_;

  ThreadPool thread_pool_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_NEW_H_
