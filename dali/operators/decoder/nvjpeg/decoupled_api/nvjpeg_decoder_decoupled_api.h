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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_DECOUPLED_API_H_
#define DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_DECOUPLED_API_H_

#include <nvjpeg.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <numeric>
#include <atomic>

#include "dali/pipeline/operator/operator.h"
#include "dali/operators/decoder/nvjpeg/decoupled_api/nvjpeg_helper.h"
#include "dali/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/util/image.h"
#include "dali/util/ocv.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/device_guard.h"

namespace dali {

using ImageInfo = EncodedImageInfo<int>;

class nvJPEGDecoder : public Operator<MixedBackend>, CachedDecoderImpl {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    CachedDecoderImpl(spec),
    output_image_type_(spec.GetArgument<DALIImageType>("output_type")),
    hybrid_huffman_threshold_(spec.GetArgument<unsigned int>("hybrid_huffman_threshold")),
    use_fast_idct_(spec.GetArgument<bool>("use_fast_idct")),
    output_info_(batch_size_),
    image_decoders_(batch_size_),
    image_states_(batch_size_),
    decode_params_(batch_size_),
    decoder_host_state_(batch_size_),
    decoder_huff_hybrid_state_(batch_size_),
    output_shape_(batch_size_, kOutputDim),
    pinned_buffers_(num_threads_*2),
    jpeg_streams_(num_threads_*2),
    device_buffers_(num_threads_),
    streams_(num_threads_),
    decode_events_(num_threads_),
    thread_page_ids_(num_threads_),
    device_id_(spec.GetArgument<int>("device_id")),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 spec.GetArgument<bool>("affine") /* pin threads */) {
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
    for (auto &stream : jpeg_streams_) {
      NVJPEG_CALL(nvjpegJpegStreamCreate(handle_, &stream));
    }
    for (auto &buffer : pinned_buffers_) {
      NVJPEG_CALL(nvjpegBufferPinnedCreate(handle_, nullptr, &buffer));
    }
    for (auto &buffer : device_buffers_) {
      NVJPEG_CALL(nvjpegBufferDeviceCreate(handle_, nullptr, &buffer));
    }
    for (auto &stream : streams_) {
      CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                             default_cuda_stream_priority_));
    }
    for (auto &event : decode_events_) {
      CUDA_CALL(cudaEventCreate(&event));
      CUDA_CALL(cudaEventRecord(event, streams_[0]));
    }
  }

  ~nvJPEGDecoder() override {
    try {
      thread_pool_.WaitForWork();
    } catch (const std::runtime_error &e) {
      std::cerr << "An error occurred in nvJPEG worker thread:\n"
                << e.what() << std::endl;
    }

    try {
      DeviceGuard g(device_id_);

      for (auto &stream : streams_) {
        CUDA_CALL(cudaStreamSynchronize(stream));
      }

      for (auto &state : decoder_host_state_) {
        NVJPEG_CALL(nvjpegJpegStateDestroy(state));
      }
      for (auto &state : decoder_huff_hybrid_state_) {
        NVJPEG_CALL(nvjpegJpegStateDestroy(state));
      }
      NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_host_));
      NVJPEG_CALL(nvjpegDecoderDestroy(decoder_huff_hybrid_));
      for (auto &stream  : jpeg_streams_) {
        NVJPEG_CALL(nvjpegJpegStreamDestroy(stream));
      }
      for (auto &buffer : pinned_buffers_) {
        NVJPEG_CALL(nvjpegBufferPinnedDestroy(buffer));
      }
      for (auto &buffer : device_buffers_) {
        NVJPEG_CALL(nvjpegBufferDeviceDestroy(buffer));
      }
      for (auto &params : decode_params_) {
        NVJPEG_CALL(nvjpegDecodeParamsDestroy(params));
      }
      for (auto &event : decode_events_) {
        CUDA_CALL(cudaEventDestroy(event));
      }
      for (auto &stream : streams_) {
        CUDA_CALL(cudaStreamDestroy(stream));
      }

      NVJPEG_CALL(nvjpegDestroy(handle_));
    } catch (const std::exception &e) {
      // If destroying nvJPEG resources failed we are leaking something so terminate
      std::cerr << "Fatal error: exception in ~nvJPEGDecoder():\n" << e.what() << std::endl;
      std::terminate();
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) override {
    return false;
  }

  using dali::OperatorBase::Run;
  void Run(MixedWorkspace &ws) override {
    SetupSharedSampleParams(ws);
    ParseImagesInfo(ws);
    ProcessImages(ws);
  }

 protected:
  virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  void ParseImagesInfo(MixedWorkspace &ws) {
    // Parsing and preparing metadata
    for (int i = 0; i < batch_size_; i++) {
      const auto &in = ws.Input<CPUBackend>(0, i);
      const auto *input_data = in.data<uint8_t>();
      const auto in_size = in.size();
      const auto file_name = in.GetSourceInfo();

      auto cached_shape = CacheImageShape(file_name);
      if (volume(cached_shape) > 0) {
        output_shape_.set_tensor_shape(i, {cached_shape[0], cached_shape[1], cached_shape[2]});
        continue;
      }

      ImageInfo info;
      nvjpegStatus_t ret = nvjpegGetImageInfo(handle_,
                                     static_cast<const unsigned char*>(input_data), in_size,
                                     &info.c, &info.subsampling,
                                     info.widths, info.heights);
      int64_t nchannels = NumberOfChannels(output_image_type_);
      info.nvjpeg_support = ret == NVJPEG_STATUS_SUCCESS;
      auto crop_generator = GetCropWindowGenerator(i);
      if (!info.nvjpeg_support) {
        try {
          const auto image = ImageFactory::CreateImage(
            static_cast<const uint8 *>(input_data), in_size, output_image_type_);
          const auto shape = image->PeekShape();
          info.heights[0] = shape[0];
          info.widths[0] = shape[1];
          if (output_image_type_ == DALI_ANY_DATA)
            nchannels = shape[2];

          if (crop_generator) {
            TensorShape<> shape{info.heights[0], info.widths[0]};
            info.crop_window = crop_generator(shape, "HW");
            DALI_ENFORCE(info.crop_window.IsInRange(shape));
            info.heights[0] = info.crop_window.shape[0];
            info.widths[0] = info.crop_window.shape[1];
          }
          output_info_[i] = info;
        } catch (const std::runtime_error &e) {
          DALI_FAIL(e.what() + "File: " + file_name);
        }
      } else {
        if (crop_generator) {
          TensorShape<> shape{info.heights[0], info.widths[0]};
          info.crop_window = crop_generator(shape, "HW");
          auto &crop_window = info.crop_window;
          DALI_ENFORCE(crop_window.IsInRange(shape));
          nvjpegDecodeParamsSetROI(decode_params_[i],
                                   crop_window.anchor[1], crop_window.anchor[0],
                                   crop_window.shape[1], crop_window.shape[0]);
          info.heights[0] = crop_window.shape[0];
          info.widths[0] = crop_window.shape[1];
        }

        if (ShouldUseHybridHuffman(info, input_data, in_size, hybrid_huffman_threshold_)) {
          image_decoders_[i] = decoder_huff_hybrid_;
          image_states_[i] = decoder_huff_hybrid_state_[i];
        } else {
          image_decoders_[i] = decoder_huff_host_;
          image_states_[i] = decoder_host_state_[i];
        }
      }
      output_shape_.set_tensor_shape(i, {info.heights[0], info.widths[0], nchannels});
      output_info_[i] = info;
    }
  }

  void ProcessImages(MixedWorkspace &ws) {
    // Creating output shape and setting the order of images so the largest are processed first
    // (for load balancing)
    std::vector<std::pair<size_t, size_t>> image_order(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      image_order[i] = std::make_pair(volume(output_shape_[i]), i);
    }
    std::sort(image_order.begin(), image_order.end(),
              std::greater<std::pair<size_t, size_t>>());

    auto& output = ws.Output<GPUBackend>(0);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output.set_type(type);
    output.Resize(output_shape_);
    output.SetLayout("HWC");

    for (int idx = 0; idx < batch_size_; ++idx) {
      const int i = image_order[idx].second;

      const auto &in = ws.Input<CPUBackend>(0, i);
      const auto file_name = in.GetSourceInfo();
      auto *output_data = output.mutable_tensor<uint8_t>(i);
      if (DeferCacheLoad(file_name, output_data))
        continue;

      ImageCache::ImageShape shape = output_shape_[i].to_static<3>();
      thread_pool_.DoWorkWithID(
        [this, i, file_name, &in, output_data, shape](int tid) {
          SampleWorker(i, file_name, in.size(), tid,
            in.data<uint8_t>(), output_data, streams_[tid]);
          CacheStore(file_name, output_data, shape, streams_[tid]);
        });
    }
    LoadDeferred(ws.stream());

    thread_pool_.WaitForWork();
    // wait for all work in workspace master stream
    for (int i = 0; i < num_threads_; ++i) {
      CUDA_CALL(cudaEventRecord(decode_events_[i], streams_[i]));
      CUDA_CALL(cudaStreamWaitEvent(ws.stream(), decode_events_[i], 0));
    }
  }

  inline int GetNextBufferIndex(int thread_id) {
    const int page = thread_page_ids_[thread_id];
    thread_page_ids_[thread_id] ^= 1;  // negate LSB
    return 2*thread_id + page;
  }

  // Per sample worker called in a thread of the thread pool.
  // It decodes the encoded image `input_data` (host mem) into `output_data` (device mem) with
  // nvJPEG. If nvJPEG can't handle the image, it falls back to CPU decoder implementation
  // with libjpeg.
  void SampleWorker(int sample_idx, string file_name, int in_size, int thread_id,
                    const uint8_t* input_data, uint8_t* output_data, cudaStream_t stream) {
    ImageInfo& info = output_info_[sample_idx];

    if (!info.nvjpeg_support) {
      HostFallback<StorageGPU>(input_data, in_size, output_image_type_, output_data,
                                        stream, file_name, info.crop_window, use_fast_idct_);
      return;
    }

    const int buff_idx = GetNextBufferIndex(thread_id);
    const int jpeg_stream_idx = buff_idx;
    NVJPEG_CALL(nvjpegStateAttachPinnedBuffer(image_states_[sample_idx],
                                              pinned_buffers_[buff_idx]));

    nvjpegStatus_t ret = nvjpegJpegStreamParse(handle_,
                                               static_cast<const unsigned char*>(input_data),
                                               in_size,
                                               false,
                                               false,
                                               jpeg_streams_[jpeg_stream_idx]);

    // If nvjpegJpegStreamParse failed we can skip nvjpeg's host decode step and
    // rely on the host decoder fallback
    if (ret == NVJPEG_STATUS_SUCCESS) {
      ret = nvjpegDecodeJpegHost(handle_,
                                image_decoders_[sample_idx],
                                image_states_[sample_idx],
                                decode_params_[sample_idx],
                                jpeg_streams_[jpeg_stream_idx]);
    }

    // If image is somehow not supported try host decoder
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

      CUDA_CALL(cudaEventSynchronize(decode_events_[thread_id]));
      NVJPEG_CALL_EX(nvjpegStateAttachDeviceBuffer(image_states_[sample_idx],
                                                   device_buffers_[thread_id]), file_name);

      NVJPEG_CALL_EX(nvjpegDecodeJpegTransferToDevice(
          handle_,
          image_decoders_[sample_idx],
          image_states_[sample_idx],
          jpeg_streams_[jpeg_stream_idx],
          stream), file_name);

      NVJPEG_CALL_EX(nvjpegDecodeJpegDevice(
          handle_,
          image_decoders_[sample_idx],
          image_states_[sample_idx],
          &nvjpeg_image,
          stream), file_name);
      CUDA_CALL(cudaEventRecord(decode_events_[thread_id], stream));

    } else {
      HostFallback<StorageGPU>(input_data, in_size, output_image_type_, output_data,
                                        stream, file_name, info.crop_window, use_fast_idct_);
    }
  }


  USE_OPERATOR_MEMBERS();
  nvjpegHandle_t handle_;

  // output colour format
  DALIImageType output_image_type_;

  unsigned int hybrid_huffman_threshold_;
  bool use_fast_idct_;

  // Common
  // Storage for per-image info
  std::vector<ImageInfo> output_info_;
  nvjpegJpegDecoder_t decoder_huff_host_;
  nvjpegJpegDecoder_t decoder_huff_hybrid_;

  // CPU
  // Per sample: lightweight
  std::vector<nvjpegJpegDecoder_t> image_decoders_;
  std::vector<nvjpegJpegState_t> image_states_;
  std::vector<nvjpegDecodeParams_t> decode_params_;
  std::vector<nvjpegJpegState_t> decoder_host_state_;
  std::vector<nvjpegJpegState_t> decoder_huff_hybrid_state_;
  TensorListShape<> output_shape_;

  // Per thread - double buffered
  std::vector<nvjpegBufferPinned_t> pinned_buffers_;
  std::vector<nvjpegJpegStream_t> jpeg_streams_;

  // GPU
  // Per thread
  std::vector<nvjpegBufferDevice_t> device_buffers_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> decode_events_;
  std::vector<int> thread_page_ids_;  // page index for double-buffering

  int device_id_;

  ThreadPool thread_pool_;
  static constexpr int kOutputDim = 3;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_DECOUPLED_API_NVJPEG_DECODER_DECOUPLED_API_H_
