// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_H_

#include <nvjpeg.h>

#include <opencv2/opencv.hpp>
#include <array>
#include <map>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/util/device_guard.h"
#include "dali/util/image.h"


namespace dali {

#define NVJPEG_CALL(code)                                    \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      dali::string error = dali::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"";     \
      DALI_FAIL(error);                                      \
    }                                                        \
  } while (0)

namespace memory {

int DeviceNew(void **ptr, size_t size) {
  *ptr = GPUBackend::New(size, false);

  return 0;
}

int DeviceDelete(void *ptr) {
  GPUBackend::Delete(ptr, 0, false);

  return 0;
}

}  // namespace memory

inline nvjpegOutputFormat_t GetFormat(DALIImageType type) {
  switch (type) {
    case DALI_RGB:
      return NVJPEG_OUTPUT_RGBI;
    case DALI_BGR:
      return NVJPEG_OUTPUT_BGRI;
    case DALI_GRAY:
      return NVJPEG_OUTPUT_Y;
    default:
      DALI_FAIL("Unknown output format");
  }
}

inline int GetOutputPitch(DALIImageType type) {
  switch (type) {
    case DALI_RGB:
    case DALI_BGR:
      return 3;
    case DALI_GRAY:
      return 1;
    default:
      DALI_FAIL("Unknown output format");
  }
}

inline bool SupportedSubsampling(const nvjpegChromaSubsampling_t &subsampling) {
  switch (subsampling) {
    case NVJPEG_CSS_444:
    case NVJPEG_CSS_422:
    case NVJPEG_CSS_420:
    case NVJPEG_CSS_411:
    case NVJPEG_CSS_410:
    case NVJPEG_CSS_GRAY:
    case NVJPEG_CSS_440:
      return true;
    default:
      return false;
  }
}

class nvJPEGDecoder : public Operator<MixedBackend> {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator<MixedBackend>(spec),
    max_streams_(spec.GetArgument<int>("num_threads")),
    output_type_(spec.GetArgument<DALIImageType>("output_type")),
    output_shape_(batch_size_),
    output_info_(batch_size_),
    use_batched_decode_(spec.GetArgument<bool>("use_batched_decode")),
    batched_image_idx_(batch_size_),
    batched_output_(batch_size_),
    thread_pool_(max_streams_,
                 spec.GetArgument<int>("device_id"),
                 true /* pin threads */) {
      // Setup the allocator struct to use our internal allocator
      nvjpegDevAllocator_t allocator;
      allocator.dev_malloc = &memory::DeviceNew;
      allocator.dev_free = &memory::DeviceDelete;

      // create the handles, streams and events we'll use
      streams_.reserve(max_streams_);
      states_.reserve(max_streams_);
      events_.reserve(max_streams_);

      cudaGetDevice(&device_id_);

      NVJPEG_CALL(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &allocator, &handle_));
      for (int i = 0; i < max_streams_; ++i) {
        NVJPEG_CALL(nvjpegJpegStateCreate(handle_, &states_[i]));
        CUDA_CALL(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
        CUDA_CALL(cudaEventCreate(&events_[i]));
      }
      CUDA_CALL(cudaEventCreate(&master_event_));
  }

  ~nvJPEGDecoder() noexcept(false) {
    DeviceGuard g(device_id_);
    for (int i = 0; i < max_streams_; ++i) {
      NVJPEG_CALL(nvjpegJpegStateDestroy(states_[i]));
      CUDA_CALL(cudaEventDestroy(events_[i]));
      CUDA_CALL(cudaStreamDestroy(streams_[i]));
    }
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  using dali::OperatorBase::Run;

  void Run(MixedWorkspace *ws) override {
    // TODO(slayton): Is this necessary?
    // CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    CUDA_CALL(cudaEventRecord(master_event_, ws->stream()));
    for (int i = 0; i < max_streams_; ++i) {
      CUDA_CALL(cudaStreamWaitEvent(streams_[i], master_event_, 0));
    }

    // Get dimensions
    int idx_in_batch = 0;
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      EncodedImageInfo info;

      // Get necessary image information
      NVJPEG_CALL(nvjpegGetImageInfo(handle_,
                                     static_cast<const unsigned char*>(data), in_size,
                                     &info.c, &info.subsampling,
                                     info.widths, info.heights));

      // Store pertinent info for later
      const int image_depth = (output_type_ == DALI_GRAY) ? 1 : 3;
      output_shape_[i] = Dims({info.heights[0], info.widths[0], image_depth});
      output_info_[i] = info;

      // note if we can't use nvjpeg for this image
      if (!SupportedSubsampling(info.subsampling)) {
        output_info_[i].nvjpeg_support = false;
      } else {
        // Store the index for batched api
        batched_image_idx_[i] = idx_in_batch;
        output_info_[i].nvjpeg_support = true;
        idx_in_batch++;
      }
    }

    // Resize the output (contiguous)
    auto *output = ws->Output<GPUBackend>(0);
    output->Resize(output_shape_);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output->set_type(type);

    if (use_batched_decode_) {
      int images_in_batch = idx_in_batch;
      batched_output_.resize(images_in_batch);

      // setup this batch for nvjpeg with the number of images to be handled
      // by nvjpeg within this batch (!= batch_size if fallbacks are needed)
      NVJPEG_CALL(nvjpegDecodeBatchedInitialize(handle_,
                                                states_[0],
                                                images_in_batch,
                                                max_streams_,
                                                GetFormat(output_type_)));

      for (int i = 0; i < batch_size_; ++i) {
        auto& in = ws->Input<CPUBackend>(0, i);
        auto in_size = in.size();
        const auto *data = in.data<uint8_t>();
        auto *output_data = output->mutable_tensor<uint8_t>(i);

        auto &info = output_info_[i];

        // Setup outputs for images that will be processed via nvjpeg-batched
        if (info.nvjpeg_support) {
          batched_output_[batched_image_idx_[i]].channel[0] =
            output->mutable_tensor<uint8_t>(i);
          batched_output_[batched_image_idx_[i]].pitch[0] =
            GetOutputPitch(output_type_) * info.widths[0];
        }

        thread_pool_.DoWorkWithID(std::bind(
              [this, info, data, in_size, output_data](int idx, int tid) {
                DecodeSingleSampleHost(idx,
                                       batched_image_idx_[idx],
                                       tid,
                                       handle_,
                                       states_[0],
                                       info,
                                       data, in_size,
                                       output_data,
                                       streams_[0]);
              }, i, std::placeholders::_1));
      }
      // Sync thread-based work, assemble outputs and call batched
      thread_pool_.WaitForWork();

      // Mixed work
      NVJPEG_CALL(nvjpegDecodeBatchedPhaseTwo(handle_,
                                            states_[0],
                                            streams_[0]));

      // iDCT
      NVJPEG_CALL(nvjpegDecodeBatchedPhaseThree(handle_,
                                            states_[0],
                                            batched_output_.data(),
                                            streams_[0]));
    } else {
      // Loop over images again and decode
      for (int i = 0; i < batch_size_; ++i) {
        auto& in = ws->Input<CPUBackend>(0, i);
        auto in_size = in.size();
        const auto *data = in.data<uint8_t>();
        auto *output_data = output->mutable_tensor<uint8_t>(i);

        auto info = output_info_[i];

        thread_pool_.DoWorkWithID(std::bind(
              [this, info, data, in_size, output_data](int idx, int tid) {
                const int stream_idx = tid;
                DecodeSingleSample(idx,
                             stream_idx,
                             handle_,
                             states_[stream_idx],
                             info,
                             data, in_size,
                             output_data,
                             streams_[stream_idx]);
              }, i, std::placeholders::_1));
      }
      // Make sure work is finished being submitted
      thread_pool_.WaitForWork();
    }

    // ensure we're consistent with the main op stream
    for (int i = 0; i < max_streams_; ++i) {
      CUDA_CALL(cudaEventRecord(events_[i], streams_[i]));
      CUDA_CALL(cudaStreamWaitEvent(ws->stream(), events_[i], 0));
    }
  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

  struct EncodedImageInfo {
    bool nvjpeg_support;
    int c;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
  };

 protected:
  USE_OPERATOR_MEMBERS();

  // Decode a single sample end-to-end in a thread
  void DecodeSingleSample(int i,
                    int stream_idx,
                    nvjpegHandle_t handle,
                    nvjpegJpegState_t state,
                    const EncodedImageInfo &info,
                    const uint8 *data,
                    const size_t in_size,
                    uint8 *output,
                    cudaStream_t stream) {
    if (!info.nvjpeg_support) {
      OCVFallback(data, in_size, output, stream);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return;
    }

    nvjpegImage_t out_desc;
    out_desc.channel[0] = output;
    // out_desc.pitch = info.c * info.nWidthY;
    out_desc.pitch[0] = GetOutputPitch(output_type_) * info.widths[0];

    // Huffman Decode
    NVJPEG_CALL(nvjpegDecodePhaseOne(handle,
          state,
          data,
          in_size,
          GetFormat(output_type_),
          stream));

    // Ensure previous GPU work is finished
    CUDA_CALL(cudaStreamSynchronize(stream));

    // Memcpy of Huffman co-efficients to device
    NVJPEG_CALL(nvjpegDecodePhaseTwo(handle, state, stream));

    // iDCT and output
    NVJPEG_CALL(nvjpegDecodePhaseThree(handle, state, &out_desc, stream));
  }

  // Perform the CPU part of a batched decode on a single thread
  void DecodeSingleSampleHost(int i,
                              int nvjpeg_image_idx,
                              int thread_idx,
                              nvjpegHandle_t handle,
                              nvjpegJpegState_t state,
                              const EncodedImageInfo &info,
                              const uint8 *data,
                              const size_t in_size,
                              uint8 *output,
                              cudaStream_t stream) {
    if (!info.nvjpeg_support) {
      OCVFallback(data, in_size, output, stream);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return;
    }

    NVJPEG_CALL(nvjpegDecodeBatchedPhaseOne(handle,
                                       state,
                                       data,
                                       in_size,
                                       nvjpeg_image_idx,
                                       thread_idx,
                                       stream));
  }

  /**
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(const uint8_t* data, int size,
                   uint8_t *decoded_device_data, cudaStream_t s) {
    const int c = (output_type_ == DALI_GRAY) ? 1 : 3;
    auto decode_type = (output_type_ == DALI_GRAY) ? CV_LOAD_IMAGE_GRAYSCALE \
                                                   : CV_LOAD_IMAGE_COLOR;
    cv::Mat input(1,
                  size,
                  CV_8UC1,
                  reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(data)));
    cv::Mat tmp = cv::imdecode(input, decode_type);

    // Transpose BGR -> RGB if needed
    if (output_type_ == DALI_RGB) {
      cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    }

    CUDA_CALL(cudaMemcpyAsync(decoded_device_data,
                              tmp.ptr(),
                              tmp.rows * tmp.cols * c,
                              cudaMemcpyHostToDevice, s));
  }

  nvjpegHandle_t handle_;
  vector<nvjpegJpegState_t> states_;
  cudaEvent_t master_event_;
  vector<cudaStream_t> streams_;
  vector<cudaEvent_t> events_;

  // output colour format
  DALIImageType output_type_;

  // maximum number of streams to use to decode + convert
  const int max_streams_;

 protected:
  // Storage for per-image info
  vector<Dims> output_shape_;
  vector<EncodedImageInfo> output_info_;

  bool use_batched_decode_;
  // For batched API we need image index within the batch being
  // decoded by nvjpeg. If some images are falling back to OCV
  // this != the image index in the batch
  vector<int> batched_image_idx_;
  // output pointers
  vector<nvjpegImage_t> batched_output_;

  // Thread pool
  ThreadPool thread_pool_;

  // device id
  int device_id_;
};

}  // namespace dali
#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_H_
