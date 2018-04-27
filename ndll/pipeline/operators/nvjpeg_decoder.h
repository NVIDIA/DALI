// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_



#include <nvJPEG.h>

#include <opencv2/opencv.hpp>
#include <array>
#include <map>
#include <vector>

#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/util/thread_pool.h"
#include "ndll/util/image.h"


namespace ndll {

#define NVJPEG_CALL(code)                                    \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      ndll::string error = ndll::string("NVJPEG error \"") + \
        std::to_string(static_cast<int>(status)) + "\"";     \
      NDLL_FAIL(error);                                      \
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

void convertToRGB(array<const Npp8u*, 3> YCbCr,
                  array<Npp32s, 3> steps,
                  int c, int rgb_step,
                  Npp8u* out,
                  int outH, int outW,
                  nvjpegChromaSubsampling sampling,
                  cudaStream_t stream) {
  NppiSize s;
  s.width = outW;
  s.height = outH;

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(stream);
  switch (sampling) {
    case NVJPEG_CSS_444:
      nppiYCbCr444ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps[0], out, c * outW, s);
      break;
    case NVJPEG_CSS_422:
      nppiYCbCr422ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c * outW, s);
      break;
    case NVJPEG_CSS_420:
      nppiYCbCr420ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c * outW, s);
      break;
    case NVJPEG_CSS_411:
      nppiYCbCr411ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c *outW, s);
      break;
    case NVJPEG_CSS_440:
    case NVJPEG_CSS_410:
    case NVJPEG_CSS_GRAY:
    case NVJPEG_CSS_UNKNOWN:
    default:
      NDLL_FAIL("Unsupported subsampling format");
  }

  // set the old stream for NPP
  nppSetStream(old_stream);
}

void convertToBGR(array<const Npp8u*, 3> YCbCr,
                  array<Npp32s, 3> steps,
                  int c, int rgb_step,
                  Npp8u* out,
                  int outH, int outW,
                  nvjpegChromaSubsampling sampling,
                  cudaStream_t stream) {
  NppiSize s;
  s.width = outW;
  s.height = outH;

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(stream);
  switch (sampling) {
    case NVJPEG_CSS_444:
      nppiYCbCr444ToBGR_JPEG_8u_P3C3R(YCbCr.data(), steps[0], out, c * outW, s);
      break;
    case NVJPEG_CSS_422:
      nppiYCbCr422ToBGR_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c * outW, s);
      break;
    case NVJPEG_CSS_420:
      nppiYCbCr420ToBGR_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c * outW, s);
      break;
    case NVJPEG_CSS_411:
      nppiYCbCr411ToBGR_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, c *outW, s);
      break;
    case NVJPEG_CSS_440:
    case NVJPEG_CSS_410:
    case NVJPEG_CSS_GRAY:
    case NVJPEG_CSS_UNKNOWN:
    default:
      NDLL_FAIL("Unsupported subsampling format");
  }

  // set the old stream for NPP
  nppSetStream(old_stream);
}

bool SupportedSubsampling(const nvjpegChromaSubsampling &subsampling) {
  switch (subsampling) {
    case NVJPEG_CSS_444:
    case NVJPEG_CSS_422:
    case NVJPEG_CSS_420:
      return true;
    default:
      return false;
  }
}

class nvJPEGDecoder : public Operator<Mixed> {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator<Mixed>(spec),
    max_streams_(spec.GetArgument<int>("max_streams")),
    output_type_(spec.GetArgument<NDLLImageType>("output_type")),
    Y_{max_streams_},
    Cb_{max_streams_},
    Cr_{max_streams_},
    output_shape_(batch_size_),
    output_sampling_(batch_size_),
    output_info_(batch_size_),
    thread_pool_(max_streams_,
                 spec.GetArgument<int>("device_id"),
                 true /* pin threads */) {
      // Setup the allocator struct to use our internal allocator
      nvjpegDevAllocator allocator;
      allocator.dev_malloc = &memory::DeviceNew;
      allocator.dev_free = &memory::DeviceDelete;

      // create the handles, streams and events we'll use
      streams_.reserve(max_streams_);
      handles_.reserve(max_streams_);
      events_.reserve(max_streams_);

      for (int i = 0; i < max_streams_; ++i) {
        NVJPEG_CALL(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &allocator, &handles_[i]));
        CUDA_CALL(cudaStreamCreate(&streams_[i]));
        CUDA_CALL(cudaEventCreate(&events_[i]));
      }
  }

  ~nvJPEGDecoder() {
    for (int i = 0; i < max_streams_; ++i) {
      NVJPEG_CALL(nvjpegDestroy(handles_[i]));
      CUDA_CALL(cudaEventDestroy(events_[i]));
      CUDA_CALL(cudaStreamDestroy(streams_[i]));
    }
  }

  void Run(MixedWorkspace *ws) override {
    // TODO(slayton): Is this necessary?
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));

    // Get dimensions
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
      nvjpegChromaSubsampling subsampling;

      // Get necessary image information
      NVJPEG_CALL(nvjpegGetImageInfo(handles_[i % max_streams_],
                                     static_cast<const unsigned char*>(data), in_size,
                                     &c,
                                     &nWidthY, &nHeightY,
                                     &nWidthCb, &nHeightCb,
                                     &nWidthCr, &nHeightCr,
                                     &subsampling));

      // Store pertinent info for later
      EncodedImageInfo info;
      info.c = c;
      info.nWidthY = nWidthY;
      info.nHeightY = nHeightY;
      info.nWidthCb = nWidthCb;
      info.nHeightCb = nHeightCb;
      info.nWidthCr = nWidthCr;
      info.nHeightCr = nHeightCr;

      const int image_depth = (output_type_ == NDLL_GRAY) ? 1 : 3;
      output_shape_[i] = Dims({nHeightY, nWidthY, image_depth});
      output_sampling_[i] = subsampling;
      output_info_[i] = info;

      // note if we can't use nvjpeg for this image
      if (!SupportedSubsampling(subsampling)) {
        ocv_fallback_indices_[i] = true;
      }
    }

    // Resize the output (contiguous)
    auto *output = ws->Output<GPUBackend>(0);
    output->Resize(output_shape_);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output->set_type(type);

    // Loop over images again and decode
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();
      auto *output_data = output->mutable_tensor<uint8_t>(i);

      int count = ocv_fallback_indices_.count(i);

      auto info = output_info_[i];

      thread_pool_.DoWorkWithID(std::bind(
            [this, info, data, in_size, output_data, count](int idx, int tid) {
              const int stream_idx = tid;
              DecodeSample(idx,
                           stream_idx,
                           handles_[stream_idx],
                           info,
                           data, in_size,
                           output_data,
                           (count > 0),  // Fallback if true
                           streams_[stream_idx]);
            }, i, std::placeholders::_1));
    }
    // Make sure work is finished being submitted
    thread_pool_.WaitForWork();

    // ensure we're consistent with the main op stream
    for (int i = 0; i < max_streams_; ++i) {
      CUDA_CALL(cudaEventRecord(events_[i], streams_[i]));
      CUDA_CALL(cudaStreamWaitEvent(ws->stream(), events_[i], 0));
    }
    // Make sure next iteration isn't unnecessarily falling back to
    // opencv based on old indices
    ocv_fallback_indices_.clear();
  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

  struct EncodedImageInfo {
    int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
  };

 protected:
  USE_OPERATOR_MEMBERS();

  void DecodeSample(int i,
                    int stream_idx,
                    nvjpegHandle_t handle,
                    const EncodedImageInfo &info,
                    const uint8 *data,
                    const size_t in_size,
                    uint8 *output,
                    bool ocvFallback,
                    cudaStream_t stream) {
    if (ocvFallback) {
      OCVFallback(data, in_size, output, stream);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return;
    }
    // Huffman Decode
    NVJPEG_CALL(nvjpegDecodePlanarCPU(handle,
          data,
          in_size));

    // Ensure previous GPU work is finished
    CUDA_CALL(cudaStreamSynchronize(streams_[stream_idx]));

    // Memcpy of Huffman co-efficients to device
    NVJPEG_CALL(nvjpegDecodePlanarMemcpy(handle, stream));

    Y_[stream_idx].Resize({info.nWidthY * info.nHeightY});
    Cb_[stream_idx].Resize({info.nWidthCb * info.nHeightCb});
    Cr_[stream_idx].Resize({info.nWidthCr * info.nHeightCr});

    // perform decode
    nvjpegImageOutputPlanar image_desc;
    image_desc.p1 = Y_[stream_idx].mutable_data<uint8>();
    image_desc.pitch1 = info.nWidthY;
    image_desc.p2 = Cb_[stream_idx].mutable_data<uint8>();
    image_desc.pitch2 = info.nWidthCb;
    image_desc.p3 = Cr_[stream_idx].mutable_data<uint8>();
    image_desc.pitch3 = info.nWidthCr;

    NVJPEG_CALL(nvjpegDecodePlanarGPU(handle,
          image_desc,
          NVJPEG_OUTPUT_YUV,
          stream));

    // copy & convert from YCbCr -> RGB
    Npp8u *out_ptr = reinterpret_cast<Npp8u*>(output);
    convertToRGB(
        array<const Npp8u*, 3>{reinterpret_cast<const Npp8u*>(Y_[stream_idx].raw_data()),
        reinterpret_cast<const Npp8u*>(Cb_[stream_idx].raw_data()),
        reinterpret_cast<const Npp8u*>(Cr_[stream_idx].raw_data())},
        array<Npp32s, 3>{info.nWidthY, info.nWidthCb, info.nWidthCr},  // steps
        info.c,
        info.c * info.nWidthY,  // RGB step
        out_ptr,  // output
        info.nHeightY, info.nWidthY,  // output H, W
        output_sampling_[i],   // sampling type
        stream);
  }

  /**
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(const uint8_t* data, int size,
                   uint8_t *decoded_device_data, cudaStream_t s) {
    const int c = (output_type_ == NDLL_GRAY) ? 1 : 3;
    auto decode_type = (output_type_ == NDLL_GRAY) ? CV_LOAD_IMAGE_GRAYSCALE \
                                                   : CV_LOAD_IMAGE_COLOR;
    cv::Mat input(1,
                  size,
                  CV_8UC1,
                  reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(data)));
    cv::Mat tmp = cv::imdecode(input, decode_type);

    // Transpose BGR -> RGB if needed
    if (output_type_ == NDLL_RGB) {
      cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    }

    CUDA_CALL(cudaMemcpyAsync(decoded_device_data,
                              tmp.ptr(),
                              tmp.rows * tmp.cols * c,
                              cudaMemcpyHostToDevice, s));
  }

  vector<nvjpegHandle_t> handles_;
  vector<cudaStream_t> streams_;
  vector<cudaEvent_t> events_;

  // output colour format
  NDLLImageType output_type_;

  // maximum number of streams to use to decode + convert
  const int max_streams_;

  // Storage for individual output components (per-stream)
  vector<Tensor<GPUBackend>> Y_;
  vector<Tensor<GPUBackend>> Cb_;
  vector<Tensor<GPUBackend>> Cr_;

 protected:
  // Storage for per-image info
  vector<Dims> output_shape_;
  vector<nvjpegChromaSubsampling> output_sampling_;
  vector<EncodedImageInfo> output_info_;

  // Images that nvjpeg can't handle
  std::map<int, bool> ocv_fallback_indices_;

  // Thread pool
  ThreadPool thread_pool_;
};

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
