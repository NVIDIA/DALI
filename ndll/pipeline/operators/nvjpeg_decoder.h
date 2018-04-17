#ifndef NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_

#include <nvJPEG.h>

#include <array>

#include <opencv2/opencv.hpp>
#include "ndll/pipeline/operator.h"
#include "ndll/util/image.h"

namespace ndll {

#define NVJPEG_CALL(code)                                    \
  do {                                                       \
    nvjpegStatus_t status = code;                            \
    if (status != NVJPEG_STATUS_SUCCESS) {                   \
      ndll::string error = ndll::string("NVJPEG error \"") + \
        std::to_string((int)status) + "\"";                  \
      NDLL_FAIL(error);                                      \
    }                                                        \
  } while(0)

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

  // if (outW & 1) s.width++;

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(stream);
  switch(sampling) {
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

  if (outW & 1) s.width++;

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(stream);
  switch(sampling) {
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

class nvJPEGDecoder : public Operator {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator(spec),
    output_type_(spec.GetArgument<NDLLImageType>("output_type")),
    Y_{max_streams_},
    Cb_{max_streams_},
    Cr_{max_streams_},
    output_shape_(batch_size_),
    output_sampling_(batch_size_),
    output_info_(batch_size_) {
      // Setup the allocator struct to use our internal allocator
      nvjpegDevAllocator allocator;
      allocator.dev_malloc = &memory::DeviceNew;
      allocator.dev_free = &memory::DeviceDelete;

      // create the handle
      NVJPEG_CALL(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &allocator, &handle_));
  }

  ~nvJPEGDecoder() {
    NVJPEG_CALL(nvjpegDestroy(handle_));
  }

  void Run(MixedWorkspace *ws) override {
    // Get dimensions
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
      nvjpegChromaSubsampling subsampling;

      // Get necessary image information
      NVJPEG_CALL(nvjpegGetImageInfo(handle_,
                                     static_cast<const unsigned char*>(data), in_size,
                                     &c,
                                     &nWidthY, &nHeightY,
                                     &nWidthCb, &nHeightCb,
                                     &nWidthCr, &nHeightCr,
                                     &subsampling));

      ImageInfo info;
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

    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      if (ocv_fallback_indices_.count(i)) {
        OCVFallback(i, data, in_size, output->mutable_tensor<uint8_t>(i), ws->stream());
        continue;
      }
      // Huffman Decode
      NVJPEG_CALL(nvjpegDecodePlanarCPU(handle_,
                                        data,
                                        in_size));

      // Memcpy of Huffman co-efficients to device
      NVJPEG_CALL(nvjpegDecodePlanarMemcpy(handle_,
                                           ws->stream()));

      const int stream_idx = i % max_streams_;

      auto info = output_info_[i];

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

      NVJPEG_CALL(nvjpegDecodePlanarGPU(handle_,
                                        image_desc,
                                        NVJPEG_OUTPUT_YUV,
                                        ws->stream()));

      // copy & convert from YCbCr -> RGB
      Npp8u *out_ptr = (Npp8u*)output->raw_mutable_tensor(i);
      convertToRGB(array<const Npp8u*, 3>{(const Npp8u*)Y_[stream_idx].raw_data(),
                                          (const Npp8u*)Cb_[stream_idx].raw_data(),
                                          (const Npp8u*)Cr_[stream_idx].raw_data()}, // YCbCr data pointers
                   array<Npp32s, 3>{info.nWidthY, info.nWidthCb, info.nWidthCr}, // steps
                   info.c,
                   info.c * info.nWidthY, // RGB step
                   out_ptr, // output
                   info.nHeightY, info.nWidthY, // output H, W
                   output_sampling_[i],  // sampling type
                   ws->stream());

      // WriteHWCImage(out_ptr, info.nHeightY, info.nWidthY, info.c, "tmp_new");
      CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    }

  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

 protected:
  USE_OPERATOR_MEMBERS();

  /**
   * Fallback to openCV's cv::imdecode for all images nvjpeg can't handle
   */
  void OCVFallback(int i, const uint8_t* data, int size, uint8_t *decoded_device_data, cudaStream_t s) {
    const int c = (output_type_ == NDLL_GRAY) ? 1 : 3;
    auto decode_type = (output_type_ == NDLL_GRAY) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    cv::Mat input(1, size, CV_8UC1, reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(data)));
    cv::Mat tmp = cv::imdecode(input, decode_type);

    // Transpose BGR -> RGB if needed
    if (output_type_ == NDLL_RGB) {
      cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    }

    CUDA_CALL(cudaMemcpyAsync(decoded_device_data, tmp.ptr(), tmp.rows * tmp.cols * c, cudaMemcpyHostToDevice, s));
  }

  nvjpegHandle_t handle_;

  // output colour format
  NDLLImageType output_type_;

  // maximum number of streams to use to decode + convert
  const int max_streams_ = num_threads_;

  vector<Tensor<GPUBackend>> Y_;
  vector<Tensor<GPUBackend>> Cb_;
  vector<Tensor<GPUBackend>> Cr_;

  struct ImageInfo {
    int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
  };

  // Storage for per-image info
  vector<Dims> output_shape_;
  vector<nvjpegChromaSubsampling> output_sampling_;
  vector<ImageInfo> output_info_;

  // Images that nvjpeg can't handle
  std::map<int, bool> ocv_fallback_indices_;
};

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
