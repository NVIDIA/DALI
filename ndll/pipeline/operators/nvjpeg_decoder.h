#ifndef NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_

#include <nvJPEG.h>

#include <array>

#include "ndll/pipeline/operator.h"

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
                  int rgb_step,
                  Npp8u* out,
                  int outH, int outW,
                  nvjpegChromaSubsampling sampling) {
  NppiSize s;
  s.width = outW;
  s.height = outH;

  if (outW & 1) s.width++;

  switch(sampling) {
   case NVJPEG_CSS_444:
     //printf("444\n");
    nppiYCbCr444ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps[0], out, 3*outW, s);
    break;
   case NVJPEG_CSS_422:
     //printf("422\n");
    nppiYCbCr422ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, 3*outW, s);
   case NVJPEG_CSS_420:
     //printf("420\n");
    nppiYCbCr420ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, 3*outW, s);
    break;
   default:
    NDLL_FAIL("boooo");
  }
}

class nvJPEGDecoder : public Operator {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator(spec),
    Y_{num_streams_},
    Cb_{num_streams_},
    Cr_{num_streams_},
    output_shape_(batch_size_),
    output_sampling_(batch_size_),
    output_info_(batch_size_) {
      // Setup the allocator struct to use our internal allocator
      nvjpegDevAllocator allocator;
      allocator.dev_malloc = &memory::DeviceNew;
      allocator.dev_free = &memory::DeviceDelete;

      // create the handle
      NVJPEG_CALL(nvjpegCreate(NVJPEG_BACKEND_HYBRID, &allocator, &handle_));
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
                                     data, in_size,
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

      // Huffman Decode
      NVJPEG_CALL(nvjpegDecodePlanarCPU(handle_,
                                        data,
                                        in_size));

      output_shape_[i] = Dims({nHeightY, nWidthY, c});
      output_sampling_[i] = subsampling;
      output_info_[i] = info;

      // Memcpy of Huffman co-efficients to device
      NVJPEG_CALL(nvjpegDecodePlanarMemcpy(handle_,
                                           ws->stream()));
    }

    // Resize the output (contiguous)
    auto *output = ws->Output<GPUBackend>(0);
    output->Resize(output_shape_);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output->set_type(type);

    // Now loop over all images again and decode
    for (int i = 0; i < batch_size_; ++i) {
      const int stream_idx = i % num_streams_;
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

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
                   3 * info.nWidthY, // RGB step
                   out_ptr, // output
                   info.nHeightY, info.nWidthY, // output H, W
                   output_sampling_[i]); // sampling type


    }

  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

 protected:
  USE_OPERATOR_MEMBERS();

  nvjpegHandle_t handle_;

  // maximum number of streams to use to decode + convert
  const int num_streams_ = num_threads_;

  vector<Tensor<GPUBackend>> Y_;
  vector<Tensor<GPUBackend>> Cb_;
  vector<Tensor<GPUBackend>> Cr_;

  struct ImageInfo {
    int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
  };

  vector<Dims> output_shape_;
  vector<nvjpegChromaSubsampling> output_sampling_;
  vector<ImageInfo> output_info_;
};

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
