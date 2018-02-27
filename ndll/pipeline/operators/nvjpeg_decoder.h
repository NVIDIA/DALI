#ifndef NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_

#include <nvJPEG.h>

#include <array>

#include "ndll/pipeline/operator.h"

namespace ndll {

/*
 * nvJPEG API copy
 *
int
nvjpegGetImageInfo(const unsigned char * pData, unsigned int nLength,
				   int & nComponent,
				   int & nWidthY,  int & nHeightY,
				   int & nWidthCb, int & nHeightCb,
				   int & nWidthCr, int & nHeightCr);

int
nvjpegDecode(const unsigned char * pData, unsigned int nLength,
	         unsigned char * pY,  int nStepY,
	         unsigned char * pCb, int nStepCb,
	         unsigned char * pCr, int nStepCr);
 */

enum Sampling {
  YCbCr_444 = 0,
  YCbCr_440 = 1,
  YCbCr_422 = 2,
  YCbCr_420 = 3,
  YCbCr_411 = 4,
  YCbCr_410 = 5,
  YCbCr_UNKNOWN = 6
};

void convertToRGB(array<const Npp8u*, 3> YCbCr,
                  array<Npp32s, 3> steps,
                  int rgb_step,
                  Npp8u* out,
                  int outH, int outW,
                  Sampling sampling) {
  NppiSize s;
  s.width = outW;
  s.height = outH;

  if (outW & 1) s.width++;

  switch(sampling) {
   case YCbCr_444:
     //printf("444\n");
    nppiYCbCr444ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps[0], out, 3*outW, s);
    break;
   case YCbCr_422:
     //printf("422\n");
    nppiYCbCr422ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, 3*outW, s);
   case YCbCr_420:
     //printf("420\n");
    nppiYCbCr420ToRGB_JPEG_8u_P3C3R(YCbCr.data(), steps.data(), out, 3*outW, s);
    break;
   default:
    printf("boooo\n");
  }
}

class nvJPEGDecoder : public Operator {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator(spec),
    Y_{max_streams_},
    Cb_{max_streams_},
    Cr_{max_streams_},
    output_shape_(batch_size_),
    output_sampling_(batch_size_) {}

  void Run(MixedWorkspace *ws) override {
    // Get dimensions
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
      auto err = nvjpegGetImageInfo(data, in_size,
                                   &c,
                                   &nWidthY, &nHeightY,
                                   &nWidthCb, &nHeightCb,
                                   &nWidthCr, &nHeightCr);

      Sampling sampling = getSamplingRatio(c,
                                           nWidthY, nHeightY,
                                           nWidthCb, nHeightCb,
                                           nWidthCr, nHeightCr);

      output_shape_[i] = Dims({nHeightY, nWidthY, c});
      output_sampling_[i] = sampling;
      // printf("output shape: %d %d %d\n", nHeightY, nWidthY, c);


#if 0
      printf("%d %d %d %d %d %d\n", nWidthY, nHeightY,
                                    nWidthCb, nHeightCb,
                                    nWidthCr, nHeightCr);
#endif
    }

    // Resize the output
    auto *output = ws->Output<GPUBackend>(0);
    output->Resize(output_shape_);
    TypeInfo type = TypeInfo::Create<uint8_t>();
    output->set_type(type);

    // Now loop over all images again and decode
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
      auto err = nvjpegGetImageInfo(data, in_size,
                                   &c,
                                   &nWidthY, &nHeightY,
                                   &nWidthCb, &nHeightCb,
                                   &nWidthCr, &nHeightCr);


      // temp buffers for decode output (later)
      const int stream_idx = i % max_streams_;
      Y_[stream_idx].Resize({nWidthY * nHeightY});
      Cb_[stream_idx].Resize({nWidthCb * nHeightCb});
      Cr_[stream_idx].Resize({nWidthCr * nHeightCr});

      // perform decode
      nvjpegDecode(data, in_size,
                   Y_[stream_idx].mutable_data<unsigned char>(), nWidthY,
                   Cb_[stream_idx].mutable_data<unsigned char>(), nWidthCb,
                   Cr_[stream_idx].mutable_data<unsigned char>(), nWidthCr);

      // copy & convert from YCbCr -> RGB

      Npp8u *out_ptr = (Npp8u*)output->raw_mutable_tensor(i);
      convertToRGB(array<const Npp8u*, 3>{(const Npp8u*)Y_[stream_idx].raw_data(),
                                          (const Npp8u*)Cb_[stream_idx].raw_data(),
                                          (const Npp8u*)Cr_[stream_idx].raw_data()}, // YCbCr data pointers
                   array<Npp32s, 3>{nWidthY, nWidthCb, nWidthCr}, // steps
                   3 * nWidthY, // RGB step
                   out_ptr, // output
                   nHeightY, nWidthY, // output H, W
                   output_sampling_[i]); // sampling type


    }

  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

 protected:
  USE_OPERATOR_MEMBERS();

  // maximum number of streams to use to decode + convert
  const int max_streams_ = num_threads_;

  vector<Tensor<GPUBackend>> Y_;
  vector<Tensor<GPUBackend>> Cb_;
  vector<Tensor<GPUBackend>> Cr_;


  vector<Dims> output_shape_;
  vector<Sampling> output_sampling_;

  Sampling getSamplingRatio(int components,
                            int yWidth, int yHeight,
                            int cbWidth, int cbHeight,
                            int crWidth, int crHeight) {
    if (components == 1) {
      return YCbCr_444;
    } else {
      Sampling eComponentSampling;
      // examine input sampling factor
      //
      // TODO(Trevor): Copied this code from ICE, why does this
      // always check if the different is less than 3?
      if(yWidth == cbWidth) {
          if (yHeight == cbHeight) {
              // cout << "selected 444" << endl;
              eComponentSampling = YCbCr_444;
          } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
              // cout << "selected 440" << endl;
              eComponentSampling = YCbCr_440;
          }
      }
      else if (abs(static_cast<float>(yWidth - 2 * cbWidth)) < 3) {
          if (yHeight == cbHeight) {
              // cout << "selected 422" << endl;
              eComponentSampling = YCbCr_422;
          } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
              // cout << "selected 420" << endl;
              eComponentSampling = YCbCr_420;
          }
      }
      else if (abs(static_cast<float>(yWidth - 4 * cbWidth)) < 4) {
          if (yHeight == cbHeight) {
              // cout << "selected 411" << endl;
              eComponentSampling = YCbCr_411;
          } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
              // cout << "selected 410" << endl;
              eComponentSampling = YCbCr_410;
          }
      }

      NDLL_ENFORCE(eComponentSampling != YCbCr_UNKNOWN, "Unknown subsampling ratio");

      return eComponentSampling;
    }
  }
};

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
