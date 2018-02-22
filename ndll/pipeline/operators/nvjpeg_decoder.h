#ifndef NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_

#include <nvJPEG.h>

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

class nvJPEGDecoder : public Operator {
 public:
  explicit nvJPEGDecoder(const OpSpec& spec) :
    Operator(spec) {}

  void Run(MixedWorkspace *ws) override {
    for (int i = 0; i < batch_size_; ++i) {
      auto& in = ws->Input<CPUBackend>(0, i);
      auto in_size = in.size();
      const auto *data = in.data<uint8_t>();

      int c, nWidthY, nHeightY, nWidthCb, nHeightCb, nWidthCr, nHeightCr;
      auto err = nvjpegGetImageInfo(data, in_size,
                                    c,
                                    nWidthY, nHeightY,
                                    nWidthCb, nHeightCb,
                                    nWidthCr, nHeightCr);

      printf("%d %d %d %d %d %d\n", nWidthY, nHeightY,
                                    nWidthCb, nHeightCb,
                                    nWidthCr, nHeightCr);
      fflush(stdout);

    }

  }
  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoder);

 protected:
  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_NVJPEG_DECODER_H_
