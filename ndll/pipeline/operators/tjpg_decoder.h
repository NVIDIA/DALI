#ifndef NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_

#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend>
class TJPGDecoder : public Decoder<Backend> {
public:
  /**
   * @brief Constructs a turbo-jpeg decoder. Outputs RGB images if 
   * `color` == true, otherwise outputs grayscale images.
   */
  inline TJPGDecoder(bool color) : color_(color), c_(color ? 3 : 1) {}
  virtual inline ~TJPGDecoder() = default;

  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output) override {
    DecodeJPEGHost(input.template data<uint8>(), input.size(), color_,
        output->shape()[0], output->shape()[1], output->template data<uint8>());
  }
  
  inline vector<Index>
  InferOutputShape(const Datum<Backend> &input) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "TJPGDecoder expects 1D encoded jpeg strings as input");

    int h, w;
    GetJPEGImageDims(input.template data<uint8>(), input.size(), &h, &w);
    return {h, w, c_};
  }


  inline TJPGDecoder(TJPGDecoder &&op) noexcept {
    std::swap(color_, op.color_);
    std::swap(c_, op.c_);
  }
  inline TJPGDecoder& operator=(TJPGDecoder &&op) noexcept {
    if (&op != this) {
      std::swap(color_, op.color_);
      std::swap(c_, op.c_);
    }
    return *this;
  }
  
  TJPGDecoder(const TJPGDecoder&) = delete;
  TJPGDecoder& operator=(const TJPGDecoder&) = delete;
  
protected:
  bool color_;
  int c_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
