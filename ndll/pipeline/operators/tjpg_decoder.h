#ifndef NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_

#include <cstring>

#include <fstream>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/operator.h"
#include "ndll/util/image.h"

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
  
  inline vector<Index> InferOutputShape(
      const Datum<Backend> &input, int /* unused */, int /* unused */) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "TJPGDecoder expects 1D encoded jpeg strings as input");

    int h, w;
    GetJPEGImageDims(input.template data<uint8>(), input.size(), &h, &w);
    return {h, w, c_};
  }

  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) override {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline TJPGDecoder* Clone() const override {
    TJPGDecoder *new_decoder = new TJPGDecoder(color_);
    return new_decoder;
  }

  inline string name() const override {
    return "TJPGDecoder";
  }
  
  DISABLE_COPY_MOVE_ASSIGN(TJPGDecoder);
protected:

  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int /* unused */, int /* unused */) override {
    
    DecodeJPEGHost(input.template data<uint8>(), input.size(), color_,
        output->shape()[0], output->shape()[1], output->template data<uint8>());
  }
  
  bool color_;
  int c_;

  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_pool_;
};

template <typename Backend>
class DumpImageOp : public Transformer<Backend> {
public:
  inline DumpImageOp() {}
  virtual inline ~DumpImageOp() = default;

  // This op forwards the data and writes it to files
  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int data_idx, int /* unused */) override {
    NDLL_ENFORCE(input.shape().size() == 3);

    // Dump the data to file
    const uint8 *img = input.template data<uint8>();
    int h = input.shape()[0];
    int w = input.shape()[1];
    int c = input.shape()[2];
    DumpHWCToFile(img, h, w, c, w*c, std::to_string(data_idx));

    // Copy from input to output
    std::memcpy(output->raw_data(), input.raw_data(), input.nbytes());
  }
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int /* unused */, int /* unused */) override {
    return input_shape;
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline DumpImageOp* Clone() const override {
    return new DumpImageOp;
  }

  inline string name() const override {
    return "DumpImageOp";
  }
  
protected:
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_pool_;
};
  
} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
