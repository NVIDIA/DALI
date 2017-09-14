#ifndef NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_

#include <cstring>

#include <fstream>
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

  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline TJPGDecoder* Clone() const override {
    TJPGDecoder *new_decoder = new TJPGDecoder(color_);
    return new_decoder;
  }

  inline string name() const override {
    return "TurboJPEG Decoder";
  }
  
  DISABLE_COPY_MOVE_ASSIGN(TJPGDecoder);
protected:
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
  inline void RunPerDatumCPU(const Datum<Backend> &input, Datum<Backend> *output) override {
    // Dump the input image to file
    NDLL_ENFORCE(input.shape().size() == 3);
    const uint8 *img = input.template data<uint8>();
    int h = input.shape()[0];
    int w = input.shape()[1];
    int c = input.shape()[2];

    CUDA_ENFORCE(cudaDeviceSynchronize());
    uint8 *tmp = new uint8[h*w*c];

    CUDA_ENFORCE(cudaMemcpy2D(tmp, w*c*sizeof(uint8), img, w*c*sizeof(uint8),
            w*c*sizeof(uint8), h, cudaMemcpyDefault));

    static int i = 0;
    std::ofstream file(std::to_string(i) + ".jpg.txt");
    ++i;
    
    NDLL_ENFORCE(file.is_open());

    file << h << " " << w << " " << c << endl;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < c; ++k) {
          file << unsigned(tmp[i*w*c + j*c + k]) << " ";
        }
      }
      file << endl;
    }
    delete[] tmp;

    // Copy from input to output
    std::memcpy(output->raw_data(), input.raw_data(), input.nbytes());
  }
  
  inline vector<Index> InferOutputShapeFromShape(const vector<Index> &input_shape) override {
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
    return "Dump Image Op";
  }
  
protected:
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_pool_;
};
  
} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
