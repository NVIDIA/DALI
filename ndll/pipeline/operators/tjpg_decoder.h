#ifndef NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_

#include <cstring>

#include <fstream>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/decoder.h"
#include "ndll/pipeline/transformer.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename Backend>
class TJPGDecoder : public Decoder<Backend> {
public:
  inline TJPGDecoder(const OpSpec &spec) :
    Decoder<Backend>(spec),
    output_type_(spec.GetSingleArgument<NDLLImageType>("output_type", NDLL_RGB)),
    c_(IsColor(output_type_) ? 3 : 1) {}
  
  virtual inline ~TJPGDecoder() = default;
  
  inline vector<Index> InferOutputShape(
      const Sample<Backend> &input, int /* unused */, int /* unused */) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "TJPGDecoder expects 1D encoded jpeg strings as input");

    int h, w;
    NDLL_CALL(GetJPEGImageDims(input.template data<uint8>(), input.size(), &h, &w));
    return {h, w, c_};
  }

  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) override {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template mutable_data<uint8>();
  }
  
  inline string name() const override {
    return "TJPGDecoder";
  }
  
  DISABLE_COPY_MOVE_ASSIGN(TJPGDecoder);
protected:

  inline void RunPerSampleCPU(const Sample<Backend> &input,
      Sample<Backend> *output, int /* unused */, int /* unused */) override {
    
    NDLL_CALL(DecodeJPEGHost(input.template data<uint8>(), input.size(), output_type_,
            output->shape()[0], output->shape()[1], output->template mutable_data<uint8>()));
  }

  NDLLImageType output_type_;
  int c_;
};

template <typename Backend>
class DumpImageOp : public Transformer<Backend> {
public:
  inline DumpImageOp(const OpSpec &spec) :
    Transformer<Backend>(spec),
    suffix_(spec.GetSingleArgument<string>("suffix", "")),
    hwc_(spec.GetSingleArgument<bool>("hwc_format", true)) {}    
  
  virtual inline ~DumpImageOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int /* unused */, int /* unused */) override {
    return input_shape;
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    output->set_type(input_type);
  }
  
  inline string name() const override {
    return "DumpImageOp";
  }
  
protected:
  // This op forwards the data and writes it to files
  inline void RunPerSampleCPU(const Sample<Backend> &input,
      Sample<Backend> *output, int data_idx, int /* unused */) override {
    NDLL_ENFORCE(input.shape().size() == 3);

    if (input.type() == TypeInfo::Create<uint8>()) {
      DumpSampleHelper<uint8>(input, data_idx);
    } else if (input.type() == TypeInfo::Create<float16>()) {
      DumpSampleHelper<float16>(input, data_idx);
    } else if (input.type() == TypeInfo::Create<float>()) {
      DumpSampleHelper<float>(input, data_idx);
    } else {
      NDLL_FAIL("Unsupported data type.");
    }

    // Copy from input to output
    std::memcpy(output->raw_mutable_data(), input.raw_data(), input.nbytes());
  }

  template <typename T>
  inline void DumpSampleHelper(const Sample<Backend> &input, int data_idx) {
    // Dump the data to file
    const T *img = input.template data<T>();
    int h = input.shape()[0];
    int w = input.shape()[1];
    int c = input.shape()[2];
    if (hwc_) {
      DumpHWCToFile(img, h, w, c, std::to_string(data_idx) + suffix_);
    } else {
      DumpCHWToFile(img, h, w, c, std::to_string(data_idx) + suffix_);
    }
  }
  
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    if (input.type() == TypeInfo::Create<uint8>()) {
      DumpBatchHelper<uint8>(input);
    } else if (input.type() == TypeInfo::Create<float16>()) {
      DumpBatchHelper<float16>(input);
    } else if (input.type() == TypeInfo::Create<float>()) {
      DumpBatchHelper<float>(input);
    } else {
      NDLL_FAIL("Unsupported data type.");
    }
    
    output->Copy(input, Transformer<Backend>::stream_);
  }

  template <typename T>
  inline void DumpBatchHelper(const Batch<Backend> &input) {
    if (hwc_) {
      DumpHWCImageBatchToFile<T>(input, suffix_);
    } else {
      DumpCHWImageBatchToFile<T>(input, suffix_);
    }
  }
  
  const string suffix_;
  bool hwc_;
};
  
} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
