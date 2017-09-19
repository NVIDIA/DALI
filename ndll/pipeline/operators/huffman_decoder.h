#ifndef NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_

#include <hybrid_decoder.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

/**
 * @brief Perform the host-side part of a hybrid cpu+gpu jpeg decode. This
 * decoder must be followed by the DCTQuantInvOp to complete the jpeg decode
 */
template <typename Backend>
class HuffmanDecoder : public Decoder<Backend> {
public:
  inline HuffmanDecoder() {}
  virtual inline ~HuffmanDecoder() = default;

  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int data_idx, int thread_idx) override {
    
  }
  
  inline vector<Index> InferOutputShape(
      const Datum<Backend> &input, int data_idx, int thread_idx) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "HuffmanDecoder expects 1D encoded jpeg strings as input");

    // Parse the input jpeg and save the output data
    parseRawJpegHost(
        input.template data<uint8>(),
        input.size(),
        &tl_parser_state_[thread_idx],
        &parsed_jpegs_[data_idx]);

  }

  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<int16>();
  }
  
  inline HuffmanDecoder* Clone() const override {
    return new HuffmanDecoder;
  }

  inline string name() const override {
    return "HuffmanDecoder";
  }

  inline void set_num_threads(int num_threads) override {
    num_threads_ = num_threads;
    tl_parser_state_.resize(num_threads);
  }

  inline void set_batch_size(int batch_size) override {
    batch_size_ = batch_size;
    parsed_jpegs_.resize(batch_size);
  }
  
  DISABLE_COPY_MOVE_ASSIGN(HuffmanDecoder);
protected:
  vector<ParsedJpeg> parsed_jpegs_;
  vector<JpegParserState> tl_parser_state_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
