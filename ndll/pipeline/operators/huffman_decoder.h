#ifndef NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_

#include <hybrid_decoder.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/channel.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

/**
 * @brief Channel class to share data between the cpu & gpu
 * stages of the hybrid JPEG decoder.
 */ 
struct HybridJPEGDecodeChannel : public Channel {
  // Stores meta-data parsed from the input jpegs, including
  // quantization tables, per-component dct coefficient sizes
  // and strides, chrominance subsampling ratio, and output
  // image dimensions
  vector<ParsedJpeg> parsed_jpegs;
};

/**
 * @brief Perform the host-side part of a hybrid cpu+gpu jpeg decode. This
 * decoder must be followed by the DCTQuantInvOp to complete the jpeg decode
 */
template <typename Backend>
class HuffmanDecoder : public Decoder<Backend> {
public:
  inline HuffmanDecoder(shared_ptr<HybridJPEGDecodeChannel> channel) :
    channel_(channel) {
    NDLL_ENFORCE(channel != nullptr);
  }
  virtual inline ~HuffmanDecoder() = default;

  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int data_idx, int thread_idx) override {
    // Perform the huffman decode into the datum object
    HuffmanDecoderState &state = tl_huffman_state_[thread_idx];
    ParsedJpeg &jpeg = channel_->parsed_jpegs[data_idx];
    vector<int16*> dct_coeff_ptrs(jpeg.components);

    // Gather the pointers to each image component's dct coefficients
    int offset = 0;
    for (int i = 0; i < jpeg.components; ++i) {
      dct_coeff_ptrs[i] = output->template data<int16>() + offset;
      offset += jpeg.dctSize[i] / sizeof(int16);
    }

    // Perform the Huffman decode into the output buffer
    TimeRange _tr("HuffmanDecodePerImage");
    huffmanDecodeHost(jpeg, &state, &dct_coeff_ptrs);
  }
  
  inline vector<Index> InferOutputShape(
      const Datum<Backend> &input, int data_idx, int thread_idx) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "HuffmanDecoder expects 1D encoded jpeg strings as input");

    // Parse the input jpeg and save the output data
    ParsedJpeg &jpeg = channel_->parsed_jpegs[data_idx];
    parseRawJpegHost(
        input.template data<uint8>(),
        input.size(),
        &tl_parser_state_[thread_idx],
        &jpeg);

    // Note: The DCT coefficients for each image component can be different sizes
    // due to subsampling. 'Datum' objects do not support jagged data, so we pack
    // all the DCT coefficients into a single Datum and pass the needed meta-data
    // to the gpu-side jpeg decode through a Channel object
    int out_size = 0;
    for (int i = 0; i < jpeg.components; ++i) {
      // jpeg.dctSize[i] is in bytes, convert to elements
      out_size += jpeg.dctSize[i] / sizeof(int16);
    }
    return {out_size};
  }

  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<int16>();
  }
  
  inline HuffmanDecoder* Clone() const override {
    return new HuffmanDecoder(channel_);
  }

  inline string name() const override {
    return "HuffmanDecoder";
  }

  inline void set_num_threads(int num_threads) override {
    num_threads_ = num_threads;
    tl_parser_state_.resize(num_threads);
    tl_huffman_state_.resize(num_threads);
  }

  inline void set_batch_size(int batch_size) override {
    batch_size_ = batch_size;
    channel_->parsed_jpegs.resize(batch_size);
  }
  
  DISABLE_COPY_MOVE_ASSIGN(HuffmanDecoder);
protected:
  shared_ptr<HybridJPEGDecodeChannel> channel_;
  vector<JpegParserState> tl_parser_state_;
  vector<HuffmanDecoderState> tl_huffman_state_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
