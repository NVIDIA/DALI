#ifndef NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_

#include <cstring>

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
  
  shared_ptr<HybridJPEGDecodeChannel> channel_;
  vector<JpegParserState> tl_parser_state_;
  vector<HuffmanDecoderState> tl_huffman_state_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

template <typename Backend>
class DCTQuantInvOp : public Transformer<Backend> {
public:
  inline DCTQuantInvOp(bool color, shared_ptr<HybridJPEGDecodeChannel> channel) :
    color_(color), C_(color ? 3 : 1), channel_(channel) {
    NDLL_ENFORCE(channel != nullptr);

    // We need three buffers for our parameters
    batched_param_sizes_.resize(3);
    yuv_data_.template data<uint8>();
  }
  
  virtual inline ~DCTQuantInvOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(const vector<Index> &input_shape,
      int data_idx, int /* unused */) override {
    ParsedJpeg &jpeg = channel_->parsed_jpegs[data_idx];
    for (int i = 0; i < jpeg.components; ++i) {
      // We currently only support 8-bit quant tables
      NDLL_ENFORCE(jpeg.quantTables[i].nPrecision ==
          QuantizationTable::PRECISION_8_BIT,
          "Hybrid decode currently only supports 8-bit quantization tables");
      
      // save the yuv dims and dct steps
      yuv_dims_[data_idx*3 + i] = jpeg.yCbCrDims[i];
      dct_step_[data_idx*3 + i] = jpeg.dctLineStep[i];
    }
    // The output shape is determined by the encoded jpeg, whose meta-data
    // we access through the Channel connected to the huffman decoder
    return vector<Index>{jpeg.imgDims.height, jpeg.imgDims.width, C_};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<int16>(input_type));
    output->template data<uint8>();
  }
  
  inline DCTQuantInvOp* Clone() const override {
    return new DCTQuantInvOp(color_, channel_);
  }

  inline string name() const override {
    return "DCTQuantInvOp";
  }
  
  inline void set_num_threads(int num_threads) override {
    num_threads_ = num_threads;
  }

  inline void set_batch_size(int batch_size) override {
    batch_size_ = batch_size;
    num_component_ = batch_size_*C_;
    yuv_dims_.resize(num_component_);
    dct_step_.resize(num_component_);
    grid_info_.resize(num_component_);
    yuv_offsets_.resize(num_component_);
  }
  
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    // Run the batched kernel
    batchedDctQuantInv(
        batch_param_gpu_buffers_[1].template data<DctQuantInvImageParam>(),
        batch_param_gpu_buffers_[0].template data<uint8>(),
        batch_param_gpu_buffers_[2].template data<int>(),
        num_cuda_blocks_
        );
  }

  inline void CalculateBatchedParameterSize() override {
    int num_image_planes = 0;
    for (int i = 0; i < batch_size_; ++i) {
      // Note: Could do thread-wise reductions first
      num_image_planes += channel_->parsed_jpegs[i].components;
    }

    validateBatchedDctQuantInvParams(dct_step_.data(), yuv_dims_.data(), num_component_);
    getBatchedInvDctLaunchParams(yuv_dims_.data(), num_component_,
        &num_cuda_blocks_, grid_info_.data());

    // TODO(tgale): If we are outputting grayscale images, only do the idct
    // for the Y plane of the images and output the result directly to the
    // output batch
    batched_param_sizes_[0] = 64*num_component_; // quant tables
    batched_param_sizes_[1] = num_component_*sizeof(DctQuantInvImageParam); // dct params
    batched_param_sizes_[2] = num_cuda_blocks_;

    // Calculate the size of the YUV intermediate
    // data and resize the intermediate buffer
    size_t yuv_size = 0;
    for (int i = 0; i < num_component_; ++i) {
      yuv_offsets_[i] = yuv_size;
      yuv_size += yuv_dims_[i].height * yuv_dims_[i].width;
    }
    yuv_data_.Resize({(Index)yuv_size});
  }

  inline void SerialBatchedParameterSetup(const Batch<Backend> & /* unused */) override {
    // Setup image indices for batched idct kernel launch
    getBatchedInvDctImageIndices(yuv_dims_.data(),
        num_component_, batch_param_buffers_[2].template data<int>());
  }

  inline void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      int data_idx, int thread_idx) override {
    // Copy quant tables into mega-buffer
    ParsedJpeg &jpeg = channel_->parsed_jpegs[data_idx];
    for (int i = 0; i < jpeg.components; ++i) {
      int comp_id = data_idx*3 + i;
      std::memcpy(batch_param_buffers_[0].template data<uint8>() + (comp_id * 64),
          jpeg.quantTables[i].aTable.lowp, 64);
    }


    // Setup batched idct parameters
    //
    // TODO(tgale): Make sure we don't create params for invalid
    // components i.e. those unused by a grayscale image
    DctQuantInvImageParam *param = nullptr;
    for (int i = 0; i < 3; ++i) {
      int comp_id = data_idx*3 + i;
      param = &batch_param_buffers_[1].template data<DctQuantInvImageParam>()[i];
      param->src = static_cast<const int16*>(input.raw_datum(comp_id));
      param->srcStep = dct_step_[comp_id];
      param->dst = yuv_data_.template data<uint8>() + yuv_offsets_[comp_id];
      param->dstWidth = yuv_dims_[comp_id].width;
      param->gridInfo = grid_info_[comp_id];
    }
  }
  
  bool color_;
  int C_, num_component_;
  shared_ptr<HybridJPEGDecodeChannel> channel_;

  // image meta-data extracted from parsed jpegs
  vector<NppiSize> yuv_dims_;
  vector<int> dct_step_;
  int num_cuda_blocks_;
  vector<int2> grid_info_;

  // Stored intermediate result of idct+iquant step
  Tensor<Backend> yuv_data_;
  vector<int> yuv_offsets_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
  using Operator<Backend>::batched_param_sizes_;
  using Operator<Backend>::batch_param_buffers_;
  using Operator<Backend>::batch_param_gpu_buffers_;
};





} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_HUFFMAN_DECODER_H_
