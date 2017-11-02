#ifndef NDLL_PIPELINE_OPERATORS_HYBRID_JPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_HYBRID_JPG_DECODER_H_

#include <cstring>

// TODO(tgale): Fix this include setup so that we can use this
// from external code
#include <third_party/hybrid_decode/include/hybrid_decoder.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/decoder.h"
#include "ndll/pipeline/transformer.h"
#include "ndll/util/image.h"

namespace ndll {

// Define the ParseJpeg struct as a ndll type so we can use it
// in Tensor objects to share data between the huffman & idct
// stages of the hybrid jpeg decode.
NDLL_REGISTER_TYPE(ParsedJpeg);

/**
 * @brief Perform the host-side part of a hybrid cpu+gpu jpeg decode. This
 * decoder must be followed by the DCTQuantInvOp to complete the jpeg decode
 */
template <typename Backend>
class HuffmanDecoder : public Decoder<Backend> {
public:
  inline HuffmanDecoder(const OpSpec &spec) :
    Decoder<Backend>(spec), jpeg_meta_(spec.ExtraOutput(0)) {
    // Resize per-image & per-thread data
    tl_parser_state_.resize(num_threads_);
    tl_huffman_state_.resize(num_threads_);
    
    jpeg_meta_->template mutable_data<ParsedJpeg>();
    jpeg_meta_->Resize({batch_size_});
  }
    
  virtual inline ~HuffmanDecoder() = default;
  
  inline vector<Index> InferOutputShape(
      const Sample<Backend> &input, int data_idx, int thread_idx) override {
    NDLL_ENFORCE(input.shape().size() == 1,
        "HuffmanDecoder expects 1D encoded jpeg strings as input");

    // Parse the input jpeg and save the output data
    ParsedJpeg &jpeg = jpeg_meta_->template mutable_data<ParsedJpeg>()[data_idx];
    parseRawJpegHost(
        input.template data<uint8>(),
        input.size(),
        &tl_parser_state_[thread_idx],
        &jpeg);

    // Note: The DCT coefficients for each image component can be different sizes
    // due to subsampling. 'Sample' objects do not support jagged data, so we pack
    // all the DCT coefficients into a single Sample and pass the needed meta-data
    // to the gpu-side jpeg decode through our extra output Tensor.
    int out_size = 0;
    for (int i = 0; i < jpeg.components; ++i) {
      // jpeg.dctSize[i] is in bytes, convert to elements
      out_size += jpeg.dctSize[i] / sizeof(int16);
    }
    return {out_size};
  }

  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template mutable_data<int16>();
  }
  
  inline string name() const override {
    return "HuffmanDecoder";
  }

  DISABLE_COPY_MOVE_ASSIGN(HuffmanDecoder);
protected:
  inline void RunPerSampleCPU(const Sample<Backend> &input,
      Sample<Backend> *output, int data_idx, int thread_idx) override {
    // Perform the huffman decode into the sample object
    HuffmanDecoderState &state = tl_huffman_state_[thread_idx];
    ParsedJpeg &jpeg = jpeg_meta_->template mutable_data<ParsedJpeg>()[data_idx];
    vector<int16*> dct_coeff_ptrs(jpeg.components);

    // Gather the pointers to each image component's dct coefficients
    int offset = 0;
    for (int i = 0; i < jpeg.components; ++i) {
      dct_coeff_ptrs[i] = output->template mutable_data<int16>() + offset;
      offset += jpeg.dctSize[i] / sizeof(int16);
    }

    // Perform the Huffman decode into the output buffer
    huffmanDecodeHost(jpeg, &state, &dct_coeff_ptrs);
  }

  shared_ptr<Tensor<CPUBackend>> jpeg_meta_;
  vector<JpegParserState> tl_parser_state_;
  vector<HuffmanDecoderState> tl_huffman_state_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
};

// Note: Our handling of grayscale images is a bit nuanced. In the case that we are outputting
// color (rgb) images, we always rely on the paramters setup calls from hybrid_decoder (i.e.
// validateBatchedDctQuantInvParams, getBatchedInvDctLaunchParams, getBatchedInvDctImageIndices)
// to handle grayscale images that may be mixed into our input batch. Even for grayscale images
// (images with a single component), we create parameters for all three components. The component
// indices that are set up by the hybrid_decoder calls will take into account that these components
// are non-existent and create no indices for these components. This is a small waste of memory
// with our extra parameters, but greatly simplifies the code so it is ok for now.
//
// When we are outputting grayscale images, we only bother to do the idcts for the chrominance
// planes of the jpegs. We do this into our intermediate buffer because the output of the idct
// is still aligned to jpeg block stuff, and then we just perform dev-to-dev copies to get
// the final grayscale images into the output buffer
template <typename Backend>
class DCTQuantInvOp : public Transformer<Backend> {
public:
  inline DCTQuantInvOp(const OpSpec &spec) :
    Transformer<Backend>(spec),
    output_type_(spec.GetSingleArgument<NDLLImageType>("output_type", NDLL_RGB)),
    color_(IsColor(output_type_)), C_(color_ ? 3 : 1),
    jpeg_meta_(spec.ExtraInput(0)) {
    // Make sure the jpeg meta tensor has already been setup
    NDLL_ENFORCE(jpeg_meta_->shape() == vector<Index>{batch_size_}, "Invalid extra "
        "input Tensor. HuffmanDecoder must be added to the pipeline before the "
        "DCTQuantInvOp");
    NDLL_ENFORCE(IsType<ParsedJpeg>(jpeg_meta_->type()), "Invalid extra input Tensor. "
        "HuffmanDecoder must be added to the pipeline before the DCTQuantInvOp");
      
    // Resize per-thread & per-image data
    num_component_ = batch_size_*C_;    
    yuv_dims_.resize(num_component_);
    dct_step_.resize(num_component_);
    grid_info_.resize(num_component_);
    yuv_offsets_.resize(num_component_);
    
    if (color_) {
      img_rois_.resize(batch_size_);
      img_steps_.resize(batch_size_);
      img_offsets_.resize(batch_size_);
    }

    // Pre-size our buffers based on the user passed in hint
    size_t pixels_per_image = spec.GetSingleArgument<size_t>("pixels_per_image_hint", 0);
    yuv_data_.Resize({(Index)pixels_per_image * batch_size_});
    if (color_) strided_imgs_.Resize({(Index)pixels_per_image * batch_size_});

    // We need three buffers for our parameters
    param_sizes_.resize(3);
    yuv_data_.template mutable_data<uint8>();
    strided_imgs_.template mutable_data<uint8>();
  }
    
  virtual inline ~DCTQuantInvOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(const vector<Index> &input_shape,
      int data_idx, int /* unused */) override {
    ParsedJpeg &jpeg = jpeg_meta_->template mutable_data<ParsedJpeg>()[data_idx];
    for (int i = 0; i < C_; ++i) {
      // Note: We only iterate over the number of output channels. In the case we
      // are outputting color images, we want to copy all the image meta-data
      // (even the zero-dim components for grayscale images) so that the launch
      // parameters are set up correctly. In the case we are outputing grayscale
      // images, we don't care about the chrominance planes of the input images
      
      // We currently only support 8-bit quant tables
      NDLL_ENFORCE(jpeg.quantTables[i].nPrecision ==
          QuantizationTable::PRECISION_8_BIT,
          "Hybrid decode currently only supports 8-bit quantization tables");
      
      // save the yuv dims and dct steps
      yuv_dims_[data_idx*C_ + i] = jpeg.yCbCrDims[i];
      dct_step_[data_idx*C_ + i] = jpeg.dctLineStep[i];
    }
    
    // Calculate output image meta-data
    if (color_) {
      int tmp;
      getImageSizeStepAndOffset(
          jpeg.imgDims.width,
          jpeg.imgDims.height,
          jpeg.components, jpeg.sRatio,
          &img_rois_[data_idx],
          &img_steps_[data_idx],
          &tmp);
    }
    
    // The output shape is determined by the encoded jpeg, whose meta-data
    // we access through the extra input tensor connected to the huffman decoder
    return vector<Index>{jpeg.imgDims.height, jpeg.imgDims.width, C_};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<int16>(input_type));
    output->template mutable_data<uint8>();
  }
  
  inline string name() const override {
    return "DCTQuantInvOp";
  }
  
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    // Run the batched kernel
    batchedDctQuantInv(
        gpu_param_buffers_[1].template mutable_data<DctQuantInvImageParam>(),
        gpu_param_buffers_[0].template mutable_data<uint8>(),
        gpu_param_buffers_[2].template mutable_data<int>(),
        num_cuda_blocks_
        );

    // DEBUG dump the output
    // for (int i = 0; i < num_component_; ++i) {
    //   DumpHWCToFile(batched_param_buffers_[1].template data<DctQuantInvImageParam>()[i].dst,
    //       yuv_dims_[i].height, yuv_dims_[i].width, 1,
    //       yuv_dims_[i].width, "yuv_img_" + std::to_string(i));
    // }

    if (color_) {
      // Convert the strided and subsampled YUV images to unstrided,
      // RGB or BGR images packed densely into the output batch
      YUVToColorHelper(output);
    } else {
      for (int i = 0; i < batch_size_; ++i) {
        // Note: Need to do a 2D memcpy to handle padding in the width dimension
        const vector<Index> &out_dims = output->sample_shape(i);
        CUDA_CALL(cudaMemcpy2DAsync(
                output->template mutable_sample<uint8>(i), // dst
                out_dims[1], // dptich
                yuv_data_.template data<uint8>() + yuv_offsets_[i], // src
                yuv_dims_[i].width, // spitch
                out_dims[1], // width
                out_dims[0], // height
                cudaMemcpyDeviceToDevice,
                stream_));
      }
    }
  }

  inline void YUVToColorHelper(Batch<Backend> *output) {
    uint8 *yuv_data_ptr = yuv_data_.template mutable_data<uint8>();
    for (int i = 0; i < batch_size_; ++i) {
      ParsedJpeg &jpeg = jpeg_meta_->template mutable_data<ParsedJpeg>()[i];
      if (jpeg.components == 3) {
        uint8 *yuv_planes[3] = {yuv_data_ptr + yuv_offsets_[i*3],
                                yuv_data_ptr + yuv_offsets_[i*3 + 1],
                                yuv_data_ptr + yuv_offsets_[i*3 + 2]};
        int yuv_steps[3] = {yuv_dims_[i*3].width,
                            yuv_dims_[i*3 + 1].width,
                            yuv_dims_[i*3 + 2].width};
        uint8 *strided_img = strided_imgs_.template mutable_data<uint8>() + img_offsets_[i];
        ComponentSampling sampling_ratio = jpeg.sRatio;

        // Note: NPP yuv->rgb + upsample methods are very rigid. With odd dimensioned
        // images, we need to stride the output image. We could add support for strided
        // images in batches, but that would be a pain and force batched ops to support
        // this if they want to be used w/ hybrid jpeg decode. To handle strides,
        // we do the yuv->rgb+upsample into a tmp buffer and then do a small dev2dev
        // 2d memcpy so that the output is dense.
          
        if (output_type_ == NDLL_RGB) {
          // Run the yuv->rgb + upsampling kernel
          yCbCrToRgb((const uint8**)yuv_planes,
              yuv_steps, strided_img, img_steps_[i],
              img_rois_[i], sampling_ratio);
        } else if (output_type_ == NDLL_BGR) {
          // Run the yuv->bgr + upsampling kernel
          yCbCrToBgr((const uint8**)yuv_planes,
              yuv_steps, strided_img, img_steps_[i],
              img_rois_[i], sampling_ratio);
        } else {
          NDLL_FAIL("Unsupported output image type.");
        }

        // Run a 2D memcpy to get rid of image stride
        const vector<Index> &out_dims = output->sample_shape(i);
        CUDA_CALL(cudaMemcpy2DAsync(
                output->template mutable_sample<uint8>(i), // dst
                out_dims[1]*C_, // dpitch
                strided_img, // src
                img_rois_[i].width*C_, // spitch
                out_dims[1]*C_, // width
                out_dims[0], // height
                cudaMemcpyDeviceToDevice,
                stream_
                ));
      } else { // Handle grayscale image
        // For grayscale images, convert to RGB straight into the output batch
        uint8 *y_plane = yuv_data_ptr + yuv_offsets_[i*3];
        int step = yuv_dims_[i*3].width;
        yToRgb(y_plane, step,
            output->template mutable_sample<uint8>(i),
            img_steps_[i], img_rois_[i],
            stream_);
      }
    }
  }
  
  inline void CalculateBatchedParameterSize() override {
    int num_image_planes = 0;
    for (int i = 0; i < batch_size_; ++i) {
      num_image_planes +=
        jpeg_meta_->template mutable_data<ParsedJpeg>()[i].components;
    }

    NDLL_ENFORCE(validateBatchedDctQuantInvParams(dct_step_.data(),
            yuv_dims_.data(), num_component_));
    getBatchedInvDctLaunchParams(yuv_dims_.data(), num_component_,
        &num_cuda_blocks_, grid_info_.data());

    // Setup the sizes for all of our batched parameters
    param_sizes_[0] = 64*num_component_; // quant tables
    param_sizes_[1] = num_component_*sizeof(DctQuantInvImageParam); // dct params
    param_sizes_[2] = num_cuda_blocks_*sizeof(int); // img idxs
    
    // Calculate the size of the YUV intermediate
    // data and resize the intermediate buffer
    size_t yuv_size = 0;
    for (int i = 0; i < num_component_; ++i) {
      yuv_offsets_[i] = yuv_size;
      yuv_size += yuv_dims_[i].height * yuv_dims_[i].width;
    }
    yuv_data_.Resize({(Index)yuv_size});
    
    if (color_) {
      // Calculate the size of the strided image intermediate buffer
      // & resize. We waste a little memory here allocating tmp storage
      // for grayscale images
      size_t strided_imgs_size = 0;
      for (int i = 0; i < batch_size_; ++i) {
        img_offsets_[i] = strided_imgs_size;
        strided_imgs_size += img_rois_[i].width*img_rois_[i].height*C_;
      }
      strided_imgs_.Resize({(Index)strided_imgs_size});
    }
  }

  inline void SerialBatchedParameterSetup(const Batch<Backend>& /* unused */,
      Batch<Backend>* /* unused */) override {
    // Setup image indices for batched idct kernel launch
    getBatchedInvDctImageIndices(yuv_dims_.data(),
        num_component_, param_buffers_[2].template mutable_data<int>());
  }

  inline void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend>* /* unused */, int data_idx, int thread_idx) override {
    // Copy quant tables into mega-buffer
    ParsedJpeg &jpeg = jpeg_meta_->template mutable_data<ParsedJpeg>()[data_idx];
    for (int i = 0; i < C_; ++i) {
      int comp_id = data_idx*C_ + i;
      std::memcpy(param_buffers_[0].template mutable_data<uint8>() + (comp_id * 64),
          jpeg.quantTables[i].aTable.lowp, 64);
    }


    // Setup batched idct parameters
    DctQuantInvImageParam *param = nullptr;
    int dct_offset = 0;
    for (int i = 0; i < C_; ++i) {
      int comp_id = data_idx*C_ + i;
      param = &param_buffers_[1].template mutable_data<DctQuantInvImageParam>()[comp_id];
      param->src = input.template sample<int16>(data_idx) + dct_offset;
      param->srcStep = dct_step_[comp_id];
      param->dst = yuv_data_.template mutable_data<uint8>() + yuv_offsets_[comp_id];
      param->dstWidth = yuv_dims_[comp_id].width;
      param->gridInfo = grid_info_[comp_id];

      // Offset for the next set of DCT coefficients for this image
      dct_offset += jpeg.dctSize[i] / sizeof(int16);
    }
  }

  NDLLImageType output_type_;
  bool color_;
  int C_, num_component_;
  shared_ptr<Tensor<CPUBackend>> jpeg_meta_;

  // image meta-data extracted from parsed jpegs
  vector<NppiSize> yuv_dims_;
  vector<int> dct_step_;
  int num_cuda_blocks_;
  vector<int2> grid_info_;

  // Stored intermediate result of idct+iquant step
  Tensor<Backend> yuv_data_;
  vector<int> yuv_offsets_;

  // Image meta-data used for yuv->rgb conversion
  Tensor<Backend> strided_imgs_;
  vector<NppiSize> img_rois_;
  vector<int> img_steps_;
  vector<int> img_offsets_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_;
  using Operator<Backend>::param_sizes_;
  using Operator<Backend>::param_buffers_;
  using Operator<Backend>::gpu_param_buffers_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_HYBRID_JPG_DECODER_H_
