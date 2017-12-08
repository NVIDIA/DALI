// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_HYBRID_DECODE_H_
#define NDLL_PIPELINE_OPERATORS_HYBRID_DECODE_H_

// TODO(tgale): Fix this include setup so that we can use this
// from external code
#include <third_party/hybrid_decode/include/hybrid_decoder.h>

#include <cstring>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"
#include "ndll/util/image.h"

namespace ndll {

NDLL_REGISTER_TYPE(ParsedJpeg);
NDLL_REGISTER_TYPE(DctQuantInvImageParam);

/**
 * @brief Perform the host-side part of a hybrid cpu+gpu jpeg decode. This
 * decoder must be followed by the DCTQuantInvOp to complete the jpeg decode
 */
template <typename Backend>
class HuffmanDecoder : public Operator<Backend> {
 public:
  explicit inline HuffmanDecoder(const OpSpec &spec) :
    Operator<Backend>(spec) {
    // Resize per-image & per-thread data
    tl_parser_state_.resize(num_threads_);
    tl_huffman_state_.resize(num_threads_);
  }

  virtual inline ~HuffmanDecoder() = default;

  inline int MaxNumInput() const override { return 1; }
  inline int MinNumInput() const override { return 1; }
  inline int MaxNumOutput() const override { return 2; }
  inline int MinNumOutput() const override { return 2; }

  DISABLE_COPY_MOVE_ASSIGN(HuffmanDecoder);

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto dct_coeff = ws->Output<CPUBackend>(0);
    auto jpeg_meta = ws->Output<CPUBackend>(1);

    // Validate input
    NDLL_ENFORCE(input.ndim() == 1,
        "Input must be 1D encoded jpeg string.");
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Input must be stored as uint8 data.");

    // Resize the output and parse the input data
    jpeg_meta->Resize({1});
    ParsedJpeg *jpeg = jpeg_meta->template mutable_data<ParsedJpeg>();
    parseRawJpegHost(
        input.template data<uint8>(),
        input.size(),
        &tl_parser_state_[ws->thread_idx()],
        jpeg);

    // Note: The DCT coefficients for each image component
    // can be different sizes due to subsampling. Single
    // Tensors cannot be jagged, so we pack all the DCT
    // coefficients and pass the needed meta-data to the
    // DCTQuantInv op through the jpeg_meta output
    int total_size = 0;
    vector<int> offsets(jpeg->components, 0);
    for (int i = 0; i < jpeg->components; ++i) {
      offsets[i] = total_size;
      total_size += jpeg->dctSize[i] / sizeof(int16);
    }

    // Resize output dct coeffs and setup for the huffman
    dct_coeff->Resize({total_size});
    vector<int16*> dct_ptrs(jpeg->components);
    HuffmanDecoderState &state =
      tl_huffman_state_[ws->thread_idx()];

    for (int i = 0; i < jpeg->components; ++i) {
      dct_ptrs[i] = dct_coeff->template mutable_data<int16>() +
        offsets[i];
    }

    // Run the huffman decode
    huffmanDecodeHost(*jpeg, &state, &dct_ptrs);
  }

  vector<JpegParserState> tl_parser_state_;
  vector<HuffmanDecoderState> tl_huffman_state_;

  USE_OPERATOR_MEMBERS();
};

// Note: Our handling of grayscale images is a bit nuanced. In
// the case that we are outputting color (rgb) images, we always
// rely on the paramters setup calls from hybrid_decoder (i.e.
// validateBatchedDctQuantInvParams, getBatchedInvDctLaunchParams,
// getBatchedInvDctImageIndices) to handle grayscale images that
// may be mixed into our input batch. Even for grayscale images
// (images with a single component), we create parameters for all
// three components. The component indices that are set up by the
// hybrid_decoder calls will take into account that these components
// are non-existent and create no indices for these components.
// This is a small waste of memory with our extra parameters, but
// greatly simplifies the code so it is ok for now.
//
// When we are outputting grayscale images, we only bother to do
// the idcts for the chrominance planes of the jpegs. We do this
// into our intermediate buffer because the output of the idct
// is still aligned to jpeg block stuff, and then we just perform
// dev-to-dev copies to get the final grayscale images into the
// output buffer
template <typename Backend>
class DCTQuantInv : public Operator<Backend> {
 public:
  explicit inline DCTQuantInv(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<NDLLImageType>("output_type", NDLL_RGB)),
    color_(IsColor(output_type_)), C_(color_ ? 3 : 1) {
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
    size_t pixels_per_image = spec.GetArgument<size_t>("bytes_per_sample_hint", 0);
    yuv_data_.Resize({(Index)pixels_per_image * batch_size_});
    if (color_) strided_imgs_.Resize({(Index)pixels_per_image * batch_size_});

    // We need three buffers for our parameters
    yuv_data_.template mutable_data<uint8>();
    strided_imgs_.template mutable_data<uint8>();

    // Resize our quant tables and dct params
    quant_tables_.Resize({64*num_component_});
    dct_params_.Resize({num_component_});
  }

  virtual inline ~DCTQuantInv() = default;

  inline int MaxNumInput() const override { return 2; }
  inline int MinNumInput() const override { return 2; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }

 protected:
  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    DataDependentKernelSetup(ws);

    batchedDctQuantInv(
        dct_params_gpu_.template mutable_data<DctQuantInvImageParam>(),
        quant_tables_gpu_.template mutable_data<uint8>(),
        block_image_indices_gpu_.template mutable_data<int>(),
        num_cuda_blocks_);

    auto output = ws->Output<GPUBackend>(0);
    if (color_) {
      // Convert the strided and subsampled YUV images to unstrided,
      // RGB or BGR images packed densely into the output batch
      YUVToColorHelper(ws);
    } else {
      for (int i = 0; i < batch_size_; ++i) {
        // Note: Need to do a 2D memcpy to handle padding in the width dimension
        const vector<Index> &out_dims = output->tensor_shape(i);
        CUDA_CALL(cudaMemcpy2DAsync(
                output->template mutable_tensor<uint8>(i),  // dst
                out_dims[1],  // dptich
                yuv_data_.template data<uint8>() + yuv_offsets_[i],  // src
                yuv_dims_[i].width,  // spitch
                out_dims[1],  // width
                out_dims[0],  // height
                cudaMemcpyDeviceToDevice,
                ws->stream()));
      }
    }
    // Set the stream back
    nppSetStream(old_stream);
  }

  inline void DataDependentKernelSetup(DeviceWorkspace *ws) {
    auto &dct_coeff = ws->Input<GPUBackend>(0);
    auto &jpeg_meta = ws->Input<CPUBackend>(1);
    auto output = ws->Output<GPUBackend>(0);
    NDLL_ENFORCE(dct_coeff.ntensor() == batch_size_,
        "Expected dct coefficients for each image.");
    NDLL_ENFORCE(IsType<int16>(dct_coeff.type()),
        "Expected dct data in int16");
    NDLL_ENFORCE(jpeg_meta.ntensor() == batch_size_,
        "Expected jpeg meta-data for each image.");
    NDLL_ENFORCE(IsType<ParsedJpeg>(jpeg_meta.type()),
        "Expected ParsedJpeg structs for jpeg meta.");

    vector<Dims> output_shape(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
      const ParsedJpeg &jpeg = jpeg_meta.template data<ParsedJpeg>()[i];
      for (int j = 0; j < C_; ++j) {
        int comp_id = i*C_ + j;
        // Gather meta-data from the inputs
        NDLL_ENFORCE(jpeg.quantTables[j].nPrecision ==
            QuantizationTable::PRECISION_8_BIT, "Hybrid "
            "decoder only supports 8-bit quant tables.");
        yuv_dims_[comp_id] = jpeg.yCbCrDims[j];
        dct_step_[comp_id] = jpeg.dctLineStep[j];

        // Pack the quant tables into our param buffer
        std::memcpy(quant_tables_.template mutable_data<uint8>() + comp_id*64,
            jpeg.quantTables[j].aTable.lowp, 64);
      }

      if (color_) {
        // Calculate output image meta-data
        int tmp;
        getImageSizeStepAndOffset(
            jpeg.imgDims.width,
            jpeg.imgDims.height,
            jpeg.components, jpeg.sRatio,
            &img_rois_[i],
            &img_steps_[i],
            &tmp);
      }
      // Set the output shape for this image
      output_shape[i] =
        {jpeg.imgDims.height, jpeg.imgDims.width, C_};
    }

    // Move the quantization tables to device
    quant_tables_gpu_.Copy(quant_tables_, ws->stream());

    // Resize the output
    output->Resize(output_shape);

    // Validate our parameters and setup launch meta-data
    NDLL_ENFORCE(validateBatchedDctQuantInvParams(dct_step_.data(),
            yuv_dims_.data(), num_component_));
    getBatchedInvDctLaunchParams(yuv_dims_.data(), num_component_,
        &num_cuda_blocks_, grid_info_.data());

    // Setup image indices for each CTA and move to device
    block_image_indices_.Resize({num_cuda_blocks_});
    getBatchedInvDctImageIndices(yuv_dims_.data(), num_component_,
        block_image_indices_.template mutable_data<int>());
    block_image_indices_gpu_.Copy(block_image_indices_, ws->stream());

    // Resize the intermediate buffer for image yuv data
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

    // Setup idct parameters for each image
    DctQuantInvImageParam *param = nullptr;
    for (int i = 0; i < batch_size_; ++i) {
      int dct_offset = 0;
      const ParsedJpeg &jpeg = jpeg_meta.template data<ParsedJpeg>()[i];
      for (int j = 0; j < C_; ++j) {
        int comp_id = i*C_ + j;
        param = &dct_params_.template mutable_data<DctQuantInvImageParam>()[comp_id];
        param->src = dct_coeff.template tensor<int16>(i) + dct_offset;
        param->srcStep = dct_step_[comp_id];
        param->dst = yuv_data_.template mutable_data<uint8>() + yuv_offsets_[comp_id];
        param->dstWidth = yuv_dims_[comp_id].width;
        param->gridInfo = grid_info_[comp_id];

        // Offset for the next set of DCT coefficients for this image
        dct_offset += jpeg.dctSize[j] / sizeof(int16);
      }
    }

    // Move the dct parameters to device
    dct_params_gpu_.Copy(dct_params_, ws->stream());
  }

  inline void YUVToColorHelper(DeviceWorkspace *ws) {
    auto output = ws->Output<GPUBackend>(0);
    auto &jpeg_meta = ws->Input<CPUBackend>(1);

    uint8 *yuv_data_ptr = yuv_data_.template mutable_data<uint8>();
    for (int i = 0; i < batch_size_; ++i) {
      const ParsedJpeg &jpeg = jpeg_meta.template data<ParsedJpeg>()[i];
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
        const vector<Index> &out_dims = output->tensor_shape(i);
        CUDA_CALL(cudaMemcpy2DAsync(
                output->template mutable_tensor<uint8>(i),  // dst
                out_dims[1]*C_,  // dpitch
                strided_img,  // src
                img_rois_[i].width*C_,  // spitch
                out_dims[1]*C_,  // width
                out_dims[0],  // height
                cudaMemcpyDeviceToDevice,
                ws->stream()));
      } else {  // Handle grayscale image
        // For grayscale images, convert to RGB straight into the output batch
        uint8 *y_plane = yuv_data_ptr + yuv_offsets_[i*3];
        int step = yuv_dims_[i*3].width;
        yToRgb(y_plane, step,
            output->template mutable_tensor<uint8>(i),
            img_steps_[i], img_rois_[i],
            ws->stream());
      }
    }
  }

  NDLLImageType output_type_;
  bool color_;
  int C_, num_component_;

  Tensor<CPUBackend> quant_tables_, dct_params_, block_image_indices_;
  Tensor<GPUBackend> quant_tables_gpu_, dct_params_gpu_, block_image_indices_gpu_;

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

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_HYBRID_DECODE_H_
