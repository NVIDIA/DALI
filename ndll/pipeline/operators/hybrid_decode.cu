// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/hybrid_decode.h"

namespace ndll {

template<>
void DCTQuantInv<GPUBackend>::YUVToColorHelper(DeviceWorkspace *ws) {
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

template<>
void DCTQuantInv<GPUBackend>::DataDependentKernelSetup(DeviceWorkspace *ws, const int idx) {
  int input_offset = idx * 2;
  auto &dct_coeff = ws->Input<GPUBackend>(input_offset);
  auto &jpeg_meta = ws->Input<CPUBackend>(input_offset + 1);
  auto output = ws->Output<GPUBackend>(idx);
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

template<>
void DCTQuantInv<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());
  DataDependentKernelSetup(ws, idx);

  batchedDctQuantInv(
      dct_params_gpu_.template mutable_data<DctQuantInvImageParam>(),
      quant_tables_gpu_.template mutable_data<uint8>(),
      block_image_indices_gpu_.template mutable_data<int>(),
      num_cuda_blocks_);

  auto output = ws->Output<GPUBackend>(idx);
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

NDLL_REGISTER_OPERATOR(DCTQuantInv, DCTQuantInv<GPUBackend>, GPU);

}  // namespace ndll
