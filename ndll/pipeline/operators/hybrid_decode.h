// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

NDLL_REGISTER_TYPE(ParsedJpeg, NDLL_INTERNAL_PARSEDJPEG);
NDLL_REGISTER_TYPE(DctQuantInvImageParam, NDLL_INTERNAL_DCTQUANTINV_IMAGE_PARAM);

/**
 * @brief Perform the host-side part of a hybrid cpu+gpu jpeg decode. This
 * decoder must be followed by the DCTQuantInvOp to complete the jpeg decode
 */
class HuffmanDecoder : public Operator<CPUBackend> {
 public:
  explicit inline HuffmanDecoder(const OpSpec &spec) :
    Operator<CPUBackend>(spec),
    initial_dct_coeff_size_byte_(spec.GetArgument<int>("dct_bytes_hint")) {
    // Resize per-image & per-thread data
    tl_parser_state_.resize(num_threads_);
    tl_huffman_state_.resize(num_threads_);
  }

  virtual inline ~HuffmanDecoder() = default;

  DISABLE_COPY_MOVE_ASSIGN(HuffmanDecoder);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    int output_offset = idx * 2;
    auto dct_coeff = ws->Output<CPUBackend>(output_offset);
    auto jpeg_meta = ws->Output<CPUBackend>(output_offset + 1);

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
    size_t total_size = 0;
    vector<int> offsets(jpeg->components, 0);
    for (int i = 0; i < jpeg->components; ++i) {
      offsets[i] = total_size;
      total_size += jpeg->dctSize[i] / sizeof(int16);
    }

    // Resize output dct coeffs and setup for the huffman
    // Take a minimum size (empirical) and make allocate that amount,
    // before resizing (non-destructive) to the correct size.
    // Limits # of very expensive pinned alloc / free pairs
    size_t min_coeff_size = initial_dct_coeff_size_byte_ / sizeof(int16);
    // Force large allocation
    dct_coeff->Resize({static_cast<Index>(total_size < min_coeff_size ? min_coeff_size :
                                                                        total_size)});
    dct_coeff->template mutable_data<int16>();
    // Correct sizing
    dct_coeff->Resize({static_cast<Index>(total_size)});

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

 private:
  const int initial_dct_coeff_size_byte_;

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
    output_type_(spec.GetArgument<NDLLImageType>("output_type")),
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
    size_t pixels_per_image = spec.GetArgument<size_t>("bytes_per_sample_hint");
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

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void DataDependentKernelSetup(Workspace<Backend> *ws, const int idx);

  void YUVToColorHelper(Workspace<Backend> *ws);

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
