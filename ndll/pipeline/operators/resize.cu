// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>

#include "ndll/util/npp.h"

namespace ndll {

namespace {

/**
 * @brief Resizes an input batch of images.
 *
 * Note: This API is subject to change. It currently launches a kernel
 * for every image in the batch, but if we move to a fully batched kernel
 * we will likely need more meta-data setup beforehand
 *
 * This method currently uses an npp kernel, to set the stream for this
 * kernel, call 'nppSetStream()' prior to calling.
 */
NDLLError_t BatchedResize(const uint8 **in_batch, int N, int C, const NDLLSize *in_sizes,
    uint8 **out_batch, const NDLLSize *out_sizes, NDLLInterpType type) {
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT(C == 1 || C == 3);
  NDLL_ASSERT(in_sizes != nullptr);
  NDLL_ASSERT(out_sizes != nullptr);

  NppiInterpolationMode npp_type;
  NDLL_FORWARD_ERROR(NPPInterpForNDLLInterp(type, &npp_type));

  for (int i = 0; i < N; ++i) {
    NDLL_ASSERT(in_batch[i] != nullptr);
    NDLL_ASSERT(out_batch[i] != nullptr);

    // Setup region of interests to whole image
    NppiRect in_roi, out_roi;
    in_roi.x = 0; in_roi.y = 0;
    in_roi.width = in_sizes[i].width;
    in_roi.height = in_sizes[i].height;
    out_roi.x = 0; out_roi.y = 0;
    out_roi.width = out_sizes[i].width;
    out_roi.height = out_sizes[i].height;

    // TODO(tgale): Can move condition out w/ function ptr or std::function obj
    if (C == 3) {
      NDLL_CHECK_NPP(nppiResize_8u_C3R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    } else {
      NDLL_CHECK_NPP(nppiResize_8u_C1R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    }
  }
  return NDLLSuccess;
}

}  // namespace

template<>
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  for (int i = 0; i < batch_size_; ++i) {
    auto rand_a = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
    auto rand_b = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);

    per_sample_rand_[i] = std::make_pair(rand_a, rand_b);
  }
}

template<>
inline void Resize<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  NDLL_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data stored in uint8.");

  vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    // Verify the inputs
    vector<Index> input_shape = input.tensor_shape(i);
    NDLL_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");
    NDLL_ENFORCE(input_shape[2] == C_,
        "Input channel dimension does not match "
        "the output channel argument.");

    // Select resize dimensions for the output
    NDLLSize &in_size = input_sizes_[i];
    in_size.height = input_shape[0];
    in_size.width = input_shape[1];

    // retrieve the random numbers for this sample
    auto rand_a = per_sample_rand_[i].first;
    auto rand_b = per_sample_rand_[i].second;

    NDLLSize &out_size = output_sizes_[i];
    if (random_resize_ && warp_resize_) {
      // random resize + warp. Select a new size for both dims of
      // the image uniformly from the range [resize_a_, resize_b_]
      out_size.height = rand_a;
      out_size.width = rand_b;
    } else if (random_resize_) {
      // random + no warp. We select a new size of the smallest side
      // of the image uniformly in the range [resize_a_, resize_b_]
      if (in_size.width < in_size.height) {
        out_size.width = rand_a;
        out_size.height =
          static_cast<float>(in_size.height) / in_size.width * out_size.width;
      } else {
        out_size.height = rand_a;
        out_size.width =
          static_cast<float>(in_size.width) / in_size.height * out_size.height;
      }
    } else if (warp_resize_) {
      // no random + warp. We take the new dims to be h = resize_a_
      // and w = resize_b_
      out_size.height = rand_a;
      out_size.width = rand_b;
    } else {
      // no random + no warp. In this mode resize_b_ is ignored and
      // the input image is resizes such that the smallest side is
      // >= resize_a_
      if (in_size.width < in_size.height) {
        out_size.width = rand_a;
        out_size.height =
            static_cast<float>(in_size.height) / in_size.width * out_size.width;
      } else {
        out_size.height = rand_a;
        out_size.width =
            static_cast<float>(in_size.width) / in_size.height * out_size.height;
      }
    }

    // Collect the output shapes
    output_shape[i] = {out_size.height, out_size.width, C_};
  }

  // Resize the output
  output->Resize(output_shape);

  // Collect the pointers for execution
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_[i] = input.template tensor<uint8>(i);
    output_ptrs_[i] = output->template mutable_tensor<uint8>(i);
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  DataDependentSetup(ws, idx);

  // Run the kernel
  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());
  BatchedResize(
      (const uint8**)input_ptrs_.data(),
      batch_size_, C_, input_sizes_.data(),
      output_ptrs_.data(), output_sizes_.data(),
      type_);
  nppSetStream(old_stream);
}

NDLL_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace ndll
