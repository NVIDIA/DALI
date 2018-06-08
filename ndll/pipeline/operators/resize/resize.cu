// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize/resize.h"

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
    per_sample_rand_[i] = GetRandomSizes();
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
    const auto &input = ws->Input<GPUBackend>(idx);
    const bool save_attrs = spec_.HasArgument("save_attrs");
    const int outputs_per_idx = save_attrs ? 2 : 1;

    auto output = ws->Output<GPUBackend>(outputs_per_idx * idx);

    ResizeParamDescr resizeDescr(this, resizeParam_.data());
    DataDependentSetupGPU(input, output, batch_size_, false,
                            inputImages(), outputImages(), NULL, &resizeDescr);

    // Run the kernel
    cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    BatchedResize(
        (const uint8**)input_ptrs_.data(),
        batch_size_, C_, sizes(input_t).data(),
        output_ptrs_.data(), sizes(output_t).data(),
        type_);
    nppSetStream(old_stream);

    // Setup and output the resize attributes if necessary
    if (save_attrs) {
      auto *attr_output = ws->Output<CPUBackend>(outputs_per_idx * idx + 1);

      vector<Dims> resize_shape(input.ntensor());

      for (int i = 0; i < input.ntensor(); ++i) {
        resize_shape[i] = Dims{2};
      }

      attr_output->Resize(resize_shape);

      for (int i = 0; i < input.ntensor(); ++i) {
        int *t = attr_output->mutable_tensor<int>(i);
        t[0] = sizes(output_t).data()[i].height;
        t[1] = sizes(output_t).data()[i].width;
      }
    }
}

NDLL_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace ndll
