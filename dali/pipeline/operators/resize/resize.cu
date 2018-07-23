// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/pipeline/operators/resize/resize.h"

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>

#include "dali/util/npp.h"

namespace dali {

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
DALIError_t BatchedResize(const uint8 **in_batch, int N, int C, const DALISize *in_sizes,
                          uint8 **out_batch, const DALISize *out_sizes,
                          const NppiPoint *resize_param, DALIInterpType type) {
  DALI_ASSERT(N > 0);
  DALI_ASSERT(C == 1 || C == 3);
  DALI_ASSERT(in_sizes != nullptr);
  DALI_ASSERT(out_sizes != nullptr);

  NppiInterpolationMode npp_type;
  DALI_FORWARD_ERROR(NPPInterpForDALIInterp(type, &npp_type));

#define USE_CROP  0  // currently we are NOT using crop in Resize op

  typedef NppStatus (*resizeFunction) (
                        const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                        int eInterpolation);
  resizeFunction func = C == 3? nppiResize_8u_C3R : nppiResize_8u_C1R;

  for (int i = 0; i < N; ++i) {
    DALI_ASSERT(in_batch[i] != nullptr);
    DALI_ASSERT(out_batch[i] != nullptr);

    const DALISize &in_size  = in_sizes[i];
    const DALISize &out_size = out_sizes[i];

#if USE_CROP
    // Because currently nppiResize_8u_C3R/nppiResize_8u_C1R
    // do not support cropping, to get the results we need we will define and use SrcROI

    // Sizes of resized image are:
    const NppiPoint *resizeParam = resize_param + 2*i;

    const auto w1 = resizeParam->x;
    const auto h1 = resizeParam->y;

    // Upper left coordinate or the SrcROI:
    const auto cropX = in_size.width  * (resizeParam+1)->x / w1;
    const auto cropY = in_size.height * (resizeParam+1)->y / h1;

    // Width and height of SrcROI:
    const auto widthROI = in_size.width * cropW / w1;
    const auto heightROI = in_size.height * cropH / h1;

    const NppiRect in_roi = {cropX, cropY, widthROI, heightROI};
#else
    const NppiRect in_roi  = {0, 0, in_size.width, in_size.height};
    const NppiRect out_roi = {0, 0, out_size.width, out_size.height};
#endif

    DALI_CHECK_NPP((*func)(in_batch[i], in_size.width*C, in_size, in_roi,
                           out_batch[i], out_size.width*C, out_size, out_roi, npp_type));
  }
  return DALISuccess;
}

}  // namespace

template<>
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    per_sample_meta_[i] = GetTransformMeta(spec_, input_shape, ws, i, ResizeInfoNeeded());
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
    // Before we start working on the next input set, we need
    // to wait until the last one is finished. Otherwise we risk
    // overwriting data used by the kernel called for previous image
    if (idx != 0)
      CUDA_CALL(cudaStreamSynchronize(ws->stream()));
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
        resizeParam_.data(), interp_type_);
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

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
