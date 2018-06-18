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
    uint8 **out_batch, const DALISize *out_sizes, DALIInterpType type) {
  DALI_ASSERT(N > 0);
  DALI_ASSERT(C == 1 || C == 3);
  DALI_ASSERT(in_sizes != nullptr);
  DALI_ASSERT(out_sizes != nullptr);

  NppiInterpolationMode npp_type;
  DALI_FORWARD_ERROR(NPPInterpForDALIInterp(type, &npp_type));

  for (int i = 0; i < N; ++i) {
    DALI_ASSERT(in_batch[i] != nullptr);
    DALI_ASSERT(out_batch[i] != nullptr);

    // Setup region of interests to whole image
    NppiRect in_roi, out_roi;
    in_roi.x = 0; in_roi.y = 0;
    in_roi.width = in_sizes[i].width;
    in_roi.height = in_sizes[i].height;
    out_roi.x = 0; out_roi.y = 0;
    out_roi.width = out_sizes[i].width;
    out_roi.height = out_sizes[i].height;

    if (C == 3) {
      DALI_CHECK_NPP(nppiResize_8u_C3R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    } else {
      DALI_CHECK_NPP(nppiResize_8u_C1R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    }
  }
  return DALISuccess;
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

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
