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

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>

#include "dali/util/npp.h"
#include "dali/pipeline/operators/resize/resize.h"
#include "dali/kernels/static_switch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/pipeline/data/views.h"

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

    DALI_CHECK_NPP((*func)(
      in_batch[i], in_size.width*C, ToNppiSize(in_size), in_roi,
      out_batch[i], out_size.width*C, ToNppiSize(out_size), out_roi, npp_type));
  }
  return DALISuccess;
}

}  // namespace

template<>
Resize<GPUBackend>::Resize(const OpSpec &spec) : Operator<GPUBackend>(spec), ResizeAttr(spec) {
  save_attrs_ = spec_.HasArgument("save_attrs");
  outputs_per_idx_ = save_attrs_ ? 2 : 1;

  resizeParam_ = new  vector<NppiPoint>(batch_size_ * 2);
  // Resize per-image data
  input_ptrs_.resize(batch_size_);
  output_ptrs_.resize(batch_size_);
  sizes_[0].resize(batch_size_);
  sizes_[1].resize(batch_size_);

  // Per set-of-sample TransformMeta
  per_sample_meta_.resize(batch_size_);
}

template<>
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  resample_params_.resize(batch_size_);

  auto interp_min = spec_.GetArgument<DALIInterpType>("min_filter");
  auto interp_mag = spec_.GetArgument<DALIInterpType>("mag_filter");
  auto interp_default = spec_.GetArgument<DALIInterpType>("interp_type");
  if (interp_default != DALI_INTERP_LINEAR && interp_mag == DALI_INTERP_LINEAR)
    interp_mag = interp_default;  // old syntax used

  kernels::ResamplingFilterType interp2resample[] = {
    kernels::ResamplingFilterType::Nearest,
    kernels::ResamplingFilterType::Linear,
    kernels::ResamplingFilterType::Linear,  // cubic - not supported
    kernels::ResamplingFilterType::Lanczos3,
    kernels::ResamplingFilterType::Triangular,
    kernels::ResamplingFilterType::Gaussian
  };

  kernels::FilterDesc min_filter = { interp2resample[interp_min], 0 };
  kernels::FilterDesc mag_filter = { interp2resample[interp_mag], 0 };

  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    per_sample_meta_[i] = GetTransformMeta(spec_, input_shape, ws, i, ResizeInfoNeeded());
    resample_params_[i][0].output_size = per_sample_meta_[i].rsz_h;
    resample_params_[i][1].output_size = per_sample_meta_[i].rsz_w;
    resample_params_[i][0].min_filter = resample_params_[i][1].min_filter = min_filter;
    resample_params_[i][0].mag_filter = resample_params_[i][1].mag_filter = mag_filter;
  }

  context_.gpu.stream = ws->stream();
  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;
  requirements_ = Kernel::GetRequirements(context_, view<const uint8_t, 3>(input), resample_params_);
  scratch_alloc_.Reserve(requirements_.scratch_sizes);
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
    context_.gpu.stream = ws->stream();
    const auto &input = ws->Input<GPUBackend>(idx);

    auto &output = ws->Output<GPUBackend>(outputs_per_idx_ * idx);

    ResizeParamDescr resizeDescr(this, resizeParam_->data());
    DataDependentSetupGPU(input, output, batch_size_, false,
                            inputImages(), outputImages(), NULL, &resizeDescr);

    auto scratchpad = scratch_alloc_.GetScratchpad();
    context_.scratchpad = &scratchpad;
    using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;

    cudaStreamSynchronize(ws->stream());
    //auto start = std::chrono::high_resolution_clock::now();
    Kernel::Run(context_, view<uint8_t, 3>(output), view<const uint8_t, 3>(input), resample_params_);
    cudaStreamSynchronize(ws->stream());
    //auto end = std::chrono::high_resolution_clock::now();
    //auto resample_time = end-start;
    //cout << "Resample: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(resample_time).count() << "ms\n";

    // Run the kernel
    /*cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    start = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(ws->stream());
    int C_ = IsColor(spec_.GetArgument<DALIImageType>("image_type")) ? 3 : 1;
    BatchedResize(
        (const uint8**)input_ptrs_.data(),
        batch_size_, C_, sizes(input_t).data(),
        output_ptrs_.data(), sizes(output_t).data(),
        resizeParam_->data(), getInterpType());
    nppSetStream(old_stream);
    cudaStreamSynchronize(ws->stream());
    end = std::chrono::high_resolution_clock::now();
    auto npp_time = end-start;
    cout << "NPP: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(npp_time).count() << "ms\n";*/

    // Setup and output the resize attributes if necessary
    if (save_attrs_) {
      TensorList<CPUBackend> attr_output_cpu;
      vector<Dims> resize_shape(input.ntensor());

      for (size_t i = 0; i < input.ntensor(); ++i) {
        resize_shape[i] = Dims{2};
      }

      attr_output_cpu.Resize(resize_shape);

      for (size_t i = 0; i < input.ntensor(); ++i) {
        int *t = attr_output_cpu.mutable_tensor<int>(i);
        t[0] = sizes(input_t).data()[i].height;
        t[1] = sizes(input_t).data()[i].width;
      }
      ws->Output<GPUBackend>(outputs_per_idx_ * idx + 1).Copy(attr_output_cpu, ws->stream());
    }
}

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
