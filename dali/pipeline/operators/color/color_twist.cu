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


#include "dali/util/npp.h"
#include "dali/pipeline/operators/color/color_twist.h"

namespace dali {

typedef NppStatus (*colorTwistFunc)(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aTwist[3][4]);

template <>
void ColorTwistBase<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8_t>(input.type()),
      "Color augmentations accept only uint8 tensors");
  auto &output = ws->Output<GPUBackend>(idx);
  output.ResizeLike(input);

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());

  for (size_t i = 0; i < input.ntensor(); ++i) {
    if (!augments_.empty()) {
      float matrix[nDim][nDim];
      float * m = reinterpret_cast<float*>(matrix);
      IdentityMatrix(m);
      for (size_t j = 0; j < augments_.size(); ++j) {
        augments_[j]->Prepare(i, spec_, ws);
        (*augments_[j])(m);
      }
      NppiSize size;
      size.height = input.tensor_shape(i)[0];
      size.width = input.tensor_shape(i)[1];
      const int nStep = C_ * size.width;  // W * C
      colorTwistFunc twist_func = C_ == 3 ? nppiColorTwist32f_8u_C3R : nppiColorTwist32f_8u_C1R;
      DALI_CHECK_NPP(twist_func(input.tensor<uint8_t>(i),
                                nStep,
                                output.mutable_tensor<uint8_t>(i),
                                nStep,
                                size,
                                matrix));
    } else {
      CUDA_CALL(cudaMemcpyAsync(output.raw_mutable_tensor(i),
                                input.raw_tensor(i),
                                volume(input.tensor_shape(i)),
                                cudaMemcpyDefault,
                                ws->stream()));
    }
  }
  nppSetStream(old_stream);
}

DALI_REGISTER_OPERATOR(Brightness, BrightnessAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Contrast, ContrastAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Hue, HueAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Saturation, SaturationAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistAdjust<GPUBackend>, GPU);

}  // namespace dali
