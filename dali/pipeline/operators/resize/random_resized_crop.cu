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


#include <vector>
#include <cmath>

#include "dali/util/npp.h"
#include "dali/pipeline/operators/resize/random_resized_crop.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

template<>
void RandomResizedCrop<GPUBackend>::RunImpl(DeviceWorkspace * ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  const int newH = size_[0];
  const int newW = size_[1];

  auto &output = ws->Output<GPUBackend>(idx);
  output.set_type(input.type());

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    const int C = input.tensor_shape(i)[2];
    output_shape[i] = {newH, newW, C};
  }
  output.Resize(output_shape);

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());

  /*
   * workaround for out of bound acces from nppiResize
   * add one more line in the last image
   * it should not break anything as only one operator at the time can operate on the GPU
   * and other should not keep any reference to the internals of the input tensor list
  */
  vector<Dims> new_shape(input.shape());
  new_shape[batch_size_ - 1][1] += 1;
  size_t new_size = 0;
  for (auto &s : new_shape) {
    new_size += volume(s);
  }
  new_size *= input.type().size();

  if (new_size > input.capacity()) {
    // reallocate only if current tensor doesn't have enough capcity
    TensorList<GPUBackend> tmp_tensor_list;
    auto &mutable_input = const_cast<TensorList<GPUBackend>&>(input);
    tmp_tensor_list.ShareData(const_cast<TensorList<GPUBackend>*>(&input));
    mutable_input.Resize(new_shape);
    // copy data and shape back to new, bigger storage
    mutable_input.Copy(tmp_tensor_list, ws->stream());
  }

  for (int i = 0; i < batch_size_; ++i) {
    const CropWindow &crop = params_->crops[i];
    NppiRect in_roi, out_roi;
    in_roi.x = crop.x;
    in_roi.y = crop.y;
    in_roi.width = crop.w;
    in_roi.height = crop.h;
    out_roi.x = 0;
    out_roi.y = 0;
    out_roi.width = newW;
    out_roi.height = newH;

    const int H = input.tensor_shape(i)[0];  // HWC
    const int W = input.tensor_shape(i)[1];  // HWC
    const int C = input.tensor_shape(i)[2];  // HWC

    NppiSize input_size, output_size;

    input_size.width = W;
    input_size.height = H;

    output_size.width = newW;
    output_size.height = newH;

    NppiInterpolationMode npp_interp_type;
    DALI_ENFORCE(NPPInterpForDALIInterp(interp_type_, &npp_interp_type) == DALISuccess,
        "Unsupported interpolation type");

    switch (C) {
      case 3:
        DALI_CHECK_NPP(nppiResize_8u_C3R(input.tensor<uint8_t>(i),
                                         W*C,
                                         input_size,
                                         in_roi,
                                         output.mutable_tensor<uint8_t>(i),
                                         newW*C,
                                         output_size,
                                         out_roi,
                                         npp_interp_type));
        break;
      case 1:
        DALI_CHECK_NPP(nppiResize_8u_C1R(input.tensor<uint8_t>(i),
                                         W*C,
                                         input_size,
                                         in_roi,
                                         output.mutable_tensor<uint8_t>(i),
                                         newW*C,
                                         output_size,
                                         out_roi,
                                         npp_interp_type));
        break;
      default:
        DALI_FAIL("RandomResizedCrop is implemented only for images"
            " with C = 1 or 3, but encountered C = " + to_string(C) + ".");
    }
  }
  nppSetStream(old_stream);
}

template<>
void RandomResizedCrop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");
  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];

    params_->crops[i] = params_->crop_gens[i].GenerateCropWindow(H, W);
  }
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<GPUBackend>, GPU);

}  // namespace dali
