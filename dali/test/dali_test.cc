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

#include "dali/test/dali_test_single_op.h"
#include <gtest/gtest.h>
#include <vector>


#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"

int main(int argc, char **argv) {
  dali::DALIInit(dali::OpSpec("CPUAllocator"),
      dali::OpSpec("PinnedCPUAllocator"),
      dali::OpSpec("GPUAllocator"));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace dali {

int CheckBuffers(const TensorList<CPUBackend> *t1, const TensorList<CPUBackend> *t2,
                 int imgIdx, uint32_t testCheckType, int C, double eps, double *pMean,
                 const vector<Index> *shape, DALITensorLayout layout,
                 bool checkBestMatch) {
  const auto idx = layout == DALI_NHWC? 0 : 1;
  const auto H = shape? (*shape)[idx] : 0;
  const auto W = shape? (*shape)[idx + 1] : 0;
  const auto len = shape? H * W * C : t1->size();

  const auto imgType = t2->type();
  if (imgType.id() == DALI_FLOAT16) {
    // For now the images in DALI_FLOAT16 should be converted into double
    vector<double> img1, img2;
    if (imgIdx >= 0) {
      ConvertRaster((*t1).template tensor<float16>(imgIdx), H, W, C, &img1);
      ConvertRaster((*t2).template tensor<float16>(imgIdx), H, W, C, &img2);
    } else {
      ConvertRaster(t1->data<float16>(), H, W, C, &img1);
      ConvertRaster(t2->data<float16>(), H, W, C, &img2);
    }

    return CheckBuffers<double>(len, img1.data(), img2.data(),
                testCheckType, C, eps, pMean, shape, layout, checkBestMatch);
  }


  if (imgIdx >= 0) {
    DALI_IMAGE_TYPE_SWITCH_NO_FLOAT16(imgType.id(), imgType,
        return CheckBuffers(len,
                    (*t1).template tensor<imgType>(imgIdx),
                    (*t2).template tensor<imgType>(imgIdx),
                     testCheckType, C, eps, pMean, shape, layout, checkBestMatch);
    )
  }

  // Analyze the whole buffer
  DALI_IMAGE_TYPE_SWITCH_NO_FLOAT16(imgType.id(), Dtype,
      return CheckBuffers < Dtype > (len,
                   t1->data<Dtype>(),
                   t2->data<Dtype>(),
                   testCheckType, C, eps, pMean);
  )

  return 0;
}

}  // namespace dali
