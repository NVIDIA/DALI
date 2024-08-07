// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <utility>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/dltensor.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

TEST(DLMTensorPtr, CPU) {
  Tensor<CPUBackend> tensor;
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, tensor.device_id());
  ASSERT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCPU);
  ASSERT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, GPU) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({100, 50, 1}, DALI_INT32);
  SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, tensor.device_id());
  ASSERT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[2], 1);
  ASSERT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int) * 8);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
  ASSERT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, CPUList) {
  TensorList<CPUBackend> tlist;
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_FLOAT64);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(double) * 8);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCPU);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  ASSERT_EQ(tlist.tensor_shape(1).size(), 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(double) * 8);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCPU);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, GPUList) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_UINT8);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLUInt);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCUDA);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.device.device_id, tlist.device_id());

  ASSERT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLUInt);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCUDA);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.device.device_id, tlist.device_id());
}

struct TestDLTensorResource: public DLTensorResource {
  TestDLTensorResource(TensorShape<> shape, bool &called)
  : DLTensorResource(std::move(shape))
  , called(called) {
    called = false;
  }

  bool &called;

  ~TestDLTensorResource() override {
    called = true;
  }
};

TEST(DLMTensorPtr, Cleanup) {
  Tensor<CPUBackend> tensor;
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  bool deleter_called = false;
  {
    auto dlm_tensor = MakeDLTensor(tensor.raw_mutable_data(),
                                   tensor.type(),
                                   false, -1,
                                   std::make_unique<TestDLTensorResource>(tensor.shape(),
                                                                          deleter_called));
  }
  ASSERT_TRUE(deleter_called);
}

}  // namespace dali
