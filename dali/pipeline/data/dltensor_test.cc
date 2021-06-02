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

#include <gtest/gtest.h>
#include <utility>
#include "dali/pipeline/data/dltensor.h"

namespace dali {

TEST(DLMTensorPtr, CPU) {
  Tensor<CPUBackend> tensor;
  tensor.set_type(TypeInfo::Create<float>());
  tensor.Resize({100, 50, 3});
  DLMTensorPtr dlm_tensor = GetDLTensorView(tensor);
  ASSERT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCPU);
  ASSERT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, GPU) {
  Tensor<GPUBackend> tensor;
  tensor.set_type(TypeInfo::Create<int>());
  tensor.Resize({100, 50, 1});
  DLMTensorPtr dlm_tensor = GetDLTensorView(tensor);
  ASSERT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensor->dl_tensor.shape[2], 1);
  ASSERT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
  ASSERT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int) * 8);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  ASSERT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
  ASSERT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, CPUList) {
  TensorList<CPUBackend> tlist;
  tlist.set_type(TypeInfo::Create<double>());
  tlist.Resize({{100, 50, 1}, {50, 30, 3}});
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
  tlist.set_type(TypeInfo::Create<uint8>());
  tlist.Resize({{100, 50, 1}, {50, 30, 3}});
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLUInt);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(uint8) * 8);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCUDA);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);
  ASSERT_EQ(dlm_tensors[0]->dl_tensor.device.device_id, tlist.device_id());

  ASSERT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLUInt);
  ASSERT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(uint8) * 8);
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
  tensor.set_type(TypeInfo::Create<float>());
  tensor.Resize({100, 50, 3});
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
