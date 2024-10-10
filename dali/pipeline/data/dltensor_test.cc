// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(DLPackTest, DLType) {
  DLDataType dl;
  for (DALIDataType dali : {
      DALI_BOOL,
      DALI_FLOAT16, DALI_FLOAT, DALI_FLOAT64,
      DALI_INT8, DALI_UINT8,
      DALI_INT16, DALI_UINT16,
      DALI_INT32, DALI_UINT32,
      DALI_INT64, DALI_UINT64 }) {
    dl = ToDLType(dali);
    TypeInfo info = TypeTable::GetTypeInfo(dali);
    EXPECT_EQ(dl.lanes, 1);
    EXPECT_EQ(dl.bits, info.size() * 8);
    if (info.name().find("uint") == 0) {
      EXPECT_EQ(dl.code, kDLUInt);
    } else if (info.name().find("int") == 0) {
      EXPECT_EQ(dl.code, kDLInt);
    } else if (info.name().find("float") == 0) {
      EXPECT_EQ(dl.code, kDLFloat);
    } else if (info.name().find("bool") == 0) {
      EXPECT_EQ(dl.code, kDLBool);
    }

    EXPECT_EQ(ToDALIType(dl), dali) << "Conversion back to DALI type yielded a different type.";
  }
}

TEST(DLPackTest, DLTypeToString) {
  EXPECT_EQ(to_string(DLDataType{ kDLBool, 8, 1 }), "b8");
  EXPECT_EQ(to_string(DLDataType{ kDLBfloat, 16, 1 }), "bf16");
  EXPECT_EQ(to_string(DLDataType{ kDLFloat, 32, 4 }), "f32x4");
  EXPECT_EQ(to_string(DLDataType{ kDLUInt, 16, 2 }), "u16x2");
  EXPECT_EQ(to_string(DLDataType{ kDLInt, 64, 1 }), "i64");
  EXPECT_EQ(to_string(DLDataType{ 123, 8, 16 }), "<unknown:123>8x16");
}

namespace {

void TestSampleViewCPU(bool pinned) {
  Tensor<CPUBackend> tensor;
  tensor.set_pinned(pinned);
  tensor.set_device_id(0);
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, tensor.is_pinned(), tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, pinned ? kDLCUDAHost : kDLCPU);
  EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

}  // namespace

TEST(DLMTensorPtr, ViewCPU) {
  TestSampleViewCPU(false);
}

TEST(DLMTensorPtr, ViewPinnedCPU) {
  TestSampleViewCPU(true);
}

TEST(DLMTensorPtr, CPUShared) {
  Tensor<CPUBackend> tensor;
  tensor.set_pinned(false);
  tensor.set_device_id(0);
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_EQ(tensor.get_data_ptr().use_count(), 2) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCPU);
    EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
  }
  EXPECT_EQ(tensor.get_data_ptr().use_count(), 1) << "Reference leaked.";
}

TEST(DLMTensorPtr, ViewGPU) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({100, 50, 1}, DALI_INT32);
  SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int) * 8);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, CPUList) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(false);
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_FLOAT64);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  EXPECT_EQ(tlist.tensor_shape(1).size(), 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
}


TEST(DLMTensorPtr, CPUSharedList) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(false);
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_FLOAT64);
  const auto &ptr = unsafe_owner(tlist);
  EXPECT_EQ(ptr.use_count(), 3);
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_EQ(ptr.use_count(), 5);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  EXPECT_EQ(tlist.tensor_shape(1).size(), 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  dlm_tensors.clear();
  EXPECT_EQ(ptr.use_count(), 3);
}

TEST(DLMTensorPtr, GPUList) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_UINT8);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_id, tlist.device_id());

  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_id, tlist.device_id());
}

struct TestDLPayload {
  explicit TestDLPayload(bool &destroyed)
  : destroyed(destroyed) {}

  ~TestDLPayload() {
    destroyed = true;
  }

  bool &destroyed;
};


TEST(DLMTensorPtr, Cleanup) {
  bool deleter_called = false;
  {
    auto rsrc = DLTensorResource<TestDLPayload>::Create(deleter_called);
    auto dlm_tensor = ToDLMTensor(std::move(rsrc));
    EXPECT_EQ(rsrc, nullptr);
  }
  EXPECT_TRUE(deleter_called);
}

}  // namespace dali
