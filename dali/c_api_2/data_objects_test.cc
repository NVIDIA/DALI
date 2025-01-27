// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/c_api_2/data_objects.h"
#include <gtest/gtest.h>
#include "dali/c_api_2/managed_handle.h"

TEST(CAPI2_TensorListTest, NullHandle) {
  daliTensorList_h h = nullptr;
  int ref = 0;
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorListIncRef(h, &ref));
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorListDecRef(h, &ref));
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorListRefCount(h, &ref));
}

TEST(CAPI2_TensorListTest, CreateDestroy) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  placement.pinned = false;
  daliTensorList_h h = nullptr;
  daliResult_t r = daliTensorListCreate(&h, placement);
  ASSERT_NE(h, nullptr);
  dali::c_api::TensorListHandle tl(h);
  ASSERT_EQ(h, tl.get());
  ASSERT_EQ(r, DALI_SUCCESS);

  int ref = -1;
  EXPECT_EQ(daliTensorListRefCount(h, &ref), DALI_SUCCESS);
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  EXPECT_EQ(daliTensorListDecRef(h, &ref), DALI_SUCCESS);
  EXPECT_EQ(ref, 0);
}

inline auto CreateTensorList(daliBufferPlacement_t placement) {
  auto tl = dali::c_api::TensorListInterface::Create(placement);
  return dali::c_api::TensorListHandle(tl.release());
}

TEST(CAPI2_TensorListTest, Resize) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_GPU;
  auto tl = CreateTensorList(placement);
  int64_t shapes[] = {
    480, 640, 3,
    600, 800, 3,
    348, 720, 1,  //
  };
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, DALI_UINT32, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, -1, 3, DALI_UINT32, shapes), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, 4, -1, DALI_UINT32, shapes), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = -1;
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, DALI_UINT32, shapes), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = 480;
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, DALI_UINT32, shapes), DALI_SUCCESS);

  
}
