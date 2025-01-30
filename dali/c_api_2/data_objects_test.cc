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
#include "dali/core/span.h"

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
  daliTensorList_h handle;
  auto err = daliTensorListCreate(&handle, placement);
  switch (err) {
  case DALI_SUCCESS:
    break;
  case DALI_ERROR_OUT_OF_MEMORY:
    throw std::bad_alloc();
  case DALI_ERROR_INVALID_ARGUMENT:
    throw std::invalid_argument(daliGetLastErrorMessage());
  default:
    throw std::runtime_error(daliGetLastErrorMessage());
  }
  return dali::c_api::TensorListHandle(handle);
}

void TestTensorListResize(daliStorageDevice_t storage_device) {
  daliBufferPlacement_t placement{};
  placement.device_type = storage_device;
  auto tl = CreateTensorList(placement);
  int64_t shapes[] = {
    480, 640, 3,
    600, 800, 4,
    348, 720, 1,
    1080, 1920, 3
  };
  daliDataType_t dtype = DALI_UINT32;

  EXPECT_EQ(daliTensorListResize(tl, 4, 3, nullptr, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, -1, 3, shapes, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, 4, -1, shapes, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, "ABCD"), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = -1;
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, "HWC"), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = 480;
  EXPECT_EQ(daliTensorListResize(tl, 1, 3, shapes, dtype, "HWC"), DALI_SUCCESS);
  // resize, but keep the layout
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, nullptr), DALI_SUCCESS);

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  ptrdiff_t offset = 0;
  const char *base;
  for (int i = 0; i < 4; i++) {
    daliTensorDesc_t desc{};
    EXPECT_EQ(daliTensorListGetTensorDesc(tl, &desc, i), DALI_SUCCESS);
    ASSERT_EQ(desc.ndim, 3);
    ASSERT_NE(desc.data, nullptr);
    if (i == 0)
      base = static_cast<char *>(desc.data);
    EXPECT_EQ(desc.data, base + offset);
    EXPECT_EQ(desc.dtype, dtype);
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(desc.shape[j], shapes[3 * i + j]);
    size_t sample_bytes = volume(dali::make_cspan(desc.shape, desc.ndim)) * element_size;
    if (storage_device == DALI_STORAGE_GPU) {
      // Check that the data is accessible for the GPU
      EXPECT_EQ(cudaMemset(desc.data, 0, sample_bytes), cudaSuccess);
    } else {
      // Check that the data is accessible for the CPU
      memset(desc.data, 0, sample_bytes);  // just not crashing is OK
    }
    offset += sample_bytes;
  }
  if (storage_device == DALI_STORAGE_GPU) {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
}


TEST(CAPI2_TensorListTest, AttachBuffer) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  auto tl = CreateTensorList(placement);
  using element_t = int;
  daliDataType_t dtype = dali::type2id<element_t>::value;
  dali::TensorListShape<> lshape({
    { 480, 640, 3 },
    { 600, 800, 4 },
    { 348, 720, 1 },
    { 1080, 1920, 3 }
  });
  auto size = lshape.num_elements();
  std::unique_ptr<element_t> data(new element_t[size]);

  ptrdiff_t offsets[4] = {};
  for (int i = 1; i < 4; i++)
    offsets[i] = offsets[i - 1] + volume(lshape[i - 1]) * sizeof(element_t);

  struct DeleterCtx {
    void *expected_data;
    int buffer_delete_count;
    int context_delete_count;
  };
  DeleterCtx ctx = { data.get(), 0, 0 };
  daliDeleter_t deleter = {};
  deleter.deleter_ctx = &ctx;
  deleter.delete_buffer = [](void *vctx, void *data, const cudaStream_t *stream) {
    ASSERT_NE(data, nullptr);
    auto *ctx = static_cast<DeleterCtx *>(vctx);
    EXPECT_EQ(ctx->context_delete_count, 0);
    EXPECT_EQ(ctx->buffer_delete_count, 0);
    EXPECT_EQ(data, ctx->expected_data);
    ctx->buffer_delete_count++;
    delete [] static_cast<element_t *>(data);
  };
  deleter.destroy_context = [](void *vctx) {
    auto *ctx = static_cast<DeleterCtx *>(vctx);
    EXPECT_EQ(ctx->context_delete_count, 0);
    EXPECT_EQ(ctx->buffer_delete_count, 1);
    ctx->context_delete_count++;
  };

  ASSERT_EQ(daliTensorListAttachBuffer(
      tl,
      lshape.num_samples(),
      lshape.sample_dim(),
      lshape.data(),
      dtype,
      "HWC",
      data.get(),
      offsets,
      deleter), DALI_SUCCESS);

  void *data_ptr = data.release();  // the buffer is now owned by the tensor list

  ptrdiff_t offset = 0;
  const char *base = static_cast<const char *>(data_ptr);
  for (int i = 0; i < 4; i++) {
    daliTensorDesc_t desc{};
    EXPECT_EQ(daliTensorListGetTensorDesc(tl, &desc, i), DALI_SUCCESS);
    ASSERT_EQ(desc.ndim, 3);
    ASSERT_NE(desc.data, nullptr);
    EXPECT_EQ(desc.data, base + offset);
    EXPECT_EQ(desc.dtype, dtype);
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(desc.shape[j], lshape[i][j]);
    size_t sample_bytes = volume(dali::make_cspan(desc.shape, desc.ndim)) * sizeof(element_t);
    offset += sample_bytes;
  }

  tl.reset();

  EXPECT_EQ(ctx.buffer_delete_count, 1) << "Buffer deleter not called";
  EXPECT_EQ(ctx.context_delete_count, 1) << "Deleter context not destroyed";
}

TEST(CAPI2_TensorListTest, ResizeCPU) {
  TestTensorListResize(DALI_STORAGE_CPU);
}

TEST(CAPI2_TensorListTest, ResizeGPU) {
  TestTensorListResize(DALI_STORAGE_GPU);
}




TEST(CAPI2_TensorTest, NullHandle) {
  daliTensor_h h = nullptr;
  int ref = 0;
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorIncRef(h, &ref));
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorDecRef(h, &ref));
  EXPECT_EQ(DALI_ERROR_INVALID_HANDLE, daliTensorRefCount(h, &ref));
}

TEST(CAPI2_TensorTest, CreateDestroy) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  placement.pinned = false;
  daliTensor_h h = nullptr;
  daliResult_t r = daliTensorCreate(&h, placement);
  ASSERT_NE(h, nullptr);
  dali::c_api::TensorHandle tl(h);
  ASSERT_EQ(h, tl.get());
  ASSERT_EQ(r, DALI_SUCCESS);

  int ref = -1;
  EXPECT_EQ(daliTensorRefCount(h, &ref), DALI_SUCCESS);
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  EXPECT_EQ(daliTensorDecRef(h, &ref), DALI_SUCCESS);
  EXPECT_EQ(ref, 0);
}


inline auto CreateTensor(daliBufferPlacement_t placement) {
  daliTensor_h handle;
  auto err = daliTensorCreate(&handle, placement);
  switch (err) {
  case DALI_SUCCESS:
    break;
  case DALI_ERROR_OUT_OF_MEMORY:
    throw std::bad_alloc();
  case DALI_ERROR_INVALID_ARGUMENT:
    throw std::invalid_argument(daliGetLastErrorMessage());
  default:
    throw std::runtime_error(daliGetLastErrorMessage());
  }
  return dali::c_api::TensorHandle(handle);
}

void TestTensorResize(daliStorageDevice_t storage_device) {
  daliBufferPlacement_t placement{};
  placement.device_type = storage_device;
  auto t = CreateTensor(placement);
  int64_t shape[] = {
    1080, 1920, 3
  };
  daliDataType_t dtype = DALI_INT16;

  EXPECT_EQ(daliTensorResize(t, 3, nullptr, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorResize(t, -1, shape, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, "ABCD"), DALI_ERROR_INVALID_ARGUMENT);
  shape[0] = -1;
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, "HWC"), DALI_ERROR_INVALID_ARGUMENT);
  shape[0] = 1;
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, "HWC"), DALI_SUCCESS);

  shape[0] = 1080;
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, nullptr), DALI_SUCCESS);

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  ptrdiff_t offset = 0;
  daliTensorDesc_t desc{};
  EXPECT_EQ(daliTensorGetDesc(t, &desc), DALI_SUCCESS);
  ASSERT_EQ(desc.ndim, 3);
  ASSERT_NE(desc.data, nullptr);
  EXPECT_STREQ(desc.layout, "HWC");
  EXPECT_EQ(desc.dtype, dtype);
  for (int j = 0; j < 3; j++)
    EXPECT_EQ(desc.shape[j], shape[j]);
  size_t sample_bytes = volume(dali::make_cspan(desc.shape, desc.ndim)) * element_size;
  if (storage_device == DALI_STORAGE_GPU) {
    // Check that the data is accessible for the GPU
    EXPECT_EQ(cudaMemset(desc.data, 0, sample_bytes), cudaSuccess);
  } else {
    // Check that the data is accessible for the CPU
    memset(desc.data, 0, sample_bytes);  // just not crashing is OK
  }
  if (storage_device == DALI_STORAGE_GPU) {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
}

TEST(CAPI2_TensorTest, ResizeCPU) {
  TestTensorResize(DALI_STORAGE_CPU);
}

TEST(CAPI2_TensorTest, ResizeGPU) {
  TestTensorResize(DALI_STORAGE_GPU);
}
