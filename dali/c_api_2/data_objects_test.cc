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
#include "dali/core/device_guard.h"

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
  ASSERT_EQ(r, DALI_SUCCESS) << daliGetLastErrorMessage();

  int ref = -1;
  EXPECT_EQ(daliTensorListRefCount(h, &ref), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  EXPECT_EQ(daliTensorListDecRef(h, &ref), DALI_SUCCESS) << daliGetLastErrorMessage();
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
  placement.pinned = true;
  int64_t shapes[] = {
    480, 640, 3,
    600, 800, 4,
    348, 720, 1,
    1080, 1920, 3
  };
  daliDataType_t dtype = DALI_UINT32;

  auto tl = CreateTensorList(placement);

  daliBufferPlacement_t test_placement{};
  EXPECT_EQ(daliTensorListGetBufferPlacement(tl, &test_placement), DALI_SUCCESS);
  EXPECT_EQ(test_placement.device_type, placement.device_type);
  EXPECT_EQ(test_placement.device_id, placement.device_id);
  EXPECT_EQ(test_placement.pinned, placement.pinned);

  EXPECT_EQ(daliTensorListResize(tl, 4, 3, nullptr, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, -1, 3, shapes, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, 4, -1, shapes, dtype, nullptr), DALI_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, "ABCD"), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = -1;
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, "HWC"), DALI_ERROR_INVALID_ARGUMENT);
  shapes[0] = 480;
  EXPECT_EQ(daliTensorListResize(tl, 1, 3, shapes, dtype, "HWC"), DALI_SUCCESS)
      << daliGetLastErrorMessage();
  // resize, but keep the layout
  EXPECT_EQ(daliTensorListResize(tl, 4, 3, shapes, dtype, nullptr), DALI_SUCCESS)
      << daliGetLastErrorMessage();

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  EXPECT_EQ(daliTensorListGetShape(tl, nullptr, nullptr, nullptr), DALI_SUCCESS)
      << daliGetLastErrorMessage();
  {
    int nsamples = -1, ndim = -1;
    const int64_t *shape_data = nullptr;
    EXPECT_EQ(daliTensorListGetShape(tl, &nsamples, &ndim, &shape_data), DALI_SUCCESS)
        << daliGetLastErrorMessage();
    ASSERT_NE(shape_data, nullptr);
    EXPECT_EQ(nsamples, 4);
    EXPECT_EQ(ndim, 3);
    for (int i = 0, k = 0; i < 4; i++)
      for (int d = 0; d < 3; d++, k++) {
        EXPECT_EQ(shapes[k], shape_data[k]) << " @ sample " << i << " dim " << d;
      }
  }

  ptrdiff_t offset = 0;
  const char *base;
  for (int i = 0; i < 4; i++) {
    daliTensorDesc_t desc{};
    EXPECT_EQ(daliTensorListGetTensorDesc(tl, &desc, i), DALI_SUCCESS) << daliGetLastErrorMessage();
    ASSERT_EQ(desc.ndim, 3);
    if (i == 0)
      base = static_cast<char *>(desc.data);
    EXPECT_EQ(desc.data, base + offset);
    EXPECT_EQ(desc.dtype, dtype);
    ASSERT_NE(desc.shape, nullptr);
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

struct TestDeleterCtx {
  void *expected_data;
  int buffer_delete_count;
  int context_delete_count;
};

template <typename element_t>
inline std::pair<daliDeleter_t, std::unique_ptr<TestDeleterCtx>>
MakeTestDeleter(element_t *expected_data) {
  auto ctx = std::unique_ptr<TestDeleterCtx>(new TestDeleterCtx{ expected_data, 0, 0 });
  daliDeleter_t deleter = {};
  deleter.deleter_ctx = ctx.get();
  deleter.delete_buffer = [](void *vctx, void *data, const cudaStream_t *stream) {
    ASSERT_NE(data, nullptr);
    auto *ctx = static_cast<TestDeleterCtx *>(vctx);
    EXPECT_EQ(ctx->context_delete_count, 0);
    EXPECT_EQ(ctx->buffer_delete_count, 0);
    EXPECT_EQ(data, ctx->expected_data);
    ctx->buffer_delete_count++;
    // do not actually delete the data
  };
  deleter.destroy_context = [](void *vctx) {
    auto *ctx = static_cast<TestDeleterCtx *>(vctx);
    EXPECT_EQ(ctx->context_delete_count, 0);
    EXPECT_EQ(ctx->buffer_delete_count, 1);
    ctx->context_delete_count++;
  };
  return { deleter, std::move(ctx) };
}

TEST(CAPI2_TensorListTest, AttachBuffer) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  using element_t = int;
  daliDataType_t dtype = dali::type2id<element_t>::value;
  dali::TensorListShape<> lshape({
    { 480, 640, 3 },
    { 600, 800, 4 },
    { 348, 720, 1 },
    { 1080, 1920, 3 }
  });
  auto size = lshape.num_elements();
  std::unique_ptr<element_t[]> data(new element_t[size]);

  ptrdiff_t offsets[4] = {};
  for (int i = 1; i < 4; i++)
    offsets[i] = offsets[i - 1] + volume(lshape[i - 1]) * sizeof(element_t);

  auto [deleter, ctx] = MakeTestDeleter(data.get());

  auto tl = CreateTensorList(placement);
  ASSERT_EQ(daliTensorListAttachBuffer(
      tl,
      lshape.num_samples(),
      lshape.sample_dim(),
      lshape.data(),
      dtype,
      "HWC",
      data.get(),
      offsets,
      deleter), DALI_SUCCESS) << daliGetLastErrorMessage();

  // The deleter doesn't actually delete - we still own the data.

  ptrdiff_t offset = 0;
  const char *base = reinterpret_cast<const char *>(data.get());
  for (int i = 0; i < 4; i++) {
    daliTensorDesc_t desc{};
    EXPECT_EQ(daliTensorListGetTensorDesc(tl, &desc, i), DALI_SUCCESS) << daliGetLastErrorMessage();
    ASSERT_EQ(desc.ndim, 3);
    EXPECT_EQ(desc.data, base + offset);
    EXPECT_EQ(desc.dtype, dtype);
    ASSERT_NE(desc.shape, nullptr);
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(desc.shape[j], lshape[i][j]);
    EXPECT_STREQ(desc.layout, "HWC");
    size_t sample_bytes = volume(dali::make_cspan(desc.shape, desc.ndim)) * sizeof(element_t);
    offset += sample_bytes;
  }

  tl.reset();

  EXPECT_EQ(ctx->buffer_delete_count, 1) << "Buffer deleter not called";
  EXPECT_EQ(ctx->context_delete_count, 1) << "Deleter context not destroyed";
}


TEST(CAPI2_TensorListTest, AttachSamples) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  using element_t = int;
  daliDataType_t dtype = dali::type2id<element_t>::value;
  dali::TensorListShape<> lshape({
    { 480, 640, 3 },
    { 600, 800, 4 },
    { 348, 720, 1 },
    { 1080, 1920, 3 }
  });
  auto size = lshape.num_elements();
  int N = lshape.num_samples();
  std::vector<std::unique_ptr<element_t[]>> data(N);

  for (int i = 0; i < N; i++) {
    data[i].reset(new element_t[size]);
  }

  std::vector<daliDeleter_t> deleters(N);
  std::vector<std::unique_ptr<TestDeleterCtx>> deleter_ctxs(N);

  for (int i = 0; i < N; i++) {
    std::tie(deleters[i], deleter_ctxs[i]) = MakeTestDeleter(data[i].get());
  }

  std::vector<daliTensorDesc_t> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i].ndim = lshape.sample_dim();
    samples[i].dtype = dtype;
    samples[i].layout = i == 0 ? "HWC" : nullptr;
    samples[i].shape = lshape.tensor_shape_span(i).data();
    samples[i].data = data[i].get();
  }

  auto tl = CreateTensorList(placement);
  ASSERT_EQ(daliTensorListAttachSamples(
      tl,
      lshape.num_samples(),
      -1,
      DALI_NO_TYPE,
      nullptr,
      samples.data(),
      deleters.data()), DALI_SUCCESS) << daliGetLastErrorMessage();

  // The deleter doesn't actually delete - we still own the data.
  for (int i = 0; i < 4; i++) {
    daliTensorDesc_t desc{};
    EXPECT_EQ(daliTensorListGetTensorDesc(tl, &desc, i), DALI_SUCCESS) << daliGetLastErrorMessage();
    ASSERT_EQ(desc.ndim, 3);
    EXPECT_EQ(desc.data, data[i].get());
    EXPECT_EQ(desc.dtype, dtype);
    ASSERT_NE(desc.shape, nullptr);
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(desc.shape[j], lshape[i][j]);
    EXPECT_STREQ(desc.layout, "HWC");
  }

  tl.reset();

  for (auto &ctx : deleter_ctxs) {
    EXPECT_EQ(ctx->buffer_delete_count, 1) << "Buffer deleter not called";
    EXPECT_EQ(ctx->context_delete_count, 1) << "Deleter context not destroyed";
  }
}


TEST(CAPI2_TensorListTest, ViewAsTensor) {
  int num_dev = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_dev));
  // use the last device
  dali::DeviceGuard dg(num_dev - 1);

  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  placement.pinned = true;
  using element_t = int;
  daliDataType_t dtype = dali::type2id<element_t>::value;
  dali::TensorListShape<> lshape = dali::uniform_list_shape(4, { 480, 640, 3 });
  auto size = lshape.num_elements();
  std::unique_ptr<element_t[]> data(new element_t[size]);

  ptrdiff_t sample_size = volume(lshape[0]) * sizeof(element_t);

  ptrdiff_t offsets[4] = {
    0,
    1 * sample_size,
    2 * sample_size,
    3 * sample_size,
  };

  auto [deleter, ctx] = MakeTestDeleter(data.get());

  auto tl = CreateTensorList(placement);
  ASSERT_EQ(daliTensorListAttachBuffer(
      tl,
      lshape.num_samples(),
      lshape.sample_dim(),
      lshape.data(),
      dtype,
      "HWC",
      data.get(),
      offsets,
      deleter), DALI_SUCCESS) << daliGetLastErrorMessage();

  // The deleter doesn't actually delete - we still own the data.

  daliTensor_h ht = nullptr;
  EXPECT_EQ(daliTensorListViewAsTensor(tl, &ht), DALI_SUCCESS) << daliGetLastErrorMessage();
  ASSERT_NE(ht, nullptr);
  dali::c_api::TensorHandle t(ht);

  daliBufferPlacement_t tensor_placement{};
  EXPECT_EQ(daliTensorGetBufferPlacement(ht, &tensor_placement), DALI_SUCCESS);
  EXPECT_EQ(tensor_placement.device_type, placement.device_type);
  EXPECT_EQ(tensor_placement.device_id, placement.device_id);
  EXPECT_EQ(tensor_placement.pinned, placement.pinned);

  daliTensorDesc_t desc{};
  EXPECT_EQ(daliTensorGetDesc(t, &desc), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(desc.data, data.get());
  ASSERT_NE(desc.shape, nullptr);
  EXPECT_EQ(desc.shape[0], lshape.num_samples());
  ASSERT_EQ(desc.ndim, 4);
  EXPECT_EQ(desc.shape[1], lshape[0][0]);
  EXPECT_EQ(desc.shape[2], lshape[0][1]);
  EXPECT_EQ(desc.shape[3], lshape[0][2]);
  EXPECT_STREQ(desc.layout, "NHWC");
  EXPECT_EQ(desc.dtype, dtype);
  EXPECT_EQ(daliTensorGetShape(t, nullptr, nullptr), DALI_SUCCESS) << daliGetLastErrorMessage();
  int ndim = -1;
  const int64_t *shape = nullptr;
  EXPECT_EQ(daliTensorGetShape(t, &ndim, &shape), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(ndim, 4);
  EXPECT_EQ(shape, desc.shape);

  tl.reset();

  EXPECT_EQ(ctx->buffer_delete_count, 0) << "Buffer prematurely destroyed";
  EXPECT_EQ(ctx->context_delete_count, 0) << "Deleter context prematurely destroyed";

  t.reset();

  EXPECT_EQ(ctx->buffer_delete_count, 1) << "Buffer deleter not called";
  EXPECT_EQ(ctx->context_delete_count, 1) << "Deleter context not destroyed";
}


TEST(CAPI2_TensorListTest, ViewAsTensorError) {
  daliBufferPlacement_t placement{};
  placement.device_type = DALI_STORAGE_CPU;
  using element_t = int;
  daliDataType_t dtype = dali::type2id<element_t>::value;
  dali::TensorListShape<> lshape = dali::uniform_list_shape(4, { 480, 640, 3 });
  auto size = lshape.num_elements();
  std::unique_ptr<element_t[]> data(new element_t[size]);

  ptrdiff_t sample_size = volume(lshape[0]) * sizeof(element_t);

  // The samples are not in order
  ptrdiff_t offsets[4] = {
    0,
    2 * sample_size,
    1 * sample_size,
    3 * sample_size,
  };

  auto [deleter, ctx] = MakeTestDeleter(data.get());

  auto tl = CreateTensorList(placement);
  ASSERT_EQ(daliTensorListAttachBuffer(
      tl,
      lshape.num_samples(),
      lshape.sample_dim(),
      lshape.data(),
      dtype,
      "HWC",
      data.get(),
      offsets,
      deleter), DALI_SUCCESS) << daliGetLastErrorMessage();

  // The deleter doesn't actually delete - we still own the data.

  daliTensor_h ht = nullptr;
  EXPECT_EQ(daliTensorListViewAsTensor(tl, &ht), DALI_ERROR_INVALID_OPERATION);
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
  ASSERT_EQ(r, DALI_SUCCESS) << daliGetLastErrorMessage();

  int ref = -1;
  EXPECT_EQ(daliTensorRefCount(h, &ref), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  EXPECT_EQ(daliTensorDecRef(h, &ref), DALI_SUCCESS) << daliGetLastErrorMessage();
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
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, "HWC"), DALI_SUCCESS)
      << daliGetLastErrorMessage();

  shape[0] = 1080;
  EXPECT_EQ(daliTensorResize(t, 3, shape, dtype, nullptr), DALI_SUCCESS)
      << daliGetLastErrorMessage();

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  ptrdiff_t offset = 0;
  daliTensorDesc_t desc{};
  EXPECT_EQ(daliTensorGetDesc(t, &desc), DALI_SUCCESS) << daliGetLastErrorMessage();
  ASSERT_EQ(desc.ndim, 3);
  EXPECT_STREQ(desc.layout, "HWC");
  EXPECT_EQ(desc.dtype, dtype);
  ASSERT_NE(desc.shape, nullptr);
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

TEST(CAPI2_TensorTest, SourceInfo) {
  auto t = CreateTensor({});
  const char *out_src_info = "junk";
  EXPECT_EQ(daliTensorGetSourceInfo(t, &out_src_info), DALI_SUCCESS);
  EXPECT_EQ(out_src_info, nullptr);

  EXPECT_EQ(daliTensorSetSourceInfo(t, "source_info"), DALI_SUCCESS);
  EXPECT_EQ(daliTensorGetSourceInfo(t, &out_src_info), DALI_SUCCESS);
  EXPECT_STREQ(out_src_info, "source_info");
}

TEST(CAPI2_TensorListTest, SourceInfo) {
  auto t = CreateTensorList({});
  ASSERT_EQ(daliTensorListResize(t, 5, 0, nullptr, DALI_UINT8, nullptr), DALI_SUCCESS);

  const char *out_src_info = "junk";
  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 0), DALI_SUCCESS);
  EXPECT_EQ(out_src_info, nullptr);

  EXPECT_EQ(daliTensorListSetSourceInfo(t, 0, "quick"), DALI_SUCCESS);
  EXPECT_EQ(daliTensorListSetSourceInfo(t, 2, "brown"), DALI_SUCCESS);
  EXPECT_EQ(daliTensorListSetSourceInfo(t, 4, "fox"), DALI_SUCCESS);

  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 0), DALI_SUCCESS);
  EXPECT_STREQ(out_src_info, "quick");
  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 1), DALI_SUCCESS);
  EXPECT_EQ(out_src_info, nullptr);
  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 2), DALI_SUCCESS);
  EXPECT_STREQ(out_src_info, "brown");
  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 3), DALI_SUCCESS);
  EXPECT_EQ(out_src_info, nullptr);
  EXPECT_EQ(daliTensorListGetSourceInfo(t, &out_src_info, 4), DALI_SUCCESS);
  EXPECT_STREQ(out_src_info, "fox");
}
