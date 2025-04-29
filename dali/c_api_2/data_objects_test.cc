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
#include <tuple>
#include "dali/dali_cpp_wrappers.h"
#include "dali/core/span.h"
#include "dali/core/device_guard.h"
#include "dali/c_api_2/test_utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/core/mm/memory.h"
#include "dali/core/cuda_stream_pool.h"

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
  CHECK_DALI(daliTensorListRefCount(h, &ref));
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  CHECK_DALI(daliTensorListDecRef(h, &ref));
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
  CHECK_DALI(daliTensorListGetBufferPlacement(tl, &test_placement));
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
  CHECK_DALI(daliTensorListResize(tl, 1, 3, shapes, dtype, "HWC"));
  // resize, but keep the layout
  CHECK_DALI(daliTensorListResize(tl, 4, 3, shapes, dtype, nullptr));

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  CHECK_DALI(daliTensorListGetShape(tl, nullptr, nullptr, nullptr));

  daliDataType_t reported_dtype = DALI_NO_TYPE;
  CHECK_DALI(daliTensorListGetDType(tl, &reported_dtype));
  EXPECT_EQ(reported_dtype, dtype);

  {
    int nsamples = -1, ndim = -1;
    const int64_t *shape_data = nullptr;
    ASSERT_EQ(daliTensorListGetShape(tl, &nsamples, &ndim, &shape_data), DALI_SUCCESS)
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
    CHECK_DALI(daliTensorListGetTensorDesc(tl, &desc, i));
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
  size_t reported_byte_size = 0;
  CHECK_DALI(daliTensorListGetByteSize(tl, &reported_byte_size));
  EXPECT_EQ(reported_byte_size, static_cast<size_t>(offset));

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
    CHECK_DALI(daliTensorListGetTensorDesc(tl, &desc, i));
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
    CHECK_DALI(daliTensorListGetTensorDesc(tl, &desc, i));
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
  CHECK_DALI(daliTensorListViewAsTensor(tl, &ht));
  ASSERT_NE(ht, nullptr);
  dali::c_api::TensorHandle t(ht);

  daliBufferPlacement_t tensor_placement{};
  CHECK_DALI(daliTensorGetBufferPlacement(ht, &tensor_placement));
  EXPECT_EQ(tensor_placement.device_type, placement.device_type);
  EXPECT_EQ(tensor_placement.device_id, placement.device_id);
  EXPECT_EQ(tensor_placement.pinned, placement.pinned);

  daliTensorDesc_t desc{};
  CHECK_DALI(daliTensorGetDesc(t, &desc));
  EXPECT_EQ(desc.data, data.get());
  ASSERT_NE(desc.shape, nullptr);
  EXPECT_EQ(desc.shape[0], lshape.num_samples());
  ASSERT_EQ(desc.ndim, 4);
  EXPECT_EQ(desc.shape[1], lshape[0][0]);
  EXPECT_EQ(desc.shape[2], lshape[0][1]);
  EXPECT_EQ(desc.shape[3], lshape[0][2]);
  EXPECT_STREQ(desc.layout, "NHWC");
  EXPECT_EQ(desc.dtype, dtype);
  CHECK_DALI(daliTensorGetShape(t, nullptr, nullptr));
  int ndim = -1;
  const int64_t *shape = nullptr;
  CHECK_DALI(daliTensorGetShape(t, &ndim, &shape));
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
  CHECK_DALI(daliTensorRefCount(h, &ref));
  EXPECT_EQ(ref, 1);
  ref = -1;

  h = tl.release();
  CHECK_DALI(daliTensorDecRef(h, &ref));
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
  CHECK_DALI(daliTensorResize(t, 3, shape, dtype, "HWC"));

  shape[0] = 1080;
  CHECK_DALI(daliTensorResize(t, 3, shape, dtype, nullptr));

  size_t element_size = dali::TypeTable::GetTypeInfo(dtype).size();

  ptrdiff_t offset = 0;
  daliTensorDesc_t desc{};
  CHECK_DALI(daliTensorGetDesc(t, &desc));
  ASSERT_EQ(desc.ndim, 3);
  EXPECT_STREQ(desc.layout, "HWC");
  EXPECT_EQ(desc.dtype, dtype);
  ASSERT_NE(desc.shape, nullptr);
  for (int j = 0; j < 3; j++)
    EXPECT_EQ(desc.shape[j], shape[j]);
  size_t sample_bytes = volume(dali::make_cspan(desc.shape, desc.ndim)) * element_size;
  size_t reported_byte_size = 0;
  CHECK_DALI(daliTensorGetByteSize(t, &reported_byte_size));
  EXPECT_EQ(reported_byte_size, sample_bytes);
  daliDataType_t reported_dtype = DALI_NO_TYPE;
  CHECK_DALI(daliTensorGetDType(t, &reported_dtype));
  EXPECT_EQ(reported_dtype, dtype);

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
  CHECK_DALI(daliTensorGetSourceInfo(t, &out_src_info));
  EXPECT_EQ(out_src_info, nullptr);

  CHECK_DALI(daliTensorSetSourceInfo(t, "source_info"));
  CHECK_DALI(daliTensorGetSourceInfo(t, &out_src_info));
  EXPECT_STREQ(out_src_info, "source_info");
}

TEST(CAPI2_TensorListTest, SourceInfo) {
  auto t = CreateTensorList({});
  ASSERT_EQ(daliTensorListResize(t, 5, 0, nullptr, DALI_UINT8, nullptr), DALI_SUCCESS)
    << daliGetLastErrorMessage();

  const char *out_src_info = "junk";
  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 0));
  EXPECT_EQ(out_src_info, nullptr);

  CHECK_DALI(daliTensorListSetSourceInfo(t, 0, "quick"));
  CHECK_DALI(daliTensorListSetSourceInfo(t, 2, "brown"));
  CHECK_DALI(daliTensorListSetSourceInfo(t, 4, "fox"));

  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 0));
  EXPECT_STREQ(out_src_info, "quick");
  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 1));
  EXPECT_EQ(out_src_info, nullptr);
  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 2));
  EXPECT_STREQ(out_src_info, "brown");
  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 3));
  EXPECT_EQ(out_src_info, nullptr);
  CHECK_DALI(daliTensorListGetSourceInfo(t, &out_src_info, 4));
  EXPECT_STREQ(out_src_info, "fox");
}

template <typename T>
void FillTensorList(dali::TensorList<dali::CPUBackend> &tl, T start) {
  auto view = dali::view<T>(tl);
  T value = start;
  for (int i = 0; i < view.num_samples(); i++) {
    auto tv = view[i];
    for (int64_t j = 0, n = tv.num_elements(); j < n; j++)
      tv.data[j] = value++;
  }
}

template <typename T>
void FillTensorList(dali::TensorList<dali::GPUBackend> &tl, T start) {
  dali::TensorList<dali::CPUBackend> cpu;
  cpu.Resize(tl.shape(), tl.type());
  FillTensorList<T>(cpu, start);
  tl.Copy(cpu);
  CUDA_CALL(cudaDeviceSynchronize());
}

template <typename T>
void FillTensorList(
      daliTensorList_h tl,
      const dali::TensorListShape<> &shape,
      T start) {
  CHECK_DALI(daliTensorListResize(
    tl,
    shape.num_samples(),
    shape.sample_dim(),
    shape.shapes.data(),
    dali::type2id<T>::value,
    nullptr));
  daliBufferPlacement_t placement;
  CHECK_DALI(daliTensorListGetBufferPlacement(tl, &placement));
  auto *tl_ptr = static_cast<dali::c_api::ITensorList*>(tl);
  if (placement.device_type == DALI_STORAGE_GPU)
    FillTensorList<T>(*tl_ptr->Unwrap<dali::GPUBackend>(), start);
  else
    FillTensorList<T>(*tl_ptr->Unwrap<dali::CPUBackend>(), start);
}


using CopyTestParams = std::tuple<
daliStorageDevice_t,    // data object device
daliStorageDevice_t,    // target storage device
daliCopyFlags_t>;       // flags

class CAPI2_CopyOutTest : public ::testing::TestWithParam<CopyTestParams> {
};

INSTANTIATE_TEST_SUITE_P(
      CAPI2_CopyOutTest,
      CAPI2_CopyOutTest,
      testing::ValuesIn(std::vector<CopyTestParams>({
  { DALI_STORAGE_CPU, DALI_STORAGE_CPU, DALI_COPY_DEFAULT },

  { DALI_STORAGE_GPU, DALI_STORAGE_GPU, DALI_COPY_DEFAULT },
  { DALI_STORAGE_GPU, DALI_STORAGE_GPU, DALI_COPY_USE_KERNEL },
  { DALI_STORAGE_GPU, DALI_STORAGE_GPU, DALI_COPY_SYNC },

  { DALI_STORAGE_CPU, DALI_STORAGE_GPU, DALI_COPY_DEFAULT },
  { DALI_STORAGE_CPU, DALI_STORAGE_GPU, DALI_COPY_USE_KERNEL },
  { DALI_STORAGE_CPU, DALI_STORAGE_GPU, DALI_COPY_SYNC },

  { DALI_STORAGE_GPU, DALI_STORAGE_CPU, DALI_COPY_DEFAULT },
  { DALI_STORAGE_GPU, DALI_STORAGE_CPU, DALI_COPY_USE_KERNEL },
  { DALI_STORAGE_GPU, DALI_STORAGE_CPU, DALI_COPY_SYNC },
})));

TEST_P(CAPI2_CopyOutTest, CopyOut) {
  CopyTestParams param = this->GetParam();
  auto tl_backend = std::get<0>(param);
  auto copy_backend = std::get<1>(GetParam());
  auto flags = std::get<2>(GetParam());
  daliBufferPlacement_t placement{};
  placement.device_id = 0;
  placement.device_type = tl_backend;
  placement.pinned = true;
  auto tl = CreateTensorList(placement);
  auto lease = dali::CUDAStreamPool::instance().Get();
  dali::TensorListShape<> shape = {
    { 480, 640, 3 },
    { 1080, 1920, 3, },
    { 600, 800, 3 },
  };
  using T = int;
  FillTensorList<T>(tl, shape, 123);
  dali::mm::uptr<T> dst_cpu, dst;
  dst_cpu = dali::mm::alloc_raw_unique<T, dali::mm::memory_kind::host>(shape.num_elements());
  if (copy_backend == DALI_STORAGE_GPU) {
    dst = dali::mm::alloc_raw_unique<T, dali::mm::memory_kind::device>(shape.num_elements());
  } else {
    dst = dali::mm::alloc_raw_unique<T, dali::mm::memory_kind::host>(shape.num_elements());
  }

  daliBufferPlacement_t dst_placement{};
  dst_placement.pinned = true;
  dst_placement.device_type = copy_backend;
  cudaStream_t stream = lease;
  bool h2h = tl_backend == DALI_STORAGE_CPU && copy_backend == DALI_STORAGE_CPU;
  size_t size = shape.num_elements() * sizeof(T);
  CHECK_DALI(daliTensorListCopyOut(tl, dst.get(), dst_placement, h2h ? nullptr : &stream, flags));
  if (copy_backend == DALI_STORAGE_GPU) {
    if (flags & DALI_COPY_SYNC)
      stream = 0;  // deliberately use a different stream, as no sync should be necessary
    CUDA_CALL(cudaMemcpyAsync(dst_cpu.get(), dst.get(), size, cudaMemcpyDefault, stream));
  } else {
    if (!(flags & DALI_COPY_SYNC))
      CUDA_CALL(cudaStreamSynchronize(stream));  // synchronize manually
    // else - no synchronization should be required
    memcpy(dst_cpu.get(), dst.get(), size);
  }
  for (int64_t i = 0, n = shape.num_elements(); i < n; i++)
    ASSERT_EQ(dst_cpu.get()[i], i + 123) << "at i = " << i;
}
