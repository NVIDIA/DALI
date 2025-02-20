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
#include "dali/c_api_2/error_handling.h"
#include "dali/c_api_2/utils.h"

namespace dali::c_api {

RefCountedPtr<ITensor> ITensor::Create(daliBufferPlacement_t placement) {
  Validate(placement);
  switch (placement.device_type) {
    case DALI_STORAGE_CPU:
    {
      auto tl = std::make_shared<Tensor<CPUBackend>>();
      tl->set_pinned(placement.pinned);
      if (placement.pinned)
        tl->set_device_id(placement.device_id);
      return Wrap(std::move(tl));
    }
    case DALI_STORAGE_GPU:
    {
      auto tl = std::make_shared<Tensor<GPUBackend>>();
      tl->set_pinned(placement.pinned);
      tl->set_device_id(placement.device_id);
      return Wrap(std::move(tl));
    }
    default:
      assert(!"Unreachable code");
      return {};
  }
}

RefCountedPtr<ITensorList> ITensorList::Create(daliBufferPlacement_t placement) {
  Validate(placement);
  switch (placement.device_type) {
    case DALI_STORAGE_CPU:
    {
      auto tl = std::make_shared<TensorList<CPUBackend>>();
      tl->set_pinned(placement.pinned);
      if (placement.pinned)
        tl->set_device_id(placement.device_id);
      return Wrap(std::move(tl));
    }
    case DALI_STORAGE_GPU:
    {
      auto tl = std::make_shared<TensorList<GPUBackend>>();
      tl->set_pinned(placement.pinned);
      tl->set_device_id(placement.device_id);
      return Wrap(std::move(tl));
    }
    default:
      assert(!"Unreachable code");
      return {};
  }
}

ITensor *ToPointer(daliTensor_h handle) {
  if (!handle)
    throw NullHandle("Tensor");
  return static_cast<ITensor *>(handle);
}

ITensorList *ToPointer(daliTensorList_h handle) {
  if (!handle)
    throw NullHandle("TensorList");
  return static_cast<ITensorList *>(handle);
}

}  // namespace dali::c_api

using namespace dali::c_api;  // NOLINT

//////////////////////////////////////////////////////////////////////////////
// Tensor
//////////////////////////////////////////////////////////////////////////////

daliResult_t daliTensorCreate(daliTensor_h *out, daliBufferPlacement_t placement) {
  DALI_PROLOG();
  CHECK_OUTPUT(out);
  auto t = dali::c_api::ITensor::Create(placement);
  *out = t.release();  // no throwing allowed after this line!
  DALI_EPILOG();
}

daliResult_t daliTensorIncRef(daliTensor_h t, int *new_ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(t);
  int r = ptr->IncRef();
  if (new_ref)
    *new_ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorDecRef(daliTensor_h t, int *new_ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(t);
  int r = ptr->DecRef();
  if (new_ref)
    *new_ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorRefCount(daliTensor_h t, int *ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(t);
  int r = ptr->RefCount();
  CHECK_OUTPUT(ref);
  *ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorAttachBuffer(
      daliTensor_h tensor,
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      daliDeleter_t deleter) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  ptr->AttachBuffer(ndim, shape, dtype, layout, data, deleter);
  DALI_EPILOG();
}

daliResult_t daliTensorResize(
      daliTensor_h tensor,
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  ptr->Resize(ndim, shape, dtype, layout);
  DALI_EPILOG();
}

daliResult_t daliTensorSetLayout(
      daliTensor_h tensor,
      const char *layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  ptr->SetLayout(layout);
  DALI_EPILOG();
}

daliResult_t daliTensorGetLayout(
      daliTensor_h tensor,
      const char **layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  CHECK_OUTPUT(layout);
  *layout = ptr->GetLayout();
  DALI_EPILOG();
}

daliResult_t daliTensorGetStream(
      daliTensor_h tensor,
      cudaStream_t *out_stream) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  CHECK_OUTPUT(out_stream);
  auto str = ptr->GetStream();
  *out_stream = str.has_value() ? *str : cudaStream_t(-1);
  return str.has_value() ? DALI_SUCCESS : DALI_NO_DATA;
  DALI_EPILOG();
}

daliResult_t daliTensorSetStream(
      daliTensor_h tensor,
      const cudaStream_t *stream,
      daliBool synchronize) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  ptr->SetStream(ToOptional(stream), synchronize);
  DALI_EPILOG();
}

daliResult_t daliTensorGetDesc(
      daliTensor_h tensor,
      daliTensorDesc_t *out_desc) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  CHECK_OUTPUT(out_desc);
  *out_desc = ptr->GetDesc();
  DALI_EPILOG();
}

daliResult_t daliTensorGetShape(
      daliTensor_h tensor,
      int *out_ndim,
      const int64_t **out_shape) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  auto &shape = ptr->GetShape();
  if (out_ndim)
    *out_ndim = shape.sample_dim();
  if (out_shape)
    *out_shape = shape.data();
  DALI_EPILOG();
}

//////////////////////////////////////////////////////////////////////////////
// TensorList
//////////////////////////////////////////////////////////////////////////////

daliResult_t daliTensorListCreate(daliTensorList_h *out, daliBufferPlacement_t placement) {
  DALI_PROLOG();
  CHECK_OUTPUT(out);
  auto tl = dali::c_api::ITensorList::Create(placement);
  *out = tl.release();  // no throwing allowed after this line!
  DALI_EPILOG();
}

daliResult_t daliTensorListIncRef(daliTensorList_h tl, int *new_ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tl);
  int r = ptr->IncRef();
  if (new_ref)
    *new_ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorListDecRef(daliTensorList_h tl, int *new_ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tl);
  int r = ptr->DecRef();
  if (new_ref)
    *new_ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorListRefCount(daliTensorList_h tl, int *ref) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tl);
  CHECK_OUTPUT(ref);
  int r = ptr->RefCount();
  *ref = r;
  DALI_EPILOG();
}

daliResult_t daliTensorListAttachBuffer(
      daliTensorList_h tensor_list,
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->AttachBuffer(num_samples, ndim, shapes, dtype, layout, data, sample_offsets, deleter);
  DALI_EPILOG();
}

daliResult_t daliTensorListAttachSamples(
      daliTensorList_h tensor_list,
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->AttachSamples(num_samples, ndim, dtype, layout, samples, sample_deleters);
  DALI_EPILOG();
}

daliResult_t daliTensorListResize(
      daliTensorList_h tensor_list,
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->Resize(num_samples, ndim, shapes, dtype, layout);
  DALI_EPILOG();
}

daliResult_t daliTensorListSetLayout(
      daliTensorList_h tensor_list,
      const char *layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->SetLayout(layout);
  DALI_EPILOG();
}

daliResult_t daliTensorListGetLayout(
      daliTensorList_h tensor_list,
      const char **layout) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  CHECK_OUTPUT(layout);
  *layout = ptr->GetLayout();
  DALI_EPILOG();
}

daliResult_t daliTensorListGetStream(
      daliTensorList_h tensor_list,
      cudaStream_t *out_stream) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  CHECK_OUTPUT(out_stream);
  auto str = ptr->GetStream();
  *out_stream = str.has_value() ? *str : cudaStream_t(-1);
  return str.has_value() ? DALI_SUCCESS : DALI_NO_DATA;
  DALI_EPILOG();
}

daliResult_t daliTensorListSetStream(
      daliTensorList_h tensor_list,
      const cudaStream_t *stream,
      daliBool synchronize) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->SetStream(ToOptional(stream), synchronize);
  DALI_EPILOG();
}

daliResult_t daliTensorListGetShape(
      daliTensorList_h tensor,
      int *out_num_samples,
      int *out_ndim,
      const int64_t **out_shape) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor);
  auto &shape = ptr->GetShape();
  if (out_ndim)
    *out_ndim = shape.sample_dim();
  if (out_num_samples)
    *out_num_samples = shape.num_samples();
  if (out_shape)
    *out_shape = shape.shapes.data();
  DALI_EPILOG();
}

daliResult_t daliTensorListGetTensorDesc(
      daliTensorList_h tensor_list,
      daliTensorDesc_t *out_tensor,
      int sample_idx) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  CHECK_OUTPUT(out_tensor);
  *out_tensor = ptr->GetTensorDesc(sample_idx);
  DALI_EPILOG();
}

daliResult_t daliTensorListViewAsTensor(
      daliTensorList_h tensor_list,
      daliTensor_h *out_tensor) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  CHECK_OUTPUT(out_tensor);
  auto t = ptr->ViewAsTensor();
  *out_tensor = t.release();  // no throwing allowed after this line
  DALI_EPILOG();
}
