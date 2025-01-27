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

namespace dali::c_api {

RefCountedPtr<TensorListInterface> TensorListInterface::Create(daliBufferPlacement_t placement) {
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
      throw std::invalid_argument(make_string("Invalid storage device: ", placement.device_type));
  }
}

TensorListInterface *ToPointer(daliTensorList_h handle) {
  if (!handle)
    throw NullHandle("TensorList");
  return static_cast<TensorListInterface *>(handle);
}

}  // namespace dali::c_api

using namespace dali::c_api;  // NOLINT

daliResult_t daliTensorListCreate(daliTensorList_h *out, daliBufferPlacement_t placement) {
  DALI_PROLOG();
  auto tl = dali::c_api::TensorListInterface::Create(placement);
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
  if (!ref)
    throw std::invalid_argument("The output pointer must not be NULL.");
  int r = ptr->RefCount();
  *ref = r;
  DALI_EPILOG();
}

DALI_API daliResult_t daliTensorListAttachBuffer(
      daliTensorList_h tensor_list,
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const int64_t *shapes,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->AttachBuffer(num_samples, ndim, dtype, layout, shapes, data, sample_offsets, deleter);
  DALI_EPILOG();
}

DALI_API daliResult_t daliTensorListAttachSamples(
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
      daliDataType_t dtype,
      const int64_t *shapes) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  ptr->Resize(num_samples, ndim, dtype, shapes);
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
  if (!layout)
    throw std::invalid_argument("The output parameter `layout` must not be be NULL");
  *layout = ptr->GetLayout();
  DALI_EPILOG();
}

daliResult_t daliTensorListGetStream(
      daliTensorList_h tensor_list,
      cudaStream_t *out_stream) {
  DALI_PROLOG();
  auto *ptr = ToPointer(tensor_list);
  if (!out_stream)
    throw std::invalid_argument("The output parameter `out_stream` must not be NULL");
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
  std::optional<cudaStream_t> opt_str;
  if (stream)
    opt_str = *stream;
  else
    opt_str = std::nullopt;
  ptr->SetStream(opt_str, synchronize);
  DALI_EPILOG();
}
