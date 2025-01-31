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

#ifndef DALI_C_API_2_DATA_OBJECTS_H_
#define DALI_C_API_2_DATA_OBJECTS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/c_api_2/ref_counting.h"
#include "dali/c_api_2/validation.h"


struct _DALITensorList {};
struct _DALITensor {};

namespace dali {
namespace c_api {

//////////////////////////////////////////////////////////////////////////////
// Interfaces
//////////////////////////////////////////////////////////////////////////////

class ITensor : public _DALITensor, public RefCountedObject {
 public:
  virtual ~ITensor() = default;

  virtual void Resize(
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout) = 0;

  virtual void AttachBuffer(
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      daliDeleter_t deleter) = 0;

  virtual daliBufferPlacement_t GetBufferPlacement() const = 0;

  virtual const char *GetLayout() const = 0;

  virtual void SetLayout(const char *layout) = 0;

  virtual void SetStream(std::optional<cudaStream_t> stream, bool synchronize) = 0;

  virtual std::optional<cudaStream_t> GetStream() const = 0;

  virtual std::optional<cudaEvent_t> GetReadyEvent() const = 0;

  virtual cudaEvent_t GetOrCreateReadyEvent() = 0;

  virtual daliTensorDesc_t GetDesc() const = 0;

  static RefCountedPtr<ITensor> Create(daliBufferPlacement_t placement);
};


class ITensorList : public _DALITensorList, public RefCountedObject {
 public:
  virtual ~ITensorList() = default;

  virtual void Resize(
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout) = 0;

  virtual void AttachBuffer(
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) = 0;

  virtual void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) = 0;

  virtual daliBufferPlacement_t GetBufferPlacement() const = 0;

  virtual const char *GetLayout() const = 0;

  virtual void SetLayout(const char *layout) = 0;

  virtual void SetStream(std::optional<cudaStream_t> stream, bool synchronize) = 0;

  virtual std::optional<cudaStream_t> GetStream() const = 0;

  virtual std::optional<cudaEvent_t> GetReadyEvent() const = 0;

  virtual cudaEvent_t GetOrCreateReadyEvent() = 0;

  virtual daliTensorDesc_t GetTensorDesc(int sample) const = 0;

  virtual RefCountedPtr<ITensor> ViewAsTensor() const = 0;

  static RefCountedPtr<ITensorList> Create(daliBufferPlacement_t placement);
};


//////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////


struct BufferDeleter {
  daliDeleter_t deleter;
  AccessOrder deletion_order;

  void operator()(void *data) {
    if (deleter.delete_buffer) {
      cudaStream_t stream = deletion_order.stream();
      deleter.delete_buffer(deleter.deleter_ctx, data,
                            deletion_order.is_device() ? &stream : nullptr);
    }
    if (deleter.destroy_context) {
      deleter.destroy_context(deleter.deleter_ctx);
    }
  }
};

template <typename Backend>
class TensorWrapper : public ITensor {
 public:
  explicit TensorWrapper(std::shared_ptr<Tensor<Backend>> t) : t_(std::move(t)) {}

  void Resize(
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout) override {
    ValidateShape(ndim, shape);
    Validate(dtype);
    if (layout)
      Validate(TensorLayout(layout), ndim);
    t_->Resize(TensorShape<>(make_cspan(shape, ndim)), dtype);
    if (layout)
      t_->SetLayout(layout);
  }

  void AttachBuffer(
      int ndim,
      const int64_t *shape,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      daliDeleter_t deleter) override {
    ValidateShape(ndim, shape);
    Validate(dtype);
    if (layout)
      Validate(TensorLayout(layout), ndim);

    TensorShape<> tshape(make_cspan(shape, ndim));
    size_t num_elements = volume(tshape);
    if (num_elements > 0 && !data)
      throw std::invalid_argument("The data buffer must not be NULL for a non-empty tensor.");

    TensorLayout new_layout = {};

    if (!layout) {
      if (ndim == t_->ndim())
        new_layout = t_->GetLayout();
    } else {
      new_layout = layout;
      Validate(new_layout, ndim);
    }

    t_->Reset();
    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();

    std::shared_ptr<void> buffer;
    if (!deleter.delete_buffer && !deleter.destroy_context) {
      buffer = std::shared_ptr<void>(data, [](void *){});
    } else {
      buffer = std::shared_ptr<void>(data, BufferDeleter{deleter, t_->order()});
    }

    t_->ShareData(
      std::move(buffer),
      num_elements * element_size,
      t_->is_pinned(),
      tshape,
      dtype,
      t_->device_id(),
      t_->order());

    if (layout)
      t_->SetLayout(new_layout);
  }

  daliBufferPlacement_t GetBufferPlacement() const override {
    daliBufferPlacement_t placement;
    placement.device_id = t_->device_id();
    StorageDevice dev = backend_to_storage_device<Backend>::value;
    placement.device_type = static_cast<daliStorageDevice_t>(dev);
    placement.pinned = t_->is_pinned();
    return placement;
  }

  void SetStream(std::optional<cudaStream_t> stream, bool synchronize) override {
    t_->set_order(stream.has_value() ? AccessOrder(*stream) : AccessOrder::host(), synchronize);
  }

  void SetLayout(const char *layout_string) {
    if (layout_string) {
      TensorLayout layout(layout_string);
      Validate(layout, t_->ndim());
      t_->SetLayout(layout);
    } else {
      t_->SetLayout("");
    }
  }

  const char *GetLayout() const override {
    auto &layout = t_->GetLayout();
    return !layout.empty() ? layout.data() : nullptr;
  }

  std::optional<cudaStream_t> GetStream() const override {
    auto o = t_->order();
    if (o.is_device())
      return o.stream();
    else
      return std::nullopt;
  }

  std::optional<cudaEvent_t> GetReadyEvent() const override {
    auto &e = t_->ready_event();
    if (e)
      return e.get();
    else
      return std::nullopt;
  }

  cudaEvent_t GetOrCreateReadyEvent() override {
    auto &e = t_->ready_event();
    if (e)
      return e.get();
    int device_id = t_->device_id();
    if (device_id < 0)
      throw std::runtime_error("The tensor list is not associated with a CUDA device.");
    t_->set_ready_event(CUDASharedEvent::Create(device_id));
    return t_->ready_event().get();
  }

  daliTensorDesc_t GetDesc() const override {
    auto &shape = t_->shape();
    daliTensorDesc_t desc{};
    desc.ndim = shape.sample_dim();
    desc.data = t_->raw_mutable_data();
    desc.dtype = t_->type();
    desc.layout = GetLayout();
    desc.shape = shape.data();
    return desc;
  }

 private:
  std::shared_ptr<Tensor<Backend>> t_;
};

template <typename Backend>
RefCountedPtr<TensorWrapper<Backend>> Wrap(std::shared_ptr<Tensor<Backend>> tl) {
  return RefCountedPtr<TensorWrapper<Backend>>(new TensorWrapper<Backend>(std::move(tl)));
}

template <typename Backend>
class TensorListWrapper : public ITensorList {
 public:
  explicit TensorListWrapper(std::shared_ptr<TensorList<Backend>> tl) : tl_(std::move(tl)) {}

  void Resize(
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout) override {
    Validate(dtype);
    ValidateShape(num_samples, ndim, shapes);
    if (layout)
      Validate(TensorLayout(layout), ndim);
    std::vector<int64_t> shape_data(shapes, shapes + ndim * num_samples);
    tl_->Resize(TensorListShape<>(std::move(shape_data), num_samples, ndim), dtype);
    if (layout)
      tl_->SetLayout(layout);
  }

  void AttachBuffer(
      int num_samples,
      int ndim,
      const int64_t *shapes,
      daliDataType_t dtype,
      const char *layout,
      void *data,
      const ptrdiff_t *sample_offsets,
      daliDeleter_t deleter) override {
    ValidateShape(num_samples, ndim, shapes);
    Validate(dtype);

    if (!data && num_samples > 0) {
      for (int i = 0; i < num_samples; i++) {
        auto sample_shape = make_cspan(&shapes[i*ndim], ndim);

        if (volume(sample_shape) > 0)
          throw std::invalid_argument(
            "The pointer to the data buffer must not be null for a non-empty tensor list.");

        if (sample_offsets && sample_offsets[i])
          throw std::invalid_argument(
            "All sample_offsets must be zero when the data pointer is NULL.");
      }
    }

    TensorLayout new_layout = {};

    if (!layout) {
      if (ndim == tl_->sample_dim())
        new_layout = tl_->GetLayout();
    } else {
      new_layout = layout;
      Validate(new_layout, ndim);
    }

    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_sample_dim(ndim);
    tl_->SetLayout(new_layout);
    tl_->set_type(dtype);
    ptrdiff_t next_offset = 0;
    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();

    std::shared_ptr<void> buffer;
    if (!deleter.delete_buffer && !deleter.destroy_context) {
      buffer = std::shared_ptr<void>(data, [](void *){});
    } else {
      buffer = std::shared_ptr<void>(data, BufferDeleter{deleter, tl_->order()});
    }

    bool is_contiguous = true;
    if (sample_offsets) {
      for (int i = 0; i < num_samples; i++) {
        if (sample_offsets[i] != next_offset) {
          is_contiguous = false;
          break;
        }
        auto num_elements = volume(make_cspan(&shapes[i*ndim], ndim));
        next_offset += num_elements * element_size;
      }
    }

    if (is_contiguous) {
      tl_->SetContiguity(BatchContiguity::Contiguous);
      std::vector<int64_t> shape_data(shapes, shapes + ndim * num_samples);
      TensorListShape<> tl_shape(shape_data, num_samples, ndim);
      tl_->ShareData(
        std::move(buffer),
        next_offset,
        tl_->is_pinned(),
        tl_shape,
        dtype,
        tl_->device_id(),
        tl_->order(),
        new_layout);
    } else {
      tl_->SetContiguity(BatchContiguity::Automatic);
      next_offset = 0;

      for (int i = 0; i < num_samples; i++) {
        TensorShape<> sample_shape(make_cspan(&shapes[i*ndim], ndim));
        void *sample_data;
        size_t sample_bytes = volume(sample_shape) * element_size;
        if (sample_offsets) {
          sample_data = static_cast<char *>(data) + sample_offsets[i];
        } else {
          sample_data = static_cast<char *>(data) + next_offset;
          next_offset += sample_bytes;
        }
        tl_->SetSample(
          i,
          std::shared_ptr<void>(buffer, sample_data),
          sample_bytes,
          tl_->is_pinned(),
          sample_shape,
          dtype,
          tl_->device_id(),
          tl_->order(),
          new_layout);
      }
    }
  }

  void AttachSamples(
      int num_samples,
      int ndim,
      daliDataType_t dtype,
      const char *layout,
      const daliTensorDesc_t *samples,
      const daliDeleter_t *sample_deleters) override {
    ValidateNumSamples(num_samples);
    if (num_samples > 0 && !samples)
      throw std::invalid_argument("The pointer to sample descriptors must not be NULL.");
    if (ndim < 0) {
      if (num_samples == 0)
        throw std::invalid_argument(
          "The number of dimensions must not be negative when num_samples is 0.");
      else
        ndim = samples[0].ndim;
    }
    if (dtype == DALI_NO_TYPE) {
      if (num_samples == 0)
        throw std::invalid_argument(
          "A valid data type must be provided when there's no sample to take it from.");
      dtype = samples[0].dtype;
    }
    Validate(dtype);

    for (int i = 0; i < num_samples; i++) {
      if (ndim && !samples[i].shape)
        throw std::invalid_argument(make_string("Got NULL shape in sample ", i, "."));
      if (samples[i].dtype != dtype)
        throw std::invalid_argument(make_string("Unexpected data type in sample ", i, ". Got: ",
          samples[i].dtype, ", expected ", dtype, "."));
      ValidateSampleShape(i, make_cspan(samples[i].shape, samples[i].ndim), ndim);;

      if (!samples[i].data && volume(make_cspan(samples[i].shape, ndim)))
        throw std::invalid_argument(make_string(
            "Got NULL data pointer in a non-empty sample ", i, "."));
    }

    TensorLayout new_layout = {};

    if (!layout) {
      if (ndim == tl_->sample_dim())
        new_layout = tl_->GetLayout();
    } else {
      new_layout = layout;
      Validate(new_layout, ndim);
    }

    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_sample_dim(ndim);
    tl_->SetLayout(new_layout);

    auto deletion_order = tl_->order();

    auto type_info = TypeTable::GetTypeInfo(dtype);
    auto element_size = type_info.size();
    for (int i = 0; i < num_samples; i++) {
      TensorShape<> sample_shape(make_cspan(samples[i].shape, samples[i].ndim));
      size_t sample_bytes = volume(sample_shape) * element_size;
      std::shared_ptr<void> sample_ptr;
      if (sample_deleters) {
        sample_ptr = std::shared_ptr<void>(
          samples[i].data,
          BufferDeleter{sample_deleters[i], deletion_order});
      } else {
        sample_ptr = std::shared_ptr<void>(samples[i].data, [](void*) {});
      }

      tl_->SetSample(
        i,
        sample_ptr,
        sample_bytes,
        tl_->is_pinned(),
        sample_shape,
        dtype,
        tl_->device_id(),
        tl_->order(),
        new_layout);
    }
  }

  daliBufferPlacement_t GetBufferPlacement() const override {
    daliBufferPlacement_t placement;
    placement.device_id = tl_->device_id();
    StorageDevice dev = backend_to_storage_device<Backend>::value;
    placement.device_type = static_cast<daliStorageDevice_t>(dev);
    placement.pinned = tl_->is_pinned();
    return placement;
  }

  void SetStream(std::optional<cudaStream_t> stream, bool synchronize) override {
    tl_->set_order(stream.has_value() ? AccessOrder(*stream) : AccessOrder::host(), synchronize);
  }

  void SetLayout(const char *layout_string) {
    if (layout_string) {
      TensorLayout layout(layout_string);
      Validate(layout, tl_->sample_dim());
      tl_->SetLayout(layout);
    } else {
      tl_->SetLayout("");
    }
  }

  const char *GetLayout() const override {
    auto &layout = tl_->GetLayout();
    return !layout.empty() ? layout.data() : nullptr;
  }

  std::optional<cudaStream_t> GetStream() const override {
    auto o = tl_->order();
    if (o.is_device())
      return o.stream();
    else
      return std::nullopt;
  }

  std::optional<cudaEvent_t> GetReadyEvent() const override {
    auto &e = tl_->ready_event();
    if (e)
      return e.get();
    else
      return std::nullopt;
  }

  cudaEvent_t GetOrCreateReadyEvent() override {
    auto &e = tl_->ready_event();
    if (e)
      return e.get();
    int device_id = tl_->device_id();
    if (device_id < 0)
      throw std::runtime_error("The tensor list is not associated with a CUDA device.");
    tl_->set_ready_event(CUDASharedEvent::Create(device_id));
    return tl_->ready_event().get();
  }

  daliTensorDesc_t GetTensorDesc(int sample) const override {
    auto &shape = tl_->shape();
    if (sample < 0 || sample >= shape.num_samples())
      throw std::out_of_range(make_string("The sample index ", sample, " is out of range. "
        "Valid indices are [0..", shape.num_samples() - 1, "]."));
    daliTensorDesc_t desc{};
    desc.ndim = shape.sample_dim();
    desc.data = tl_->raw_mutable_tensor(sample);
    desc.dtype = tl_->type();
    desc.layout = GetLayout();
    desc.shape = shape.tensor_shape_span(sample).data();
    return desc;
  }

  RefCountedPtr<ITensor> ViewAsTensor() const override {
    if (!tl_->IsContiguous())
      throw std::runtime_error(
        "The TensorList is not contiguous and cannot be viewed as a Tensor.");

    auto t = std::make_shared<Tensor<Backend>>();
    auto buf = unsafe_owner(*tl_);
    auto &lshape = tl_->shape();
    TensorShape<> tshape = shape_cat(lshape.num_samples(), lshape[0]);
    t->ShareData(
      std::move(buf),
      tl_->nbytes(),
      tl_->is_pinned(),
      tshape,
      tl_->type(),
      tl_->device_id(),
      tl_->order(),
      tl_->ready_event());
    TensorLayout layout = tl_->GetLayout();
    if (layout.size() == lshape.sample_dim()) {
      t->SetLayout("N" + layout);
    }
    return Wrap(std::move(t));
  }

 private:
  std::shared_ptr<TensorList<Backend>> tl_;
};

template <typename Backend>
RefCountedPtr<TensorListWrapper<Backend>> Wrap(std::shared_ptr<TensorList<Backend>> tl) {
  return RefCountedPtr<TensorListWrapper<Backend>>(new TensorListWrapper<Backend>(std::move(tl)));
}


ITensor *ToPointer(daliTensor_h handle);
ITensorList *ToPointer(daliTensorList_h handle);

}  // namespace c_api
}  // namespace dali

#endif  // DALI_C_API_2_DATA_OBJECTS_H_
