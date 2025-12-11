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
#include <string>
#include <utility>
#include <vector>
#include "dali/dali.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/copy_to_external.h"
#include "dali/c_api_2/ref_counting.h"
#include "dali/c_api_2/validation.h"


// A dummy base that the handle points to
struct _DALITensorList {
 protected:
  _DALITensorList() = default;
  ~_DALITensorList() = default;
};

struct _DALITensor {
 protected:
  _DALITensor() = default;
  ~_DALITensor() = default;
};

namespace dali {
namespace c_api {

constexpr mm::memory_kind_id GetMemoryKind(const daliBufferPlacement_t &placement) {
  if (placement.device_type == DALI_STORAGE_GPU) {
    return mm::memory_kind_id::device;
  } else {
    assert(placement.device_type == DALI_STORAGE_CPU);
    return placement.pinned ? mm::memory_kind_id::pinned : mm::memory_kind_id::host;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Interfaces
//////////////////////////////////////////////////////////////////////////////


/** A DALI C API Tensor interface
 *
 * Please refer to the relevant C API documentation - e.g. for Resize, see daliTensorResize.
 */
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

  virtual const TensorShape<> &GetShape() const & = 0;

  virtual size_t GetByteSize() const = 0;

  virtual daliDataType_t GetDType() const = 0;

  virtual const char *GetSourceInfo() const & = 0;

  virtual void SetSourceInfo(const char *source_info) = 0;

  virtual void CopyOut(
      void *dst_buffer,
      daliBufferPlacement_t dst_buffer_placement,
      std::optional<cudaStream_t> stream,
      daliCopyFlags_t flags) = 0;


  /** Retrieves the underlying DALI Tensor<Backend> pointer.
   *
   * Returns a shared pointer to the underlying DALI object. If the backend doesn't match,
   * a null pointer is returned.
   */
  template <typename Backend>
  const std::shared_ptr<Tensor<Backend>> &Unwrap() const &;

  static RefCountedPtr<ITensor> Create(daliBufferPlacement_t placement);
};


/** A DALI C API TensorList interface
 *
 * Please refer to the relevant C API documentation - e.g. for Resize, see daliTensorListResize.
 */
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

  virtual const TensorListShape<> &GetShape() const & = 0;

  virtual size_t GetByteSize() const = 0;

  virtual daliDataType_t GetDType() const = 0;

  virtual RefCountedPtr<ITensor> ViewAsTensor() const = 0;

  virtual const char *GetSourceInfo(int sample) const & = 0;

  virtual void SetSourceInfo(int sample, const char *source_info) = 0;

  virtual void CopyOut(
    void *dst_buffer,
    daliBufferPlacement_t dst_buffer_placement,
    std::optional<cudaStream_t> stream,
    daliCopyFlags_t flags) = 0;

  /** Retrieves the underlying DALI TensorList<Backend> pointer.
   *
   * Returns a shared pointer to the underlying DALI object. If the backend doesn't match,
   * a null pointer is returned.
   */
  template <typename Backend>
  const std::shared_ptr<TensorList<Backend>> &Unwrap() const &;

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

  void SetLayout(const char *layout_string) override {
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

  const TensorShape<> &GetShape() const & override {
    return t_->shape();
  }

  size_t GetByteSize() const override {
    return t_->nbytes();
  }

  daliDataType_t GetDType() const override {
    return t_->type();
  }

  const char *GetSourceInfo() const & override {
    const char *info = t_->GetMeta().GetSourceInfo().c_str();
    if (info && !*info)
      return nullptr;
    return info;
  }

  void SetSourceInfo(const char *source_info) override {
    t_->SetSourceInfo(source_info ? source_info : "");
  }

  void CopyOut(
      void *dst_buffer,
      daliBufferPlacement_t dst_buffer_placement,
      std::optional<cudaStream_t> stream,
      daliCopyFlags_t flags) override {
    Validate(dst_buffer_placement);
    AccessOrder order = stream ? *stream : t_->order();
    mm::memory_kind_id mem_kind = GetMemoryKind(dst_buffer_placement);
    std::optional<int> dev_id;
    if (dst_buffer_placement.device_type == DALI_STORAGE_GPU)
      dev_id = dst_buffer_placement.device_id;
    CopyToExternal(dst_buffer, mem_kind, dev_id, *t_, order, flags & DALI_COPY_USE_KERNEL);
    if (flags & DALI_COPY_SYNC)
        AccessOrder::host().wait(order);
  }

  const auto &NativePtr() const & {
    return t_;
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
      ValidateNDim(ndim);
    }
    if (dtype == DALI_NO_TYPE) {
      if (num_samples == 0)
        throw std::invalid_argument(
          "A valid data type must be provided when there's no sample to take it from.");
      dtype = samples[0].dtype;
    }
    Validate(dtype);

    TensorLayout new_layout = {};

    if (!layout) {
      if (num_samples > 0) {
        new_layout = samples[0].layout;
        Validate(new_layout, ndim);
      } else if (ndim == tl_->sample_dim()) {
        new_layout = tl_->GetLayout();
      }
    } else {
      new_layout = layout;
      Validate(new_layout, ndim);
    }

    for (int i = 0; i < num_samples; i++) {
      if (ndim && !samples[i].shape)
        throw std::invalid_argument(make_string("Got NULL shape in sample ", i, "."));
      if (samples[i].dtype != dtype)
        throw std::invalid_argument(make_string("Unexpected data type in sample ", i, ". Got: ",
          samples[i].dtype, ", expected ", dtype, "."));
      ValidateSampleShape(i, make_cspan(samples[i].shape, samples[i].ndim), ndim);
      if (samples[i].layout && new_layout != samples[i].layout)
        throw std::invalid_argument(make_string("Unexpected layout \"", samples[i].layout,
            "\" in sample ", i, ". Expected: \"", new_layout, "\"."));

      if (!samples[i].data && volume(make_cspan(samples[i].shape, ndim)))
        throw std::invalid_argument(make_string(
            "Got NULL data pointer in a non-empty sample ", i, "."));
    }

    tl_->Reset();
    tl_->SetSize(num_samples);
    tl_->set_type(dtype);
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

  void SetLayout(const char *layout_string) override {
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
    ValidateSampleIdx(sample);
    daliTensorDesc_t desc{};
    auto &shape = tl_->shape();
    desc.ndim = shape.sample_dim();
    desc.data = tl_->raw_mutable_tensor(sample);
    desc.dtype = tl_->type();
    desc.layout = GetLayout();
    desc.shape = shape.tensor_shape_span(sample).data();
    return desc;
  }

  const TensorListShape<> &GetShape() const & override {
    return tl_->shape();
  }

  size_t GetByteSize() const override {
    return tl_->nbytes();
  }

  daliDataType_t GetDType() const override {
    return tl_->type();
  }

  const char *GetSourceInfo(int sample) const & override {
    ValidateSampleIdx(sample);
    const char *info = tl_->GetMeta(sample).GetSourceInfo().c_str();
    if (info && !*info)
      return nullptr;  // return empty string as NULL
    return info;
  }

  void SetSourceInfo(int sample, const char *source_info) override {
    ValidateSampleIdx(sample);
    tl_->SetSourceInfo(sample, source_info ? source_info : "");
  }

  RefCountedPtr<ITensor> ViewAsTensor() const override {
    if (!tl_->IsDenseTensor())
      throw std::runtime_error(
        "Only a densely packed list of tensors of uniform shape can be viewed as a Tensor.");

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

  void CopyOut(
      void *dst_buffer,
      daliBufferPlacement_t dst_buffer_placement,
      std::optional<cudaStream_t> stream,
      daliCopyFlags_t flags) override {
    Validate(dst_buffer_placement);
    AccessOrder order = stream ? *stream : tl_->order();
    mm::memory_kind_id mem_kind = GetMemoryKind(dst_buffer_placement);
    std::optional<int> dev_id;
    if (dst_buffer_placement.device_type == DALI_STORAGE_GPU)
      dev_id = dst_buffer_placement.device_id;
    CopyToExternal(dst_buffer, mem_kind, dev_id, *tl_, order, flags & DALI_COPY_USE_KERNEL);
    if (flags & DALI_COPY_SYNC)
        AccessOrder::host().wait(order);
  }

  const auto &NativePtr() const & {
    return tl_;
  }

  inline void ValidateSampleIdx(int idx) const {
    if (idx < 0 || idx >= tl_->num_samples()) {
      std::string message = make_string("The sample index ", idx, " is out of range.");
      if (tl_->num_samples() == 0)
        message += " The TensorList is empty.";
      else
        message += make_string("Valid indices are [0..", tl_->num_samples() - 1, "].");
      throw std::out_of_range(std::move(message));
    }
  }

 private:
  std::shared_ptr<TensorList<Backend>> tl_;
};

template <typename Backend>
RefCountedPtr<TensorListWrapper<Backend>> Wrap(std::shared_ptr<TensorList<Backend>> tl) {
  return RefCountedPtr<TensorListWrapper<Backend>>(new TensorListWrapper<Backend>(std::move(tl)));
}

template <typename Backend>
const std::shared_ptr<Tensor<Backend>> &ITensor::Unwrap() const & {
  return dynamic_cast<const TensorWrapper<Backend> &>(*this).NativePtr();
}

template <typename Backend>
const std::shared_ptr<TensorList<Backend>> &ITensorList::Unwrap() const & {
  return dynamic_cast<const TensorListWrapper<Backend> &>(*this).NativePtr();
}

ITensor *ToPointer(daliTensor_h handle);
ITensorList *ToPointer(daliTensorList_h handle);

}  // namespace c_api
}  // namespace dali

#endif  // DALI_C_API_2_DATA_OBJECTS_H_
