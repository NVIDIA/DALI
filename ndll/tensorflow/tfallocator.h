// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_TENSORFLOW_TFALLOCATOR_H_
#define NDLL_TENSORFLOW_TFALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <unordered_map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/stream_executor/stream.h"

#include "ndll/pipeline/data/allocator.h"
#include "ndll/common.h"
#include "ndll/c_api/c_api.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/pipeline/data/backend.h"

namespace tf = tensorflow;

namespace ndll {

class TFGPUAllocator : public GPUAllocator {
 public:
  explicit TFGPUAllocator(const OpSpec &spec) :
    GPUAllocator(spec),
    construction_context_(nullptr),
    context_(nullptr) {}
  virtual ~TFGPUAllocator() = default;

  void New(void **ptr, size_t bytes) override {
    LOG_LINE << "[Allocating with TF]\n";
    std::shared_ptr<tf::PersistentTensor> t = std::make_shared<tf::PersistentTensor>();
    tf::TensorShape shape;
    shape.AddDim(bytes);
    tf::Tensor* unused;
    tf::Status status;
    if (context_ == nullptr) {
      status = construction_context_->allocate_persistent(tf::DT_INT8, shape,
        t.get(), &unused);
    } else {
      status = context_->allocate_persistent(tf::DT_INT8, shape,
        t.get(), &unused);
    }

    if (!status.ok()) {
      throw status;
    }

    if (context_ != nullptr) {
      auto device_context = context_->op_device_context();
      if (device_context != nullptr) {
        device_context->stream()->BlockHostUntilDone();
      }
    }

    void* data = GetData(t);
    *ptr = data;
    allocated_tensors_.insert(std::make_pair(reinterpret_cast<intptr_t>(data), t));
  }

  void Delete(void *ptr, size_t /* unused */) override {
    LOG_LINE << "[Deleting with TF]\n";
  }

  void UpdateContext(tf::OpKernelConstruction* context) {
    construction_context_ = context;
  }

  void UpdateContext(tf::OpKernelContext* context) {
    context_ = context;
  }

 private:
  tf::OpKernelConstruction* construction_context_;
  tf::OpKernelContext* context_;
  std::unordered_map<intptr_t, std::shared_ptr<tf::PersistentTensor> > allocated_tensors_;

  void* GetData(std::shared_ptr<tf::PersistentTensor> t) {
    if (context_ == nullptr) {
      return const_cast<void*>(
          reinterpret_cast<const void*>(
            t->AccessTensor(construction_context_)->tensor_data().data()));
    } else {
      return const_cast<void*>(
          reinterpret_cast<const void*>(
            t->AccessTensor(context_)->tensor_data().data()));
    }
  }
};

NDLL_DECLARE_OPTYPE_REGISTRY(TFGPUAllocator, TFGPUAllocator);

}  // namespace ndll

void SetupTFAllocator(int device_id) {
  int dev;
  CUDA_CALL(cudaGetDevice(&dev));
  CUDA_CALL(cudaSetDevice(device_id));
  ndll::OpSpec spec("TFGPUAllocator");
  std::unique_ptr<ndll::GPUAllocator> allocator(new ndll::TFGPUAllocator(spec));
  ndll::SetGPUAllocator(std::move(allocator));
  CUDA_CALL(cudaSetDevice(dev));
}

template <typename Ctx>
void UpdateTFAllocaterContext(Ctx* context, int device_id) {
  int dev;
  CUDA_CALL(cudaGetDevice(&dev));
  CUDA_CALL(cudaSetDevice(device_id));
  ndll::TFGPUAllocator& tfGPUAllocator = dynamic_cast<ndll::TFGPUAllocator&>(
                                                        ndll::GetGPUAllocator());
  tfGPUAllocator.UpdateContext(context);
  CUDA_CALL(cudaSetDevice(dev));
}

#endif  // NDLL_TENSORFLOW_TFALLOCATOR_H_
