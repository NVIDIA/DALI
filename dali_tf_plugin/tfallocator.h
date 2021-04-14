// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TF_PLUGIN_TFALLOCATOR_H_
#define DALI_TF_PLUGIN_TFALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <unordered_map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/stream_executor/stream.h"

#include "dali/pipeline/data/allocator.h"
#include "dali/core/common.h"
#include "dali/c_api.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/data/backend.h"

namespace tf = tensorflow;

namespace dali {

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
        // Check status in TF way
        SE_CHECK_OK(device_context->stream()->BlockHostUntilDone());
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

DALI_DECLARE_OPTYPE_REGISTRY(TFGPUAllocator, TFGPUAllocator);

}  // namespace dali

void SetupTFAllocator(int device_id) {
  int dev;
  CUDA_CALL(cudaGetDevice(&dev));
  CUDA_CALL(cudaSetDevice(device_id));
  dali::OpSpec spec("TFGPUAllocator");
  std::unique_ptr<dali::GPUAllocator> allocator(new dali::TFGPUAllocator(spec));
  dali::SetGPUAllocator(std::move(allocator));
  CUDA_CALL(cudaSetDevice(dev));
}

template <typename Ctx>
void UpdateTFAllocaterContext(Ctx* context, int device_id) {
  int dev;
  CUDA_CALL(cudaGetDevice(&dev));
  CUDA_CALL(cudaSetDevice(device_id));
  dali::TFGPUAllocator& tfGPUAllocator = dynamic_cast<dali::TFGPUAllocator&>(
                                                        dali::GetGPUAllocator());
  tfGPUAllocator.UpdateContext(context);
  CUDA_CALL(cudaSetDevice(dev));
}

#endif  // DALI_TF_PLUGIN_TFALLOCATOR_H_
