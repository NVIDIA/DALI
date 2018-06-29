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

#include <chrono>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "dali/tensorflow/tfallocator.h"

#include "dali/pipeline/dali.pb.h"
#include "dali/pipeline/pipeline.h"
#include "dali/c_api/c_api.h"
#include "dali/common.h"
#include "dali/error_handling.h"

typedef std::chrono::high_resolution_clock Clock;

namespace tf = tensorflow;

#define USE_TF_ALLOCATOR 0

#define TF_DALI_CALL(FUNC)                                                         \
    do {                                                                           \
      try {                                                                        \
        FUNC;                                                                      \
      } catch (std::runtime_error& e) {                                            \
        std::string error = "DALI " + std::string(#FUNC)                           \
                            + " failed: " + std::string(e.what());                 \
        std::cout << error << std::endl;                                            \
        context->SetStatus(tf::errors::Internal(error));                           \
       return;                                                                     \
      }                                                                            \
    } while (0)

tf::TensorShape DaliToShape(int64_t* ns) {
  tf::TensorShape ts;
  for (int i = 0; ns[i] != 0; ++i)
    ts.InsertDim(i, ns[i]);
  delete ns;
  return ts;
}

REGISTER_OP("Dali")
  .Attr("serialized_pipeline: string")
  .Attr("batch_size: int = -1")
  .Attr("height: int = 0")
  .Attr("width: int = 0")
  .Attr("num_threads: int = -1")
  .Attr("device_id: int = -1")
  .Output("batch: float")
  .Output("label: float")
  .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
    int batch_size;
    int height;
    int width;
    TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batch_size));
    TF_RETURN_IF_ERROR(c->GetAttr("height", &height));
    TF_RETURN_IF_ERROR(c->GetAttr("width", &width));
    c->set_output(0, c->MakeShape({batch_size, height, width, 3}));
    return tf::Status::OK();
  });

class DaliOp : public tf::OpKernel {
 public:
  explicit DaliOp(tf::OpKernelConstruction* context)
    : OpKernel(context) {

    std::string serialized_pipeline;
    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline", &serialized_pipeline));

    int batch_size;
    int num_threads;
    int device_id;

    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id));
    this->device_id_ = device_id;
    LOG_LINE << "Initializing...\n";

    TF_DALI_CALL(daliCreatePipeline(&pipe_handle_,
                   serialized_pipeline.c_str(),
                   serialized_pipeline.length(),
                   batch_size,
                   num_threads,
                   device_id));

#if USE_TF_ALLOCATOR
    SetupTFAllocator(device_id_);
    UpdateTFAllocaterContext<tf::OpKernelConstruction>(context, device_id_);
#endif
    LOG_LINE << "Pipeline created\n";
    TF_DALI_CALL(daliRun(&pipe_handle_));
    LOG_LINE << "After first run\n";
  }

  ~DaliOp() override {
    daliDeletePipeline(&pipe_handle_);
  }

  void Compute(tf::OpKernelContext* context) override {
    auto total_s = Clock::now();
    LOG_LINE << "Computing...\n";
#if USE_TF_ALLOCATOR
    UpdateTFAllocaterContext<tf::OpKernelContext>(context, device_id_);
#endif
    LOG_LINE << "Updated context\n";
    auto s = Clock::now();
    TF_DALI_CALL(daliRun(&pipe_handle_));
    int64_t run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - s).count();
    LOG_LINE << "Before output...\n";

    s = Clock::now();
    TF_DALI_CALL(daliOutput(&pipe_handle_));
    int64_t output_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - s).count();
    LOG_LINE << "After output...\n";

    s = Clock::now();
    // Classification
    int64_t* data_tensor_shape;
    int64_t* label_tensor_shape;
    TF_DALI_CALL(data_tensor_shape = daliShapeAt(&pipe_handle_, 0));
    TF_DALI_CALL(label_tensor_shape = daliShapeAt(&pipe_handle_, 1));

    tf::Tensor* data_output_tensor = NULL;
    tf::Tensor* label_output_tensor = NULL;
    tf::TensorShape data_output_shape = DaliToShape(data_tensor_shape);
    tf::TensorShape label_output_shape = DaliToShape(label_tensor_shape);
    OP_REQUIRES_OK(context,
        context->allocate_output(0, data_output_shape, &data_output_tensor));
    OP_REQUIRES_OK(context,
        context->allocate_output(1, label_output_shape, &label_output_tensor));

    int64_t allocate_time =  std::chrono::duration_cast<std::chrono::microseconds>(
                             Clock::now() - s).count();

    s = Clock::now();
    TF_DALI_CALL(daliCopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(data_output_tensor->flat<float>().data()),
        0));
    int64_t copy0_time =  std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - s).count();

    s = Clock::now();
    TF_DALI_CALL(daliCopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(label_output_tensor->flat<float>().data()),
        1));
    int64_t copy1_time =  std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - s).count();

    int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - total_s).count();
    LOG_LINE << "[TIMES] TOTAL " << total_time << " RUN " << run_time
      << " - OUTPUT " << output_time << " - ALLOC " << allocate_time
      << " - COPY0 " << copy0_time << " - COPY1 " << copy1_time << std::endl;
  }

 private:
  daliPipelineHandle pipe_handle_;
  int device_id_;
};

REGISTER_KERNEL_BUILDER(Name("Dali").Device(tf::DEVICE_GPU), DaliOp);
