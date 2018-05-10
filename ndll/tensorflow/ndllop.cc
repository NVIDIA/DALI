// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "ndll/tensorflow/tfallocator.h"

#include "ndll/pipeline/ndll.pb.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/c_api/c_api.h"
#include "ndll/common.h"

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

namespace tf = tensorflow;

tf::TensorShape NdllToShape(int64_t* ns) {
  tf::TensorShape ts;
  for (int i = 0; ns[i] != 0; ++i)
    ts.InsertDim(i, ns[i]);
  delete ns;
  return ts;
}

REGISTER_OP("Ndll")
  .Attr("serialized_pipeline: string")
  .Attr("batch_size: int = 128")
  .Attr("height: int = 0")
  .Attr("width: int = 0")
  .Attr("num_threads: int = 2")
  .Attr("device_id: int = 0")
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

class NdllOp : public tf::OpKernel {
 public:
  explicit NdllOp(tf::OpKernelConstruction* context)
    : OpKernel(context) {

    std::string serialized_pipeline;
    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline", &serialized_pipeline));

    int batch_size;
    int num_threads;
    int device_id;

    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id));
    LOG_LINE << "Initializing...\n";

    // TODO(spanev) Use TF allocator
    CreatePipeline(&pipe_handle_,
                   serialized_pipeline.c_str(),
                   serialized_pipeline.length(),
                   batch_size,
                   num_threads,
                   device_id);

    SetupTFAllocator();
    UpdateTFAllocaterContext<tf::OpKernelConstruction>(context);

    LOG_LINE << "Pipeline created\n";
    Run(&pipe_handle_);
    LOG_LINE << "After first run\n";
  }

  ~NdllOp() override {
    DeletePipeline(&pipe_handle_);
  }

  void Compute(tf::OpKernelContext* context) override {
    auto total_s = Clock::now();
    LOG_LINE << "Computing...\n";
    std::cout << context << " " << context->num_inputs() << std::endl;
    UpdateTFAllocaterContext<tf::OpKernelContext>(context);

    LOG_LINE << "Updated context\n";
    auto s = Clock::now();
    Run(&pipe_handle_);
    long run_time =  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - s).count();
    LOG_LINE << "Before output...\n";

    s = Clock::now();
    Output(&pipe_handle_);
    long output_time =  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - s).count();
    LOG_LINE << "After output...\n";

    s = Clock::now();
    // Classification
    int64_t* data_tensor_shape = ShapeAt(&pipe_handle_, 0);
    int64_t* label_tensor_shape = ShapeAt(&pipe_handle_, 1);

    tf::Tensor* data_output_tensor = NULL;
    tf::Tensor* label_output_tensor = NULL;
    tf::TensorShape data_output_shape = NdllToShape(data_tensor_shape);
    tf::TensorShape label_output_shape = NdllToShape(label_tensor_shape);
    OP_REQUIRES_OK(context,
        context->allocate_output(0, data_output_shape, &data_output_tensor));
    OP_REQUIRES_OK(context,
        context->allocate_output(1, label_output_shape, &label_output_tensor));

    long allocate_time =  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - s).count();

    s = Clock::now();
    CopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(data_output_tensor->flat<float>().data()),
        0);
    long copy0_time =  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - s).count();

    s = Clock::now();
    CopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(label_output_tensor->flat<float>().data()),
        1);
    long copy1_time =  std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - s).count();

    long total_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - total_s).count();
    LOG_LINE << "[TIMES] TOTAL " << total_time << " RUN " << run_time << " - OUTPUT " << output_time
      << " - ALLOC " << allocate_time << " - COPY0 " << copy0_time << " - COPY1 " << copy1_time << std::endl;
  }

 private:
  PipelineHandle pipe_handle_;
};

REGISTER_KERNEL_BUILDER(Name("Ndll").Device(tf::DEVICE_GPU), NdllOp);
