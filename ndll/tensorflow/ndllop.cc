// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "ndll/pipeline/ndll.pb.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/pipeline/tfpipeline.h"
#include "ndll/common.h"

namespace tf = tensorflow;

tf::TensorShape NdllToShape(std::vector<ndll::Index> ns) {
  tf::TensorShape ts;
  for (int i = 0; i < static_cast<int>(ns.size()); ++i) {
    ts.InsertDim(i, ns[i]);
  }
  return ts;
}

REGISTER_OP("Ndll")
  .Attr("serialized_pipeline: string")
  .Attr("batch_size: int = 128")
  .Attr("num_threads: int = 2")
  .Attr("device_id: int = 0")
  .Output("batch: float")
  .Output("label: float")
  .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
    std::string serialized_pipeline;
    TF_RETURN_IF_ERROR(c->GetAttr("serialized_pipeline", &serialized_pipeline));
    // TODO(spanev) tmp solution: get static shape from pipeline
   /* {
      PipelineHandle pipe_handle;
      std::vector<ndll::Index> data_tensor_shape = ShapeAt(&pipe_handle, 0);
      std::vector<ndll::Index> label_tensor_shape = ShapeAt(&pipe_handle, 1);
      tf::TensorShape data_shape = NdllToShape(data_tensor_shape);
      tf::TensorShape label_shape = NdllToShape(label_tensor_shape);
      tf::shape_inference::ShapeHandle data_s;
      tf::shape_inference::ShapeHandle label_s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(data_shape, &data_s));
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(label_shape, &label_s));
      c->set_output(0, data_s);
      c->set_output(1, label_s);
      delete(&pipe_handle);
    }
    */
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

    CreatePipeline(&pipe_handle_,
                   serialized_pipeline.c_str(),
                   serialized_pipeline.length(),
                   batch_size,
                   num_threads,
                   device_id);

    LOG_LINE << "Pipeline created\n";
    Run(&pipe_handle_);
    LOG_LINE << "After first run\n";
  }

  ~NdllOp() override {
    DeletePipeline(&pipe_handle_);
  }

  void Compute(tf::OpKernelContext* context) override {
    LOG_LINE << "Computing...\n";
    Run(&pipe_handle_);
    LOG_LINE << "Computing...\n";
    Output(&pipe_handle_);

    LOG_LINE << "After output...\n";
    ndll::DeviceWorkspace* ws = pipe_handle_.ws;

    // Classification
    std::vector<ndll::Index> data_tensor_shape = ShapeAt(&pipe_handle_, 0);
    std::vector<ndll::Index> label_tensor_shape = ShapeAt(&pipe_handle_, 1);

    tf::Tensor* data_output_tensor = NULL;
    tf::Tensor* label_output_tensor = NULL;
    tf::TensorShape data_output_shape = NdllToShape(data_tensor_shape);
    tf::TensorShape label_output_shape = NdllToShape(label_tensor_shape);
    OP_REQUIRES_OK(context,
        context->allocate_output(0, data_output_shape, &data_output_tensor));
    OP_REQUIRES_OK(context,
        context->allocate_output(1, label_output_shape, &label_output_tensor));

    CopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(data_output_tensor->flat<float>().data()),
        0);
    CopyTensorNTo(&pipe_handle_,
        reinterpret_cast<void*>(label_output_tensor->flat<float>().data()),
        1);
  }

 private:
  std::unique_ptr<ndll::Pipeline> pipeline_;
  PipelineHandle pipe_handle_;
};

REGISTER_KERNEL_BUILDER(Name("Ndll").Device(tf::DEVICE_GPU), NdllOp);
