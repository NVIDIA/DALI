// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "ndll/pipeline/ndll.pb.h"
#include "ndll/pipeline/pipeline.h"

using namespace tensorflow;

REGISTER_OP("Ndll")
  .Attr("serialized_pipeline: string")
  .Attr("batch_size: int = 128")
  .Attr("num_threads: int = 2")
  .Attr("device_id: int = 0")
  .Output("batch: int32")
  // ignoring shape for now
  // maybe this one
  // .SetShapeFn(shape_inference::RandomShape);
  // or this one
  .SetShapeFn(shape_inference::UnknownShape);
/*  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, ??)
    return Status::OK();
  })*/


class NdllOp : Public OpKernel {
 public:
  explicit NdllOp(OpKernelConstruction* context)
    : OpKernel(context),
      first_iter(true) {
    std::string serialized_pipeline;
    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline", &serialized_pipeline));

    int batch_size;
    int num_threads;
    int device_id;

    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id));

    pipeline_.reset(new Pipeline(serialized_pipeline,
                                 batch_size,
                                 num_threads,
                                 device_id));
    pipeline_->RunCPU();
    pipeline_->RunGPU();
  }

  void Compute(OpKernelContext* context) override {
    pipeline_->RunCPU();
    pipeline_->RunGPU();

    DeviceWorkspace ws;
    pipeline_->Outputs(&ws);

    for (int i = 0; i < ws.NumOutput(); ++i) {
      if (ws.OutputIsType<CPUBackend>(i)) {
        ws.Output<CPUBackend>(i);
      } else {
        ws.Output<GPUBackend>(i);
      }
    }

    //TODO(spanev) copy ws->Outputs to output_tensor

    // TODO(spanev) find how to allocate_output: cudamemcpy ndll tensor to tf
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, );
  }
 private:
  std::unique_ptr<Pipeline> pipeline_;
  bool first_iter;
};

REGISTER_KERNEL_BUILDER(Name("Ndll").Device(DEVICE_GPU), NdllOp);
