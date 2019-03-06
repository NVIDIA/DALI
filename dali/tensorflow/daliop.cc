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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

#define USE_TF_ALLOCATOR 0

#if USE_TF_ALLOCATOR
#include "dali/tensorflow/tfallocator.h"
#endif

#include "dali/common.h"
#include "dali/c_api/c_api.h"

typedef std::chrono::high_resolution_clock Clock;

namespace tf = tensorflow;

#define TF_DALI_CALL(FUNC)                                                         \
    do {                                                                           \
      try {                                                                        \
        FUNC;                                                                      \
      } catch (std::runtime_error& e) {                                            \
        std::string error = "DALI " + std::string(#FUNC)                           \
                            + " failed: " + std::string(e.what());                 \
        std::cout << error << std::endl;                                           \
        context->SetStatus(tf::errors::Internal(error));                           \
       return;                                                                     \
      }                                                                            \
    } while (0)

struct CDeleter {
  void operator()(void *p) {
    free(p);
  }
};

template <typename T>
using AutoCPtr = std::unique_ptr<T, CDeleter>;

static tf::TensorShape DaliToShape(const AutoCPtr<int64_t>& ns) {
  tf::TensorShape ts;
  for (int i = 0; ns.get()[i] != 0; ++i)
    ts.InsertDim(i, ns.get()[i]);
  return ts;
}

REGISTER_OP("Dali")
  .Attr("serialized_pipeline: string")
  .Attr("shapes: list(shape) >= 1")
  .Attr("num_threads: int = -1")
  .Attr("device_id: int = -1")
  .Attr("prefetch_queue_depth: int = 2")
  .Attr("sparse: list(bool) = []")
  .Attr("batch_size: int = -1")
  .Output("data: dtypes")
  .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
  // To prevent replacing DALI op with constant tensor during TF constant folding process
  .SetIsStateful()
  .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
    std::vector<tf::PartialTensorShape> shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
    for (unsigned i = 0; i < shapes.size(); ++i) {
      if (shapes[i].dims() > 0) {
        tf::shape_inference::ShapeHandle passed_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
        TF_RETURN_IF_ERROR(
            c->WithRank(passed_shape, shapes[i].dims(), &passed_shape));
        c->set_output(i, passed_shape);
      }
    }
    return tf::Status::OK();
  })
  .Doc(R"doc(
DALI TensorFlow plugin

Creates a Dali pipeline for classification tasks from serialized DALI pipeline (given in `serialized_pipeline` parameter).
`shapes` must match the shape of the coresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the coresponding DALI Pipeline output tensors type.
 )doc");

class DaliOp : public tf::OpKernel {
 public:
  explicit DaliOp(tf::OpKernelConstruction* context)
    : OpKernel(context) {

    std::string serialized_pipeline;
    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline", &serialized_pipeline));

    int num_threads;
    int device_id;
    int batch_size;

    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &types_));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id));
    OP_REQUIRES_OK(context, context->GetAttr("prefetch_queue_depth", &prefetch_queue_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("sparse", &sparse_));
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size));

    // TF doing constant propagation runs all operators on the CPU first, so we need to provide
    // ability to copy memory from the GPU pipeline to the CPU seamlessly
    this->device_type_ = (context->device_type() == "CPU") ?
                          device_type_t::CPU : device_type_t::GPU;
    if (std::any_of(sparse_.begin(), sparse_.end(), [] (const bool &v) {return v;}) &&
        this->device_type_ == device_type_t::GPU) {
      OP_REQUIRES_OK(context, tf::errors::Internal("Cannot output sparse tensors on the GPU"));
    }
    this->device_id_ = device_id;
    LOG_LINE << "Initializing...\n";

    if (batch_size < 0) {
      batch_size = shapes_[0].dim_size(0);
    }

    TF_DALI_CALL(daliCreatePipeline(&pipe_handle_,
                   serialized_pipeline.c_str(),
                   serialized_pipeline.length(),
                   batch_size,
                   num_threads,
                   device_id,
                   prefetch_queue_depth_));

#if USE_TF_ALLOCATOR
    SetupTFAllocator(device_id_);
    UpdateTFAllocaterContext<tf::OpKernelConstruction>(context, device_id_);
#endif
    LOG_LINE << "Pipeline created\n";
    LOG_LINE << "Prefetching...\n";
    for (int i = 0; i < prefetch_queue_depth_; ++i) {
      TF_DALI_CALL(daliRun(&pipe_handle_));
    }
    LOG_LINE << "After first run\n";
  }

  ~DaliOp() override {
    daliDeletePipeline(&pipe_handle_);
  }

  void Compute(tf::OpKernelContext* context) override {
    auto total_s = Clock::now();

#if USE_TF_ALLOCATOR
    UpdateTFAllocaterContext<tf::OpKernelContext>(context, device_id_);
    LOG_LINE << "Updated context\n";
#endif
    LOG_LINE << "Before output...\n";

    auto s = Clock::now();
    TF_DALI_CALL(daliShareOutput(&pipe_handle_));
    int64_t output_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - s).count();
    LOG_LINE << "After output...\n";

    s = Clock::now();

    tf::OpOutputList outputs;
    std::vector<tf::Tensor*> data_output_tensors;
    // each sparse tensor need 3 tensors in total - values, indices and shape
    unsigned additional_sparse_tensors = std::accumulate(sparse_.begin(), sparse_.end(), 0) * 2;
    unsigned dali_num_out = daliGetNumOutput(&pipe_handle_);
    data_output_tensors.resize(dali_num_out + additional_sparse_tensors);

    OP_REQUIRES_OK(context, context->output_list("data", &outputs));

    for (unsigned i = 0, j = 0; i < dali_num_out; ++i, ++j) {
      bool should_be_sparse_tensor = i < sparse_.size() && sparse_[i];
      unsigned elms = 0;
      unsigned dims = 0;
      std::vector<tf::int64> max_dims;
      if (!should_be_sparse_tensor) {
        tf::TensorShape data_output_shape;
        TF_DALI_CALL(data_output_shape = DaliToShape(AutoCPtr<int64_t>(
                     daliShapeAt(&pipe_handle_, i))));
        // If tensor has shape provided it need to match
        OP_REQUIRES(context, shapes_[i].dims() <= 0 || data_output_shape == shapes_[i],
        tf::errors::InvalidArgument("DALI pipeline output shape at " + std::to_string(i) +
                                    " " + data_output_shape.DebugString() + " != "
                                    + shapes_[i].DebugString() + " plugin `shapes` argument"));
        OP_REQUIRES_OK(context, outputs.allocate(j, data_output_shape, &data_output_tensors[j]));
      } else {
        elms = daliNumTensors(&pipe_handle_, i);
        // maximum number of dimension + one additional to hold tensor list number
        dims = daliMaxDimTensors(&pipe_handle_, i) + 1;
        max_dims.resize(dims, 0);
        // first dim is number of elements in the tensor list
        max_dims[0] = elms;
        tf::TensorShape data_output_shape;
        tf::int64 total_elms = 0;
        TF_DALI_CALL(total_elms = daliNumElements(&pipe_handle_, i));
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({total_elms, dims}),
                                                 &data_output_tensors[j]));
        tf::Tensor* out_tensor = data_output_tensors[j];
        auto p_out_indices = out_tensor->flat<tf::int64>().data();
        for (unsigned n = 0; n < elms; ++n) {
          TF_DALI_CALL(data_output_shape = DaliToShape(AutoCPtr<int64_t>(
                       daliShapeAtSample(&pipe_handle_, i, n))));
          // it seems that num_elements() return 1 for empty tensors
          if (data_output_shape.dims() == 0) {
            continue;
          }
          // squeeze
          if (data_output_shape.dim_size(data_output_shape.dims() - 1) == 1) {
            data_output_shape.RemoveLastDims(1);
          }
          for (unsigned elm_idx = 0; elm_idx < data_output_shape.num_elements(); ++elm_idx) {
            unsigned idx_val = elm_idx;
            // first value of indices is tensor index
            *p_out_indices = n;
            ++p_out_indices;
            for (unsigned k = 0; k < dims - 1; ++k, ++p_out_indices) {
              const int dims_idx = k - (dims - 1 - data_output_shape.dims());
              // if current element has less dims than max then set first idices to 0
              if (k + data_output_shape.dims() < dims - 1) {
                *p_out_indices = 0;
              } else {
                max_dims[k + 1] = std::max(max_dims[k + 1], data_output_shape.dim_size(dims_idx));
                if (dims_idx < data_output_shape.dims() - 1) {
                  *p_out_indices = idx_val / data_output_shape.dim_size(dims_idx + 1);
                  idx_val %= data_output_shape.dim_size(dims_idx + 1);
                } else {
                  *p_out_indices = idx_val;
                }
              }
            }
          }
        }
        ++j;
        // allocate output
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({total_elms}),
                                                 &data_output_tensors[j]));
      }
      void *dst = nullptr;
      tf::Tensor* out_tensor = data_output_tensors[j];
      if (daliTensorSize(&pipe_handle_, i) > out_tensor->TotalBytes()) {
        context->CtxFailure(__FILE__, __LINE__,
            tf::errors::InvalidArgument("Output " + std::to_string(i) +
              " has bigger size than allocated by TensorFlow - check if type requested matches" +
              " with one produced by the DALI pipeline"));
      }
      switch (types_[j]) {
        case tf::DT_HALF:
              dst = reinterpret_cast<void*>(out_tensor->flat<uint16_t>().data());
          break;
        case tf::DT_FLOAT:
              dst = reinterpret_cast<void*>(out_tensor->flat<float>().data());
          break;
        case tf::DT_UINT8:
              dst = reinterpret_cast<void*>(out_tensor->flat<uint8_t>().data());
          break;
        case tf::DT_INT16:
              dst = reinterpret_cast<void*>(out_tensor->flat<int16_t>().data());
          break;
        case tf::DT_INT32:
              dst = reinterpret_cast<void*>(out_tensor->flat<int32_t>().data());
          break;
        case tf::DT_INT64:
              dst = reinterpret_cast<void*>(out_tensor->flat<tf::int64>().data());
          break;
        default:
          context->CtxFailure(__FILE__, __LINE__,
            tf::errors::InvalidArgument("Unsupported type: " + tf::DataTypeString(types_[i]) +
                                        "for tensor " + std::to_string(i)));
          break;
      }
      if (!should_be_sparse_tensor) {
        TF_DALI_CALL(daliCopyTensorNTo(&pipe_handle_, dst, i, this->device_type_));
      } else {
        // copy values
        TF_DALI_CALL(daliCopyTensorListNTo(&pipe_handle_, dst, i, this->device_type_));
        ++j;
        // copy out shape
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({dims}),
                                                 &data_output_tensors[j]));
        auto out_tensor = data_output_tensors[j];
        auto out_shape = out_tensor->flat<tf::int64>().data();
        for (unsigned k = 0; k < dims; ++k) {
          out_shape[k] = max_dims[k];
        }
      }
    }
    int64_t copy_time =  std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - s).count();

    TF_DALI_CALL(daliOutputRelease(&pipe_handle_));

    LOG_LINE << "Computing...\n";
    s = Clock::now();
    TF_DALI_CALL(daliRun(&pipe_handle_));
    int64_t run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - s).count();

    int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - total_s).count();

    LOG_LINE << "[TIMES] TOTAL " << total_time << " RUN " << run_time
      << " - OUTPUT " << output_time << " - ALLOC + COPY " << copy_time << std::endl;
  }

 private:
  daliPipelineHandle pipe_handle_;
  std::vector<tf::TensorShape> shapes_;
  tf::DataTypeVector types_;
  int device_id_;
  int prefetch_queue_depth_;
  device_type_t device_type_;
  std::vector<bool> sparse_;
};

using tf::int64;

REGISTER_KERNEL_BUILDER(Name("Dali").Device(tf::DEVICE_GPU), DaliOp)
REGISTER_KERNEL_BUILDER(Name("Dali").Device(tf::DEVICE_CPU), DaliOp)
