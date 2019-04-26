// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
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
#include "dali/tensorflow/tf_helper.h"

#define TF_DALI_CALL(FUNC)                                                     \
    _DALI_CALL_IMPL(FUNC, _RET_ERROR)

namespace tensorflow {

namespace data {

namespace {

class DALIDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit DALIDatasetOp(OpKernelConstruction* context) : DatasetOpKernel(context) {
    bool exec_separated;
    int cpu_prefetch_queue_depth;
    std::string serialized_pipeline;

    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline",
                                             &serialized_pipeline));
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr("sparse", &sparse_));
    OP_REQUIRES_OK(context, context->GetAttr("exec_separated", &exec_separated));
    // In exec_separated==false case, gpu_prefetch_queue_depth is the global prefetch_queue_depth_
    OP_REQUIRES_OK(context, context->GetAttr("gpu_prefetch_queue_depth", &prefetch_queue_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads_));
    OP_REQUIRES_OK(context, context->GetAttr("cpu_prefetch_queue_depth",
                                             &cpu_prefetch_queue_depth));

    std::vector<int> devices;
    OP_REQUIRES_OK(context, context->GetAttr("devices", &devices));

    pipe_handles_.resize(devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
      daliCreatePipeline(&pipe_handles_[i],
                  serialized_pipeline.c_str(),
                  serialized_pipeline.length(),
                  batch_size_,
                  num_threads_,
                  devices[i],
                  exec_separated,
                  prefetch_queue_depth_,
                  cpu_prefetch_queue_depth,
                  prefetch_queue_depth_);

    }
    for (size_t i = 0; i < devices.size(); i++) {
      for (auto& handle : pipe_handles_) {
        if (!exec_separated) {
          daliPrefetchUniform(&handle, prefetch_queue_depth_);
        } else {
          daliPrefetchSeparate(&handle,
                                cpu_prefetch_queue_depth,
                                prefetch_queue_depth_);
        }
      }
    }
  }

  ~DALIDatasetOp() {
    for (auto& handle : pipe_handles_) {
      daliDeletePipeline(&handle);
    }
  }

  void MakeDataset(OpKernelContext* context, DatasetBase** output) override {
    *output = new Dataset(context, pipe_handles_, shapes_, dtypes_, sparse_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx,
                     const std::vector<daliPipelineHandle>& pipeline_handles,
                     const std::vector<TensorShape>& shapes,
                     const DataTypeVector& types,
                     const std::vector<bool>& sparse)
          : DatasetBase(DatasetContext(ctx)),
            pipeline_handles_(pipeline_handles),
            shapes_{[shapes]() {
              std::vector<PartialTensorShape> partial_shapes;
              for (size_t i = 0; i < shapes.size(); i++) {
                std::vector<int64> dims(shapes[i].dims(), -1);
                PartialTensorShape s;
                PartialTensorShape::MakePartialShape(dims.data(), dims.size(),
                                                     &s);
                partial_shapes.push_back(s);
              }
              return partial_shapes;
            }()},
            dtypes_{types},
            sparse_{sparse} {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::DALI")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() const override {
      return "DALIDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return Status::OK();
    }


   private:
    const std::vector<daliPipelineHandle>& pipeline_handles_;
    const std::vector<PartialTensorShape> shapes_;
    const DataTypeVector& dtypes_;
    const std::vector<bool>& sparse_;

    // shareded dataset "tensorflow/core/kernels/data/shard_dataset_op.cc"
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            current_idx_{0},
            num_pipelines_{dataset()->pipeline_handles_.size()} {}

      Status GetNextInternal(IteratorContext* ctx,
                            std::vector<Tensor>* out_tensors,
                            bool* end_of_sequence) override {
        // Main logic here: copy pipeline output to `out_tensors`
        mutex_lock l(mu_);
        auto current_handle = dataset()->pipeline_handles_[current_idx_];

        TF_DALI_CALL(daliShareOutput(&current_handle));


        const auto sparse = dataset()->sparse_;
        unsigned additional_sparse_tensors = std::accumulate(sparse.begin(), sparse.end(), 0) * 2;
        unsigned dali_num_out = daliGetNumOutput(&current_handle);

        out_tensors->reserve(dali_num_out + additional_sparse_tensors);

        for (unsigned i = 0, j = 0; i < dali_num_out; ++i, ++j) {
          bool should_be_sparse_tensor = i < sparse.size() && sparse[i];
          unsigned elms = 0;
          unsigned dims = 0;
          std::vector<int64> max_dims;
          if (!should_be_sparse_tensor) {
            // dense output
            TensorShape data_output_shape;
            AllocatorAttributes attrs;
            attrs.set_on_host(false);
            attrs.set_gpu_compatible(true);
            TF_DALI_CALL(data_output_shape = DaliToShape(AutoCPtr<int64_t>(
                                                         daliShapeAt(&current_handle, i))));

            out_tensors->emplace_back(ctx->allocator(attrs),
                       dataset()->dtypes_[i], data_output_shape);
          } else {
            // sparse output
          }

          auto& out = out_tensors->back();
          void* dst = nullptr;

          switch (dataset()->dtypes_[j]) {
            case DT_HALF:
                  dst = reinterpret_cast<void*>(out.flat<uint16_t>().data());
              break;
            case DT_FLOAT:
                  dst = reinterpret_cast<void*>(out.flat<float>().data());
              break;
            case DT_UINT8:
                  dst = reinterpret_cast<void*>(out.flat<uint8_t>().data());
              break;
            case DT_INT16:
                  dst = reinterpret_cast<void*>(out.flat<int16_t>().data());
              break;
            case DT_INT32:
                  dst = reinterpret_cast<void*>(out.flat<int32_t>().data());
              break;
            case DT_INT64:
                  dst = reinterpret_cast<void*>(out.flat<int64>().data());
              break;
            default:
                std::string error = "Unsupported type: " + DataTypeString(dataset()->dtypes_[i]) +
                                    "for tensor " + std::to_string(i);
                // propagate error
            break;
          }


          if (!should_be_sparse_tensor) {
            // dense output
            // TODO(spanev) change device type to dynamic one
            const device_type_t device_type = device_type_t::GPU;
            TF_DALI_CALL(daliCopyTensorNTo(&current_handle, dst, i, device_type));
          } else {
            // sparse output
          }
        }

        current_idx_ = (current_idx_ + 1) % num_pipelines_;
        return Status::OK();
      }


     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("current_idx"), current_idx_));

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        int64 current_index;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("current_index"), &current_index));
        current_idx_ = static_cast<unsigned int>(current_index);

        return Status::OK();
      }

      mutex mu_;
      unsigned int current_idx_;
      const size_t num_pipelines_;
    };  // class Iterator
  };  // class Dataset

 protected:
  std::vector<daliPipelineHandle> pipe_handles_;
  // Do we need to keep the serialized pipeline?
  // std::string serialized_pipeline_;
  int batch_size_;
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  std::vector<bool> sparse_;
  int prefetch_queue_depth_;
  int num_threads_;
};


// Regestrations
REGISTER_OP("DALIDataset")
  // TODO(spanev): Replace to string array in order to be able to have different pipeline per GPU
  .Attr("serialized_pipeline: string")
  .Attr("shapes: list(shape) >= 1")
  .Attr("num_threads: int = -1")
  .Attr("device_id: int = -1")
  .Attr("exec_separated: bool = false")
  .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
  .Attr("devices: list(int) >= 1")
  .Attr("gpu_prefetch_queue_depth: int = 2")
  .Attr("cpu_prefetch_queue_depth: int = 2")
  .Attr("sparse: list(bool) = []")
  .Attr("batch_size: int = -1")
  .Output("handle: variant")
  // To prevent replacing DALI op with constant tensor during TF constant folding process
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    std::vector<PartialTensorShape> shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
    for (unsigned i = 0; i < shapes.size(); ++i) {
      if (shapes[i].dims() > 0) {
        shape_inference::ShapeHandle passed_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
        TF_RETURN_IF_ERROR(
            c->WithRank(passed_shape, shapes[i].dims(), &passed_shape));
        c->set_output(i, passed_shape);
      }
    }
    return Status::OK();
  })
  .Doc(R"doc(
DALI Dataset plugin
Creates a DALI dataset compatible with tf.data.Dataset from a serialized DALI pipeline
(given in `serialized_pipeline` parameter).
`shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
)doc");

REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_CPU),
                        DALIDatasetOp);
REGISTER_KERNEL_BUILDER(Name("DALIDatasetOp")
                            .Device(DEVICE_GPU)
                            .HostMemory("buffer_size")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                            // .Priority(1),
DALIDatasetOp);

}  // namespace

}  // namespace data

}  // namespace tensorflow

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12
