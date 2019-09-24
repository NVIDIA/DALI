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
#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

#define USE_TF_ALLOCATOR 0

#if USE_TF_ALLOCATOR
#include "tfallocator.h"
#endif

#include "dali/core/common.h"
#include "dali/c_api/c_api.h"

#define TF_DALI_CALL(FUNC)                                                     \
    _DALI_CALL_IMPL(FUNC, _RET_ERROR)

namespace tensorflow {

namespace data {

namespace {

class DALIDatasetOp : public DatasetOpKernel {
  public:
    explicit DALIDatasetOp(OpKernelConstruction* context) 
      : DatasetOpKernel(context) { 
        OP_REQUIRES_OK(context, context->GetAttr("pipeline", &pipeline_));
        OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads_));
        OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id_));
        OP_REQUIRES_OK(context, context->GetAttr("exec_separated", &exec_separated_));
        OP_REQUIRES_OK(context, context->GetAttr("prefetch_queue_depth", &prefetch_queue_depth_));
        OP_REQUIRES_OK(context, context->GetAttr("cpu_prefetch_queue_depth", &cpu_prefetch_queue_depth_));
        OP_REQUIRES_OK(context, context->GetAttr("gpu_prefetch_queue_depth", &gpu_prefetch_queue_depth_));
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
      }

    void MakeDataset(OpKernelContext* context, DatasetBase** output) override {
      *output = new Dataset(
        context, 
        pipeline_, 
        batch_size_, 
        num_threads_,
        device_id_,
        exec_separated_,
        prefetch_queue_depth_,
        cpu_prefetch_queue_depth_,
        gpu_prefetch_queue_depth_,
        shapes_, 
        dtypes_);
    }

  private:
    std::string pipeline_;
    int batch_size_;
    int num_threads_;
    int device_id_;
    bool exec_separated_;
    int prefetch_queue_depth_;
    int cpu_prefetch_queue_depth_;
    int gpu_prefetch_queue_depth_;
    std::vector<PartialTensorShape> shapes_;
    DataTypeVector dtypes_;

    class Dataset : public DatasetBase {
      public:
        explicit Dataset(
          OpKernelContext *context,
          const std::string pipeline,
          const int batch_size,
          const int num_threads,
          const int device_id,
          const bool exec_separated,
          const int prefetch_queue_depth,
          const int cpu_prefetch_queue_depth,
          const int gpu_prefetch_queue_depth,
          const std::vector<PartialTensorShape> &shapes,
          const DataTypeVector &dtypes) 
          : DatasetBase(DatasetContext(context)), 
          pipeline_(pipeline),
          batch_size_(batch_size),
          num_threads_(num_threads),
          device_id_(device_id),
          exec_separated_(exec_separated),
          prefetch_queue_depth_(prefetch_queue_depth),
          cpu_prefetch_queue_depth_(cpu_prefetch_queue_depth),
          gpu_prefetch_queue_depth_(gpu_prefetch_queue_depth),
          shapes_(shapes), 
          dtypes_(dtypes) {
            daliCreatePipeline(
              &pipeline_handle_,
              pipeline_.c_str(),
              pipeline_.length(),
              batch_size,
              num_threads,
              device_id,
              exec_separated,
              prefetch_queue_depth,
              cpu_prefetch_queue_depth,
              gpu_prefetch_queue_depth);

            if (!exec_separated) {
              daliPrefetchUniform(&pipeline_handle_, prefetch_queue_depth_);
            } else {
              daliPrefetchSeparate(
                &pipeline_handle_, 
                cpu_prefetch_queue_depth,
                prefetch_queue_depth);
            }
          }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(
          const string &prefix) const override {
            return absl::make_unique<Iterator>(
              Iterator::Params{this, strings::StrCat(prefix, "::DALI")}
            );
        }

        const DataTypeVector &output_dtypes() const override {
          return dtypes_;
        }

        const std::vector<PartialTensorShape> &output_shapes() const override {
          return shapes_;
        }

        string DebugString() const override { 
          return "DALI::DatasetOp()::Dataset"; }

        int64 Cardinality() const override { return kInfiniteCardinality; }

        ~Dataset() {
          daliDeletePipeline(&pipeline_handle_);
        }

      protected:
        const std::string pipeline_;
        const int batch_size_;
        const int num_threads_;
        const int device_id_;
        const bool exec_separated_;
        const int prefetch_queue_depth_;
        const int cpu_prefetch_queue_depth_;
        const int gpu_prefetch_queue_depth_;
        const std::vector<PartialTensorShape> shapes_;
        const DataTypeVector &dtypes_;
        
        daliPipelineHandle pipeline_handle_;

        Status AsGraphDefInternal(
          SerializationContext *context,
          DatasetGraphDefBuilder *b,
          Node **output) const override {

          AttrValue pipeline;
          b->BuildAttrValue<std::string>(pipeline_, &pipeline);

          AttrValue batch_size;
          b->BuildAttrValue<int>(batch_size_, &batch_size);

          AttrValue num_threads;
          b->BuildAttrValue<int>(num_threads_, &num_threads);

          AttrValue device_id;
          b->BuildAttrValue<int>(device_id_, &device_id);

          AttrValue exec_separated;
          b->BuildAttrValue<bool>(exec_separated_, &exec_separated);

          AttrValue prefetch_queue_depth;
          b->BuildAttrValue<int>(prefetch_queue_depth_, &prefetch_queue_depth);

          AttrValue cpu_prefetch_queue_depth;
          b->BuildAttrValue<int>(cpu_prefetch_queue_depth_, &cpu_prefetch_queue_depth);

          AttrValue gpu_prefetch_queue_depth;
          b->BuildAttrValue<int>(gpu_prefetch_queue_depth_, &gpu_prefetch_queue_depth);

          AttrValue shapes;
          b->BuildAttrValue<std::vector<PartialTensorShape>>(shapes_, &shapes);

          AttrValue dtypes;
          b->BuildAttrValue<DataTypeVector>(dtypes_, &dtypes);

          TF_RETURN_IF_ERROR(b->AddDataset(
            this, 
            {}, 
            { 
              std::make_pair("pipeline", pipeline),
              std::make_pair("batch_size", batch_size),
              std::make_pair("num_threads", num_threads),
              std::make_pair("device_id" ,device_id),
              std::make_pair("exec_separated", exec_separated),
              std::make_pair("prefetch_queue_depth", prefetch_queue_depth),
              std::make_pair("cpu_prefetch_queue_depth", cpu_prefetch_queue_depth),
              std::make_pair("gpu_prefetch_queue_depth", gpu_prefetch_queue_depth),
              std::make_pair("shapes", shapes),
              std::make_pair("dtypes", dtypes)
            }, 
            output));

          return Status::OK();
        }

      private:
        class Iterator : public DatasetIterator<Dataset> {
          public:
            explicit Iterator(const Params &params)
              : DatasetIterator<Dataset>(params) { }

            Status GetNextInternal(
              IteratorContext *context,
              std::vector<Tensor> *out_tensors,
              bool *end_of_sequence) override {
                tensorflow::mutex_lock l(mu_);
                cudaStream_t stream = 0;
                auto pipeline_handle = dataset()->pipeline_handle_;

                daliShareOutput(&pipeline_handle);
                
                const auto num_outputs = daliGetNumOutput(&pipeline_handle);
                for (int out_id = 0; out_id < num_outputs; ++out_id) {
                  TensorShape output_shape;
                  dataset()->shapes_[out_id].AsTensorShape(&output_shape);
                  out_tensors->emplace_back(context->allocator({}), dataset()->dtypes_[out_id], output_shape);
                  tensorflow::Tensor &output = out_tensors->operator[](out_id);

                  void* dst = nullptr;
                  switch (dataset()->dtypes_[out_id]) {
                    case DT_HALF:
                          dst = reinterpret_cast<void*>(output.flat<uint16_t>().data());
                      break;
                    case DT_FLOAT:
                          dst = reinterpret_cast<void*>(output.flat<float>().data());
                      break;
                    case DT_UINT8:
                          dst = reinterpret_cast<void*>(output.flat<uint8_t>().data());
                      break;
                    case DT_INT16:
                          dst = reinterpret_cast<void*>(output.flat<int16_t>().data());
                      break;
                    case DT_INT32:
                          dst = reinterpret_cast<void*>(output.flat<int32_t>().data());
                      break;
                    case DT_INT64:
                          dst = reinterpret_cast<void*>(output.flat<int64>().data());
                      break;
                    break;
                  }

                  daliCopyTensorNTo(&pipeline_handle, dst, out_id, device_type_t::CPU, stream);
                }
                
                *end_of_sequence = false;

                daliOutputRelease(&pipeline_handle);
                daliRun(&pipeline_handle);
                return Status::OK();
              }

          private:
            tensorflow::mutex mu_;
        };
    };
};


// Regestrations
REGISTER_KERNEL_BUILDER(
  Name("DaliDataset").Device(tensorflow::DEVICE_CPU),
  DALIDatasetOp);

REGISTER_OP("DaliDataset")
  .Attr("pipeline: string")
  .Attr("batch_size: int")
  .Attr("num_threads: int")
  .Attr("device_id: int")
  .Attr("exec_separated: bool")
  .Attr("prefetch_queue_depth: int")
  .Attr("cpu_prefetch_queue_depth: int")
  .Attr("gpu_prefetch_queue_depth: int")
  .Attr("shapes: list(shape) = [{ dim { size: 1 } dim { size: 2 } }, { dim { size: 1 } dim { size: 2 } }]")
  .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
  .Output("handle: variant")
  .SetIsStateful() 
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    std::vector<PartialTensorShape> shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
    c->ExpandOutputs(shapes.size());

    for (unsigned i = 0; i < shapes.size(); ++i) {
      if (shapes[i].dims() > 0) {
        shape_inference::ShapeHandle passed_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[0], &passed_shape));
        TF_RETURN_IF_ERROR(
            c->WithRank(passed_shape, shapes[0].dims(), &passed_shape));
        c->set_output(i, passed_shape);
      }
    }
    return Status::OK();
  });

}  // namespace
}  // namespace data
}  // namespace tensorflow

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12
