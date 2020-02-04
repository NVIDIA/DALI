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

#if TF_MAJOR_VERSION == 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#define EIGEN_USE_GPU

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

#include "dali/core/common.h"
#include "dali/c_api/c_api.h"

#define DALI_CALL(FUNC)                                                    \
do {                                                                       \
  try {                                                                    \
    FUNC;                                                                  \
  } catch (std::exception &e) {                                            \
    std::string error = "DALI " + std::string(#FUNC)                       \
                        + " failed: " + std::string(e.what());             \
    std::cout << error << std::endl;                                       \
    return ::tensorflow::errors::Internal(error);                          \
  }                                                                        \
} while (0)

using namespace tensorflow;

namespace {

class DALIDatasetOp : public DatasetOpKernel {
 public:
  explicit DALIDatasetOp(OpKernelConstruction *context)
      : DatasetOpKernel(context), is_gpu_device_(context->device_type() == "GPU") {
    FillPipelineDef(context, pipeline_def_);
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
  }

    void MakeDataset(OpKernelContext *context, DatasetBase **output) override {
      *output = new Dataset(context, pipeline_def_, shapes_, dtypes_, is_gpu_device_);
    }

 private:
  struct PipelineDef {
    std::string pipeline;
    int batch_size;
    int num_threads;
    int device_id;
    bool exec_separated;
    int prefetch_queue_depth;
    int cpu_prefetch_queue_depth;
    int gpu_prefetch_queue_depth;
  };

  static constexpr const char* const kPipeline = "pipeline";
  static constexpr const char* const kBatchSize = "batch_size";
  static constexpr const char* const kNumThreads = "num_threads";
  static constexpr const char* const kDeviceId = "device_id";
  static constexpr const char* const kExecSeparated = "exec_separated";
  static constexpr const char* const kPrefetchQueueDepth = "prefetch_queue_depth";
  static constexpr const char* const kCpuPrefetchQueueDepth = "cpu_prefetch_queue_depth";
  static constexpr const char* const kGpuPrefetchQueueDepth = "gpu_prefetch_queue_depth";

  void FillPipelineDef(OpKernelConstruction* context, PipelineDef &def) {
    OP_REQUIRES_OK(context, context->GetAttr(kPipeline, &def.pipeline));
    OP_REQUIRES_OK(context, context->GetAttr(kBatchSize, &def.batch_size));
    OP_REQUIRES_OK(context, context->GetAttr(kNumThreads, &def.num_threads));
    OP_REQUIRES_OK(context, context->GetAttr(kDeviceId, &def.device_id));
    OP_REQUIRES_OK(context, context->GetAttr(kExecSeparated, &def.exec_separated));
    OP_REQUIRES_OK(context, context->GetAttr(kPrefetchQueueDepth, &def.prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr(kCpuPrefetchQueueDepth, &def.cpu_prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr(kGpuPrefetchQueueDepth, &def.gpu_prefetch_queue_depth));
  }

  PipelineDef pipeline_def_;
  std::vector<PartialTensorShape> shapes_;
  DataTypeVector dtypes_;
  bool is_gpu_device_;

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(
      OpKernelContext *context,
      const PipelineDef pipeline_def,
      const std::vector<PartialTensorShape> &shapes,
      const DataTypeVector &dtypes,
      const bool is_gpu_device)
        : DatasetBase(DatasetContext(context)),
        pipeline_def_(pipeline_def),
        shapes_(shapes),
        dtypes_(dtypes),
        device_type_(is_gpu_device ? device_type_t::GPU : device_type_t::CPU) {

      if (is_gpu_device) {
        stream_ = context->eigen_gpu_device().stream();
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      daliPipelineHandle pipeline_handle;
      TF_CHECK_OK(InitPipeline(&pipeline_handle));
      return absl::make_unique<Iterator>(Iterator::Params{this, strings::StrCat(prefix, "::DALI")},
                                         pipeline_handle);
    }

    const DataTypeVector &output_dtypes() const override {
      return dtypes_;
    }

    const std::vector<PartialTensorShape> &output_shapes() const override {
      return shapes_;
    }

    string DebugString() const override {
      return "DALI::DatasetOp()::Dataset"; }

    tensorflow::int64 Cardinality() const override { return data::kInfiniteCardinality; }

   protected:
    PipelineDef pipeline_def_;
    const std::vector<PartialTensorShape> shapes_;
    const DataTypeVector dtypes_;
    cudaStream_t stream_ = 0;
    const device_type_t device_type_;

    daliPipelineHandle pipeline_handle_;

    Status AsGraphDefInternal(SerializationContext *context, DatasetGraphDefBuilder *b,
                              Node **output) const override {

      auto attrs = PipelineDefToGraphDefAttrs(b, pipeline_def_);

      SerializeField(attrs, b, "shapes", shapes_);
      SerializeField(attrs, b, "dtypes", dtypes_);

      TF_RETURN_IF_ERROR(b->AddDataset(this, {}, attrs, output));

      return Status::OK();
    }

   private:
   /**
    * @brief Append the AttrValue created from `filed` under `name` to `attrs` vector
    */
    template <typename T>
    void SerializeField(std::vector<std::pair<StringPiece, AttrValue>> &attrs,
                        DatasetGraphDefBuilder *b, const StringPiece &name, const T &field) const {
      AttrValue field_attr;
      b->BuildAttrValue(field, &field_attr);
      attrs.push_back(std::make_pair(name, field_attr));
    }

    std::vector<std::pair<StringPiece, AttrValue>> PipelineDefToGraphDefAttrs(
        DatasetGraphDefBuilder *b, const PipelineDef &def) const {
      std::vector<std::pair<StringPiece, AttrValue>> attrs;

      SerializeField(attrs, b, kPipeline, pipeline_def_.pipeline);
      SerializeField(attrs, b, kBatchSize, pipeline_def_.batch_size);
      SerializeField(attrs, b, kNumThreads, pipeline_def_.num_threads);
      SerializeField(attrs, b, kDeviceId, pipeline_def_.device_id);
      SerializeField(attrs, b, kExecSeparated, pipeline_def_.exec_separated);
      SerializeField(attrs, b, kPrefetchQueueDepth, pipeline_def_.prefetch_queue_depth);
      SerializeField(attrs, b, kCpuPrefetchQueueDepth, pipeline_def_.cpu_prefetch_queue_depth);
      SerializeField(attrs, b, kGpuPrefetchQueueDepth, pipeline_def_.gpu_prefetch_queue_depth);

      return attrs;
    }

    Status InitPipeline(daliPipelineHandle *pipeline_handle) const {
      DALI_CALL(daliCreatePipeline(
        pipeline_handle,
        pipeline_def_.pipeline.c_str(),
        pipeline_def_.pipeline.length(),
        pipeline_def_.batch_size,
        pipeline_def_.num_threads,
        pipeline_def_.device_id,
        pipeline_def_.exec_separated,
        pipeline_def_.prefetch_queue_depth,
        pipeline_def_.cpu_prefetch_queue_depth,
        pipeline_def_.gpu_prefetch_queue_depth));

      if (!pipeline_def_.exec_separated) {
        DALI_CALL(daliPrefetchUniform(pipeline_handle, pipeline_def_.prefetch_queue_depth));
      } else {
        DALI_CALL(daliPrefetchSeparate(pipeline_handle, pipeline_def_.cpu_prefetch_queue_depth,
                                       pipeline_def_.gpu_prefetch_queue_depth));
      }
      return Status::OK();
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params, daliPipelineHandle pipeline_handle)
          : DatasetIterator<Dataset>(params), pipeline_handle_(pipeline_handle) {}

      Status GetNextInternal(IteratorContext *context, std::vector<Tensor> *out_tensors,
                             bool *end_of_sequence) override {
        tensorflow::mutex_lock l(mu_);

        DALI_CALL(daliShareOutput(&pipeline_handle_));

        auto num_outputs = 0;
        DALI_CALL(num_outputs = daliGetNumOutput(&pipeline_handle_));

        for (int out_id = 0; out_id < num_outputs; ++out_id) {
          TensorShape output_shape;
          dataset()->shapes_[out_id].AsTensorShape(&output_shape);
          out_tensors->emplace_back(context->allocator({}), dataset()->dtypes_[out_id],
                                    output_shape);
          tensorflow::Tensor &output = out_tensors->operator[](out_id);

          void *dst = nullptr;
          switch (dataset()->dtypes_[out_id]) {
            case DT_HALF:
              dst = reinterpret_cast<void *>(output.flat<Eigen::half>().data());
              break;
            case DT_FLOAT:
              dst = reinterpret_cast<void *>(output.flat<float>().data());
              break;
            case DT_UINT8:
              dst = reinterpret_cast<void *>(output.flat<uint8_t>().data());
              break;
            case DT_INT16:
              dst = reinterpret_cast<void *>(output.flat<int16_t>().data());
              break;
            case DT_INT32:
              dst = reinterpret_cast<void *>(output.flat<int32_t>().data());
              break;
            case DT_INT64:
              dst = reinterpret_cast<void *>(output.flat<int64>().data());
              break;
            default:
              return errors::InvalidArgument(
                  "Unsupported type: " + DataTypeString(dataset()->dtypes_[out_id]) +
                  "for tensor " + std::to_string(out_id));
          }

          DALI_CALL(daliCopyTensorNTo(&pipeline_handle_, dst, out_id, dataset()->device_type_,
                                      dataset()->stream_, false));
        }

        *end_of_sequence = false;

        DALI_CALL(daliOutputRelease(&pipeline_handle_));
        DALI_CALL(daliRun(&pipeline_handle_));

        return Status::OK();
      }

      ~Iterator() {
        daliDeletePipeline(&pipeline_handle_);
      }

     private:
      tensorflow::mutex mu_;
      daliPipelineHandle pipeline_handle_;
    };  //Iterator
  };   //Dataset
};


// Regestrations
REGISTER_KERNEL_BUILDER(
  Name("DALIDataset").Device(tensorflow::DEVICE_CPU),
  DALIDatasetOp);

REGISTER_KERNEL_BUILDER(
  Name("DALIDataset")
    .Device(DEVICE_GPU)
    .HostMemory("handle"),
  DALIDatasetOp);

REGISTER_OP("DALIDataset")
  .Attr("pipeline: string")
  .Attr("batch_size: int")
  .Attr("num_threads: int")
  .Attr("device_id: int")
  .Attr("exec_separated: bool")
  .Attr("prefetch_queue_depth: int")
  .Attr("cpu_prefetch_queue_depth: int")
  .Attr("gpu_prefetch_queue_depth: int")
  .Attr("shapes: list(shape) >= 1")
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
  })
  .Doc(R"doc(
DALI Dataset plugin
Creates a DALI dataset compatible with tf.data.Dataset from a DALI pipeline.
`shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
)doc");

}  // namespace

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15
