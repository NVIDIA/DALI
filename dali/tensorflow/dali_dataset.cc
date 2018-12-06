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

#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION == 12

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

#include "dali/c_api/c_api.h"

namespace tensorflow {

namespace data {

namespace {

// TODO extract this guy. By now C&P from daliop
TensorShape DaliToShape(int64_t* ns) {
  TensorShape ts;
  for (int i = 0; ns[i] != 0; ++i)
    ts.InsertDim(i, ns[i]);
  delete[] ns;
  return ts;
}

class DALIDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  DALIDatasetOp(OpKernelConstruction* context)
  : DatasetOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("serialized_pipeline", &serialized_pipeline_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("prefetch_queue_depth", &prefetch_queue_depth_));
      OP_REQUIRES_OK(context,
                   context->GetAttr("num_threads", &num_threads_));

    std::vector<int> devices;
    OP_REQUIRES_OK(context,
                   context->GetAttr("devices", &devices));

    pipe_handles_.reserve(devices.size());

    for (size_t  i = 0; i < devices.size(); i++) {
      pipe_handles_.emplace_back(daliPipelineHandle());

      daliCreatePipeline(&pipe_handles_.back(),
                  serialized_pipeline_.c_str(),
                  serialized_pipeline_.length(),
                  128, // TODO: get batch size from shapes
                  num_threads_,
                  devices[i],
                  prefetch_queue_depth_);

      for (int j = 0; j < prefetch_queue_depth_; j++) {
        for (auto& handle : pipe_handles_) {
          daliRun(&handle);
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
    *output =
        new Dataset(context, &pipe_handles_, &shapes_, &dtypes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const std::vector<daliPipelineHandle>* pipe_handles,
        const std::vector<TensorShape>* shapes, const DataTypeVector* types)
        : DatasetBase(DatasetContext(ctx)),
          pipe_handles_{pipe_handles},
          // TODO: Temporary partial shape that will be generalized
          shapes_{std::vector<PartialTensorShape>(shapes->size(), PartialTensorShape({-1,-1}))},
          dtypes_{types} {
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::DALIDataset")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return *dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() const override { return "DALIDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return Status::OK();
    }

  protected:
    const std::vector<daliPipelineHandle>* pipe_handles_;
    const std::vector<PartialTensorShape> shapes_;
    const DataTypeVector* dtypes_;

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
              current_idx_{0},
              num_pipelines_{dataset()->pipe_handles_->size()} {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
          mutex_lock l(mu_);

          if (current_idx_ < num_pipelines_) {
            auto current_handle = dataset()->pipe_handles_->at(current_idx_);

            daliShareOutput(&current_handle);

            const auto output_count = daliGetNumOutput(&current_handle);

            for (size_t i = 0; i < output_count; i++) {
              auto shape = DaliToShape(daliShapeAt(&current_handle, i));

              Tensor out(ctx->allocator({}),
                  dataset()->dtypes_->at(i), shape);

              void *dst = nullptr;
              switch (dataset()->dtypes_->at(i)) {
                case DT_FLOAT:
                      dst = reinterpret_cast<void*>(out.flat<float>().data());
                  break;
                case DT_INT32:
                      dst = reinterpret_cast<void*>(out.flat<int>().data());
                  break;
                case DT_INT64:
                      dst = reinterpret_cast<void*>(out.flat<int64>().data());
                  break;
                case DT_HALF:
                      dst = reinterpret_cast<void*>(out.flat<uint16_t>().data());
                  break;
                default:
                  // todo assert here to catch the failure, because types are not yet converted ?
                  // todo support the new types ?
                  break;
              }

              daliCopyTensorNTo(&current_handle, dst, i);
              out_tensors->emplace_back(out);

              daliOutputRelease(&current_handle);
              daliRun(&current_handle);
            }

            ++current_idx_;
            *end_of_sequence = false;
          } else {
            current_idx_ = 0;
            *end_of_sequence = true;
          }

        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("current_idx"), current_idx_));

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        int64 current_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_index"),
                                              &current_index));
        current_idx_ = static_cast<unsigned int>(current_index);

        return Status::OK();
      }

      mutex mu_;
      unsigned int current_idx_;
      const size_t num_pipelines_;
    };
  };

protected:
  std::vector<daliPipelineHandle> pipe_handles_;
  std::string serialized_pipeline_;
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  int prefetch_queue_depth_;
  int num_threads_;
};

REGISTER_OP("DALIDataset")
    .Attr("serialized_pipeline: string")
    .Attr("shapes: list(shape) >= 1")
    .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
    .Attr("devices: list(int) >= 1")
    .Attr("prefetch_queue_depth: int = 2")
    .Attr("num_threads: int = -1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      std::vector<tensorflow::PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (unsigned int i = 0; i < shapes.size(); ++i) {
        if (shapes[i].dims() > 0) {
          tensorflow::shape_inference::ShapeHandle passed_shape;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
          TF_RETURN_IF_ERROR(c->WithRank(passed_shape,
                                         shapes[i].dims(),
                                         &passed_shape));
          c->set_output(i, passed_shape);
        }
      }
      return

          tensorflow::Status::OK();
    })
    .Doc(R"doc(
DALI Dataset plugin

Creates a Dali dataset compatible with tf.data.Dataset from a serialized DALI pipeline
(given in `serialized_pipeline` parameter).

`shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
 )doc");

REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_CPU),
                        DALIDatasetOp)

}  // namespace
}  // namespace data
}  // namespace tensorflow

#endif