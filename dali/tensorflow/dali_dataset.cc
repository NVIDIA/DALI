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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

namespace tf = tensorflow;

class DaliDatasetOp : public tf::DatasetOpKernel {
public:
  DaliDatasetOp(tf::OpKernelConstruction* context)
  : tf::DatasetOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("serialized_pipeline", &serialized_pipeline_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("devices", &devices_));

    OP_REQUIRES(
        context, shapes_.size() == 2,
        tf::errors::InvalidArgument("`shapes` must be a a list or tuple with 2 elements."));
    OP_REQUIRES(
        context, dtypes_.size() == 2,
        tf::errors::InvalidArgument("`dtypes` must be a a list or tuple with 2 elements."));
    OP_REQUIRES(
        context, !devices_.empty(),
        tf::errors::InvalidArgument("`devices` cannot be empty."));
  }

  void MakeDataset(tf::OpKernelContext* ctx, tf::DatasetBase** output) override {
  }

private:
  class Dataset : public tf::DatasetBase {
  public:
    explicit Dataset(tf::OpKernelContext* ctx, std::vector<std::string> filenames,
                     const string& compression_type, int64 buffer_size)
        : DatasetBase(tf::DatasetContext(ctx)) {
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, tf::strings::StrCat(prefix, "::DALIDataset")}));
    }

    const DataTypeVector& output_dtypes() const override {
      // Todo: Assign correct types
      static tf::DataTypeVector* dtypes = new tf::DataTypeVector({tf::DT_STRING});
      return *dtypes;
    }

    const std::vector<tf::PartialTensorShape>& output_shapes() const override {
      // TODO assign correct shapes
      static std::vector<tf::PartialTensorShape>* shapes =
          new std::vector<tf::PartialTensorShape>({{}});
      return *shapes;
    }

    std::string DebugString() const override { return "DALIDataset::Dataset"; }

  protected:
    Status AsGraphDefInternal(tf::SerializationContext* ctx,
                              tf::DatasetGraphDefBuilder* b,
                              tf::Node** output) const override {
      return tf::Status::OK();
    }

  private:
    class Iterator : public tf::DatasetIterator<Dataset> {
    public:
      explicit Iterator(const tf::Params& params)
          : tf::DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(tf::IteratorContext* ctx,
                             std::vector<tf::Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        return tf::Status::OK();
      }

    protected:
      Status SaveInternal(tf::IteratorStateWriter* writer) override {
        return tf::Status::OK();
      }

      Status RestoreInternal(tf::IteratorContext* ctx,
                             tf::IteratorStateReader* reader) override {
        return tf::Status::OK();
      }

    private:
      Status SetupStreamsLocked(tf::Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          return tf::Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      }
    };
  };

  std::string serialized_pipeline_
  std::vector<tf::TensorShape> shapes_;
  tf::DataTypeVector dtypes_;
  std::vector<int> devices_;
};

REGISTER_OP("DALIDataset")
    .Attr("serialized_pipeline: string")
    .Attr("shapes: list(shape) >= 1")
    .Attr("dtypes: list({float, int32, int64, half}) >= 1")
    .Attr("devices: list(type) >= 1")
    .Output("handle: variant")
    .SetIsStateful() // TODO: Is it necessary in order to prevent folding?
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      // define shape for the first output
      std::vector<tensorflow::PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (unsigned i = 0; i < shapes.size(); ++i) {
        if (shapes[i].dims() > 0) {
          tensorflow::shape_inference::ShapeHandle passed_shape;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
          TF_RETURN_IF_ERROR(
              c->WithRank(passed_shape, shapes[i].dims(), &passed_shape));
          c->set_output(i, passed_shape);
        }
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
DALI Dataset plugin

Creates a Dali dataset compatible with tf.data.Dataset from a serialized DALI pipeline
(given in `serialized_pipeline` parameter).

`shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
 )doc");

REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_GPU),
                        DaliDatasetOp)
