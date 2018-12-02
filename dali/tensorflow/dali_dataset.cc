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


namespace tensorflow {

namespace data {

namespace {

class DALIDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    *output =
        new Dataset(ctx);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx)
        : DatasetBase(DatasetContext(ctx)) {
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::DaliDataset")}));
    }

    const DataTypeVector& output_dtypes() const override {
      // TODO set the correct types
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      // TODO set the correct shapes
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "DALIDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return Status::OK();
      }

     private:
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      }

      mutex mu_;
    };
  };
};

REGISTER_OP("DALIDataset")
    .Attr("serialized_pipeline: string")
    .Attr("shapes: list(shape) >= 1")
    .Attr("dtypes: list({float, int32, int64, half}) >= 1")
    .Attr("devices: list(type) >= 1")
    .Output("handle: variant")
    .

    SetIsStateful()  // TODO: Is it necessary in order to prevent folding?
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      // define shape for the first output
      std::vector<tensorflow::PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (unsigned i = 0; i < shapes.

                               size();

           ++i) {
        if (shapes[i].

            dims()

            > 0) {
          tensorflow::shape_inference::ShapeHandle passed_shape;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
          TF_RETURN_IF_ERROR(c->WithRank(passed_shape,
                                         shapes[i].

                                         dims(),
                                         &passed_shape

                                         ));
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

REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_GPU),
                        DALIDatasetOp)

}  // namespace
}  // namespace data
}  // namespace tensorflow

#endif