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

// #if TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12

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
#include "tf_helper.h"

#define TF_DALI_CALL(FUNC)                                                     \
    _DALI_CALL_IMPL(FUNC, _RET_ERROR)

namespace tensorflow {

namespace data {

namespace {

class DALIDatasetOp : public DatasetOpKernel {
  public:
    explicit DALIDatasetOp(OpKernelConstruction* context) 
      : DatasetOpKernel(context) { 
        OP_REQUIRES_OK(context, context->GetAttr("value", &value_));
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));

        std::cout << "======= Dataset constructor ==========\n ";
        std::cout << "value: " << value_ << std::endl;
        std::cout << "shapes: { ";
        for(const auto &shape : shapes_)
          std::cout << shape << ", ";
        std::cout << "}" <<std::endl;
        std::cout << std::endl;

      }

    void MakeDataset(OpKernelContext* context, DatasetBase** output) override {
      *output = new Dataset(context, value_, shapes_);
    }

  private:
    int64 value_;
    std::vector<PartialTensorShape> shapes_;

    class Dataset : public DatasetBase {
      public:
        explicit Dataset(
          OpKernelContext *context, 
          const int64 value,
          const std::vector<PartialTensorShape> &shapes) 
          : DatasetBase(DatasetContext(context)), value_(value), shapes_(shapes) {}

        std::unique_ptr<IteratorBase> MakeIteratorInternal(
          const string &prefix) const override {
            return absl::make_unique<Iterator>(
              Iterator::Params{this, strings::StrCat(prefix, "::DALI")}
            );
        }

        const DataTypeVector &output_dtypes() const override {
          static DataTypeVector* dtypes = new DataTypeVector({DT_FLOAT, DT_INT64});
          return *dtypes;
        }

        const std::vector<PartialTensorShape> &output_shapes() const override {
          return shapes_;
        }

        string DebugString() const override { 
          return "DALI::DatasetOp()::Dataset"; }

        int64 Cardinality() const override { return kInfiniteCardinality; }

      protected:
        const int64 value_;
        const std::vector<PartialTensorShape> shapes_;

        Status AsGraphDefInternal(
          SerializationContext *context,
          DatasetGraphDefBuilder *b,
          Node **output) const override {

          AttrValue value;
          b->BuildAttrValue<int>(value_, &value);

          AttrValue shapes;
          b->BuildAttrValue<std::vector<PartialTensorShape>>(shapes_, &shapes);

          TF_RETURN_IF_ERROR(b->AddDataset(
            this, 
            {}, 
            { 
              std::make_pair("value", value),
              std::make_pair("shapes", shapes),
            }, 
            output));

          return Status::OK();
        }

      private:
        class Iterator : public DatasetIterator<Dataset> {
          public:
            explicit Iterator(const Params &params)
              : DatasetIterator<Dataset>(params) {}

            Status GetNextInternal(
              IteratorContext *context,
              std::vector<Tensor> *out_tensors,
              bool *end_of_sequence) override {
                tensorflow::mutex_lock l(mu_);
                TensorShape output_shape;
                dataset()->shapes_[0].AsTensorShape(&output_shape);
                out_tensors->emplace_back(context->allocator({}), DT_FLOAT, output_shape);
                
                tensorflow::Tensor &output = out_tensors->operator[](0);

                for (int i = 0; i < output.NumElements(); ++i) {
                  output.flat<float>()(i) = dataset()->value_;
                }
                
                dataset()->shapes_[1].AsTensorShape(&output_shape);
                out_tensors->emplace_back(context->allocator({}), DT_INT64, output_shape);
                
                
                *end_of_sequence = false;

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
    .Attr("value: int")
    .Attr("shapes: list(shape) = [{ dim { size: 1 } dim { size: 2 } }, { dim { size: 1 } dim { size: 2 } }]")
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

// #endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12
