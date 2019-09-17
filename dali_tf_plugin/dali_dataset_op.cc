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
  explicit DALIDatasetOp(OpKernelConstruction* context) : DatasetOpKernel(context) { }

  void MakeDataset(OpKernelContext* context, DatasetBase** output) override {
    *output = new Dataset(context);
  }

   private:
    class Dataset : public DatasetBase {
      public:
        explicit Dataset(OpKernelContext *context) 
          : DatasetBase(DatasetContext(context)) {}

        std::unique_ptr<IteratorBase> MakeIteratorInternal(
          const string &prefix) const override {
            return absl::make_unique<Iterator>(
              Iterator::Params{this, strings::StrCat(prefix, "::DALI")}
            );
        }

        const DataTypeVector &output_dtypes() const override {
          static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
          return *dtypes;
        }

        const std::vector<PartialTensorShape> &output_shapes() const override {
          static std::vector<PartialTensorShape> *shapes = 
            new std::vector<PartialTensorShape>({{}});
          return *shapes;
        }

        string DebugString() const override { 
          return "DALI::DatasetOp()::Dataset"; }

        int64 Cardinality() const override { return kInfiniteCardinality; }
\
      protected:
        Status AsGraphDefInternal(
          SerializationContext *context,
          DatasetGraphDefBuilder *b,
          Node **output) const override {

          TF_RETURN_IF_ERROR(b->AddDataset(this, {}, output));

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
                out_tensors->emplace_back(context->allocator({}), DT_INT64, TensorShape({}));
                out_tensors->back().scalar<int64>()() = 999;
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
  Name("DALIDataset").Device(tensorflow::DEVICE_CPU),
  DALIDatasetOp);

REGISTER_OP("DALIDataset")
    .Output("handle: variant")
    .SetIsStateful() 
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      return shape_inference::ScalarShape(c);
    });

}  // namespace
}  // namespace data
}  // namespace tensorflow

// #endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12
