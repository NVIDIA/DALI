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
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"


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
    int64 seed;
    OP_REQUIRES_OK(context, ParseScalarArgument<int64>(context, "seed", &seed));
    int64 seed2;
    OP_REQUIRES_OK(context, ParseScalarArgument<int64>(context, "seed2", &seed2));

    if (seed == 0 && seed2 == 0) {
      seed = random::New64();
      seed2 = random::New64();
    }

    *output = new Dataset(context, seed, seed2);
  }

   private:
    class Dataset : public DatasetBase {
      public:
        explicit Dataset(OpKernelContext *context, int64 seed, int64 seed2) 
          : DatasetBase(DatasetContext(context)), seed_(seed), seed2_(seed2) {}

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
          return strings::StrCat("DALI::DatasetOp(", seed_, ", ", seed2_, ")::Dataset"); }

        int64 Cardinality() const override { return kInfiniteCardinality; }
\
      protected:
        Status AsGraphDefInternal(
          SerializationContext *context,
          DatasetGraphDefBuilder *b,
          Node **output) const override {

          Node *seed = nullptr;
          Node *seed2 = nullptr;
          TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
          TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
          TF_RETURN_IF_ERROR(b->AddDataset(this, {seed, seed2}, output));

          return Status::OK();
        }

      private:
        const int64 seed_;
        const int64 seed2_;

        class Iterator : public DatasetIterator<Dataset> {
          public:
            explicit Iterator(const Params &params)
              : DatasetIterator<Dataset>(params),
                parent_generator_(dataset()->seed_, dataset()->seed2_),
                generator_(&parent_generator_) {}

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
            random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
              EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              
              num_random_samples_++;
              auto out = generator_();
              return out;
            }

            tensorflow::mutex mu_;
            random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
            random::SingleSampleAdapter<random::PhiloxRandom> generator_ GUARDED_BY(mu_);
            int64 num_random_samples_ GUARDED_BY(mu_) = 0;
        };
    };
};


// Regestrations
REGISTER_KERNEL_BUILDER(
  Name("DALIDataset").Device(tensorflow::DEVICE_CPU),
  DALIDatasetOp);

REGISTER_OP("DALIDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    // .Attr("output_types: list(type) >= 1")
    // .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

}  // namespace
}  // namespace data
}  // namespace tensorflow

// #endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 12
