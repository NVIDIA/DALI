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

#include "dbg.h"


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
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("pipeline", &pipeline_));

        std::cout << "======= Dataset constructor ==========   " << this << std::endl;
        // std::cout << "pipeline: " << pipeline_ << std::endl;
        std::cout << "shapes: { ";
        for(const auto &shape : shapes_)
          std::cout << shape << ", ";
        std::cout << "}" <<std::endl;
        std::cout << "dtypes: { ";
        for(const auto &t : dtypes_)
          std::cout << t << ", ";
        std::cout << "}" <<std::endl;
        std::cout << std::endl;
        dbg(this);
      }

    void MakeDataset(OpKernelContext* context, DatasetBase** output) override {
      dbg(this);
      *output = new Dataset(context, shapes_, dtypes_, pipeline_);
    }

  private:
    std::vector<PartialTensorShape> shapes_;
    std::string pipeline_;
    DataTypeVector dtypes_;

    class Dataset : public DatasetBase {
      public:
        explicit Dataset(
          OpKernelContext *context, 
          const std::vector<PartialTensorShape> &shapes,
          const DataTypeVector &dtypes,
          const std::string pipeline) 
          : DatasetBase(DatasetContext(context)), shapes_(shapes), dtypes_(dtypes), pipeline_(pipeline) {
            dbg(this);
            daliCreatePipeline(
              &pipeline_handle_,
              pipeline_.c_str(),
              pipeline_.length(),
              8,
              4,
              0,
              false,
              2,
              2,
              2);
            daliPrefetchUniform(&pipeline_handle_, 2);
          }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(
          const string &prefix) const override {
            dbg(this);
            return absl::make_unique<Iterator>(
              Iterator::Params{this, strings::StrCat(prefix, "::DALI")}
            );
        }

        const DataTypeVector &output_dtypes() const override {
          static DataTypeVector* dtypes = new DataTypeVector({DT_FLOAT, DT_INT32});
          return *dtypes;
        }

        const std::vector<PartialTensorShape> &output_shapes() const override {
          return shapes_;
        }

        string DebugString() const override { 
          return "DALI::DatasetOp()::Dataset"; }

        int64 Cardinality() const override { return kInfiniteCardinality; }

        ~Dataset() {
          dbg(this);
          daliDeletePipeline(&pipeline_handle_);
        }

      protected:
        const std::vector<PartialTensorShape> shapes_;
        const std::string pipeline_;
        const DataTypeVector &dtypes_;
        
        daliPipelineHandle pipeline_handle_;


        Status AsGraphDefInternal(
          SerializationContext *context,
          DatasetGraphDefBuilder *b,
          Node **output) const override {

          AttrValue shapes;
          b->BuildAttrValue<std::vector<PartialTensorShape>>(shapes_, &shapes);

          AttrValue dtypes;
          b->BuildAttrValue<DataTypeVector>(dtypes_, &dtypes);

          AttrValue pipeline;
          b->BuildAttrValue<std::string>(pipeline_, &pipeline);

          TF_RETURN_IF_ERROR(b->AddDataset(
            this, 
            {}, 
            { 
              std::make_pair("shapes", shapes),
              std::make_pair("dtypes", dtypes),
              std::make_pair("pipeline", pipeline)
            }, 
            output));

          return Status::OK();
        }

      private:
        class Iterator : public DatasetIterator<Dataset> {
          public:
            explicit Iterator(const Params &params)
              : DatasetIterator<Dataset>(params) 
              {
                dbg(this);
              }

            Status GetNextInternal(
              IteratorContext *context,
              std::vector<Tensor> *out_tensors,
              bool *end_of_sequence) override {
                tensorflow::mutex_lock l(mu_);

                cudaStream_t stream = 0;

                auto pipeline_handle = dataset()->pipeline_handle_;

                dbg(this);
                TF_DALI_CALL(daliShareOutput(&pipeline_handle));

                dbg(daliGetNumOutput(&pipeline_handle));



                TensorShape output_shape;
                dataset()->shapes_[0].AsTensorShape(&output_shape);
                out_tensors->emplace_back(context->allocator({}), DT_FLOAT, output_shape);
                
                tensorflow::Tensor &output = out_tensors->operator[](0);

                for (int i = 0; i < output.NumElements(); ++i) {
                  output.flat<float>()(i) = 1;
                }
                
                dataset()->shapes_[1].AsTensorShape(&output_shape);
                out_tensors->emplace_back(context->allocator({}), DT_INT32, output_shape);


                dbg(daliTensorSize(&pipeline_handle, 1));
                tensorflow::Tensor &label_output = out_tensors->operator[](1);
                auto dst = reinterpret_cast<void*>(label_output.flat<int>().data());
                TF_DALI_CALL(daliCopyTensorNTo(&pipeline_handle, dst, 1, device_type_t::CPU, stream));
                
                
                *end_of_sequence = false;

                TF_DALI_CALL(daliOutputRelease(&pipeline_handle));
                TF_DALI_CALL(daliRun(&pipeline_handle));
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
    .Attr("shapes: list(shape) = [{ dim { size: 1 } dim { size: 2 } }, { dim { size: 1 } dim { size: 2 } }]")
    .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
    .Attr("pipeline: string")
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
