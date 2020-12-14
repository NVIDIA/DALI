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
#include <sstream>

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
#include "dali/core/format.h"
#include "dali/c_api.h"
#include "dali_shape_helper.h"

#define TF_DALI_CALL(FUNC)                                                    \
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
using namespace tensorflow::data;

namespace dali_tf_impl {

class DALIDatasetOp : public DatasetOpKernel {
 public:
  explicit DALIDatasetOp(OpKernelConstruction *context)
      : DatasetOpKernel(context), is_gpu_device_(context->device_type() == "GPU"),
        context_(context) {
    FillPipelineDef(context, pipeline_def_);
    OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("output_dtypes", &dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr("fail_on_device_mismatch", &fail_on_device_mismatch_));
  }

    void MakeDataset(OpKernelContext *context, DatasetBase **output) override {
      *output = new Dataset(
        context, pipeline_def_, shapes_, dtypes_, is_gpu_device_, fail_on_device_mismatch_);
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
    bool enable_memory_stats;
  };

  static constexpr const char* const kPipeline = "pipeline";
  static constexpr const char* const kBatchSize = "batch_size";
  static constexpr const char* const kNumThreads = "num_threads";
  static constexpr const char* const kDeviceId = "device_id";
  static constexpr const char* const kExecSeparated = "exec_separated";
  static constexpr const char* const kPrefetchQueueDepth = "prefetch_queue_depth";
  static constexpr const char* const kCpuPrefetchQueueDepth = "cpu_prefetch_queue_depth";
  static constexpr const char* const kGpuPrefetchQueueDepth = "gpu_prefetch_queue_depth";
  static constexpr const char* const kGpuMemoryStats = "enable_memory_stats";

  void FillPipelineDef(OpKernelConstruction* context, PipelineDef &def) {
    OP_REQUIRES_OK(context, context->GetAttr(kPipeline, &def.pipeline));
    OP_REQUIRES_OK(context, context->GetAttr(kBatchSize, &def.batch_size));
    OP_REQUIRES_OK(context, context->GetAttr(kNumThreads, &def.num_threads));
    OP_REQUIRES_OK(context, context->GetAttr(kDeviceId, &def.device_id));
    OP_REQUIRES_OK(context, context->GetAttr(kExecSeparated, &def.exec_separated));
    OP_REQUIRES_OK(context, context->GetAttr(kPrefetchQueueDepth, &def.prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr(kCpuPrefetchQueueDepth, &def.cpu_prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr(kGpuPrefetchQueueDepth, &def.gpu_prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr(kGpuMemoryStats, &def.enable_memory_stats));
  }

  PipelineDef pipeline_def_;
  std::vector<PartialTensorShape> shapes_;
  DataTypeVector dtypes_;
  bool is_gpu_device_;
  bool fail_on_device_mismatch_;
  OpKernelConstruction *context_;

  class Dataset : public DatasetBase {
   public:
    explicit Dataset(
      OpKernelContext *context,
      const PipelineDef pipeline_def,
      const std::vector<PartialTensorShape> &shapes,
      const DataTypeVector &dtypes,
      const bool is_gpu_device,
      const bool fail_on_device_mismatch)
        : DatasetBase(DatasetContext(context)),
        pipeline_def_(pipeline_def),
        shapes_(shapes),
        dtypes_(dtypes),
        device_type_(is_gpu_device ? device_type_t::GPU : device_type_t::CPU),
        fail_on_device_mismatch_(fail_on_device_mismatch) {
      if (is_gpu_device) {
        stream_ = context->eigen_gpu_device().stream();
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      daliPipelineHandle pipeline_handle;
      TF_CHECK_OK(InitPipeline(&pipeline_handle));     
      
      return absl::make_unique<Iterator>(Iterator::Params{this, strings::StrCat(prefix, "::DALI")},
                                         pipeline_handle, pipeline_def_.enable_memory_stats);
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
    std::vector<PartialTensorShape> shapes_;
    const DataTypeVector dtypes_;
    cudaStream_t stream_ = 0;
    const device_type_t device_type_;
    const bool fail_on_device_mismatch_;

    daliPipelineHandle pipeline_handle_;

    Status AsGraphDefInternal(SerializationContext *context, DatasetGraphDefBuilder *b,
                              Node **output) const override {

      auto attrs = PipelineDefToGraphDefAttrs(b, pipeline_def_);

      SerializeField(attrs, b, "output_shapes", shapes_);
      SerializeField(attrs, b, "output_dtypes", dtypes_);
      SerializeField(attrs, b, "fail_on_device_mismatch", fail_on_device_mismatch_);

      TF_RETURN_IF_ERROR(b->AddDataset(this, {}, attrs, output));

      return Status::OK();
    }

#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3
  Status CheckExternalState() const override {
    return errors::Unimplemented("CheckExternalState is not supported for DALI dataset.");
  }
#endif

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
      SerializeField(attrs, b, kGpuMemoryStats, pipeline_def_.enable_memory_stats);

      return attrs;
    }

    Status InitPipeline(daliPipelineHandle *pipeline_handle) const {
      TF_DALI_CALL(daliCreatePipeline(
        pipeline_handle,
        pipeline_def_.pipeline.c_str(),
        pipeline_def_.pipeline.length(),
        pipeline_def_.batch_size,
        pipeline_def_.num_threads,
        pipeline_def_.device_id,
        pipeline_def_.exec_separated,
        pipeline_def_.prefetch_queue_depth,
        pipeline_def_.cpu_prefetch_queue_depth,
        pipeline_def_.gpu_prefetch_queue_depth,
        pipeline_def_.enable_memory_stats));

      if (!pipeline_def_.exec_separated) {
        TF_DALI_CALL(daliPrefetchUniform(pipeline_handle, pipeline_def_.prefetch_queue_depth));
      } else {
        TF_DALI_CALL(daliPrefetchSeparate(pipeline_handle, pipeline_def_.cpu_prefetch_queue_depth,
                                       pipeline_def_.gpu_prefetch_queue_depth));
      }
      return Status::OK();
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params, daliPipelineHandle pipeline_handle,
                        bool enable_memory_stats = false)
          : DatasetIterator<Dataset>(params), pipeline_handle_(pipeline_handle),
            enable_memory_stats_(enable_memory_stats) {}

      Status Initialize(IteratorContext* context) override {
        return CheckOutputDevices(); 
      }

      Status CheckOutputDevices() {
        auto num_outputs = daliGetNumOutput(&pipeline_handle_);
        for (auto i  = 0; i < num_outputs; ++i) {
          auto dali_device_type = daliGetOutputDevice(&pipeline_handle_, i);

          if (dali_device_type != dataset()->device_type_) {
            auto msg = dali::make_string(
              "TF device and DALI device mismatch. TF device: ",
              (dataset()->device_type_ == device_type_t::CPU ? "CPU" : "GPU"),
              ", DALI device: ",
              (dali_device_type == device_type_t::CPU ? "CPU" : "GPU"),
              " for output ",
              i);
            
            if (dataset()->fail_on_device_mismatch_) {
              return Status(
                tensorflow::error::Code::INTERNAL,
                msg);
            }
            LOG(WARNING) << "DALI LOG: CheckOutputDevice: " << msg;
          }
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext *context, std::vector<Tensor> *out_tensors,
                             bool *end_of_sequence) override {
        tensorflow::mutex_lock l(mu_);

        TF_DALI_CALL(daliShareOutput(&pipeline_handle_));

        auto num_outputs = 0;
        TF_DALI_CALL(num_outputs = daliGetNumOutput(&pipeline_handle_));

        for (int out_id = 0; out_id < num_outputs; ++out_id) {
          TensorShape output_shape;

          auto dali_shape = DaliToShape(AutoCPtr<int64_t>(daliShapeAt(&pipeline_handle_, out_id)));
          auto status = GetCompatibleShape(output_shape, dataset()->shapes_[out_id],
              dali_shape, dataset()->pipeline_def_.batch_size, out_id);
          if (status != Status::OK()) {
            return status;
          }

          auto dali_type = daliTypeAt(&pipeline_handle_, out_id);
          auto tf_type = DaliToTfType(dali_type);

          if (tf_type != dataset()->dtypes_[out_id]) {
            std::stringstream ss;
            ss << "The type provided for output `" << out_id << "` is not compatible with "
                << "the type returned by DALI Pipeline. Expected (output_types[" << out_id
                << "]): " << DataTypeString(dataset()->dtypes_[out_id]) << ", got from Pipeline: "
                << DataTypeString(tf_type) << ".";
            return errors::InvalidArgument(ss.str());
          }

          out_tensors->emplace_back(context->allocator({}), dataset()->dtypes_[out_id],
                                    output_shape);
          tensorflow::Tensor &output = out_tensors->operator[](out_id);

          void *dst = nullptr;
          switch (dataset()->dtypes_[out_id]) {
            case DT_BOOL:
              dst = reinterpret_cast<void *>(output.flat<bool>().data());
              break;
            case DT_HALF:
              dst = reinterpret_cast<void *>(output.flat<Eigen::half>().data());
              break;
            case DT_FLOAT:
              dst = reinterpret_cast<void *>(output.flat<float>().data());
              break;
            case DT_DOUBLE:
              dst = reinterpret_cast<void *>(output.flat<double>().data());
              break;
            case DT_UINT8:
              dst = reinterpret_cast<void *>(output.flat<uint8_t>().data());
              break;
            case DT_UINT16:
              dst = reinterpret_cast<void *>(output.flat<uint16_t>().data());
              break;
            case DT_UINT32:
              dst = reinterpret_cast<void *>(output.flat<uint32_t>().data());
              break;
            case DT_UINT64:
              dst = reinterpret_cast<void *>(output.flat<uint64>().data());
              break;
            case DT_INT8:
              dst = reinterpret_cast<void *>(output.flat<int8_t>().data());
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
                  " for tensor " + std::to_string(out_id));
          }

          TF_DALI_CALL(daliOutputCopy(&pipeline_handle_, dst, out_id, dataset()->device_type_,
                                      dataset()->stream_, false));
        }

        *end_of_sequence = false;

        TF_DALI_CALL(daliOutputRelease(&pipeline_handle_));
        TF_DALI_CALL(daliRun(&pipeline_handle_));

        return Status::OK();
      }

      ~Iterator() {
        if (enable_memory_stats_) {
          size_t N;
          daliExecutorMetadata *meta;
          daliGetExecutorMetadata(&pipeline_handle_, &meta, &N);
          std::cout << "DALI operator memory statistics: "  << std::endl;
          for (size_t i = 0; i < N; ++i) {
            std::cout << "Operator " << meta[i].operator_name;
            for (size_t j = 0; j < meta[i].out_num; ++j) {
              std::cout << "   output [ " << j << " ] : "
                        << meta[i].real_size[j] << "B allocated "
                        << meta[i].max_real_size[j] << "B max allocated "
                        << meta[i].reserved[j] << "B reserved"
                        << meta[i].max_reserved[j] << "B max reserved";
              if (j != meta[i].out_num - 1) {
                std::cout << ",";
              }
            }
            std::cout << std::endl;
          }
          daliFreeExecutorMetadata(meta, N);
        }
        daliDeletePipeline(&pipeline_handle_);
      }

#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3
    Status SaveInternal(
      SerializationContext* ctx, IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is not supported for DALI dataset.");
    }

    Status RestoreInternal(
      IteratorContext* ctx, IteratorStateReader* reader) override {
      return errors::Unimplemented("RestoreInternal is not supported for DALI dataset");
    }
#endif

     private:

      /**
       * @brief Get a shape that is compatible with the partial required shape (set for TF dataset)
       *        and the shape of batch returned from DALI pipeline.
       *        Allow to squeeze excess dimensions from the DALI shape but we cannot modify the
       *        rank of TF shape.
       *        If there is not unambiguous matching return error status.
       *
       * @param result the matching shape if the function returned Status::OK()
       */
      Status GetCompatibleShape(TensorShape &result, const PartialTensorShape &required_shape,
                                const TensorShape &dali_shape, int batch_size, int output_idx) {
        if (required_shape.IsCompatibleWith(dali_shape)) {
          result = dali_shape;
          return Status::OK();
        }

        // both ranks should be known at this points (otherwise shapes are compatible)
        // If they are not compatible and have the same rank or the required shape is the bigger one
        // we cannot make them compatible.
        if (required_shape.dims() >= dali_shape.dims()) {
          std::stringstream ss;
          ss << "The shape provided for output `" << output_idx << "` is not compatible with "
              << "the shape returned by DALI Pipeline. Expected (output_shapes[" << output_idx
              << "]): " << required_shape << ", got from Pipeline: " << dali_shape << ".";
          return errors::InvalidArgument(ss.str());
        }
        for (int i = 0; i < required_shape.dims(); i++) {
          result.AddDim(0);
        }
        // Non-trivial batch size case, different dims
        if (batch_size != 1) {
          // Should not happen, batch size from DALI will always match
          if (dali_shape.dim_size(0) != batch_size) {
            std::stringstream ss;
            ss << "The shape returned by DALI Pipeline for output `" << output_idx
                << "` has different `batch_size` than the one specified in `DALIDataset`. "
                << "Specified `batch_size`: " << batch_size
                << ", got from Pipeline: " << dali_shape.dim_size(0) << " in shape: "
                << dali_shape << ".";
            return errors::InvalidArgument(ss.str());
          }
          // It's easier to check it in C++ as in Python the structure is a bit convoluted
          if (!DimSizeMatch(required_shape.dim_size(0), batch_size)) {
            std::stringstream ss;
            ss << "The shape provided for output `" << output_idx << "` is not compatible with "
                << "the `batch_size` argument that was specified in `DALIDataset`. "
                << "Specified `batch_size`: " << batch_size << ", got: "
                << required_shape.dim_size(0) << " in shape: " << required_shape << ".";
            return errors::InvalidArgument(ss.str());
          }
        }
        // Special case for 1-element tensors
        if (dali_shape.num_elements() == 1) {
          TensorShape regular_shape;
          if (required_shape.AsTensorShape(&regular_shape) && regular_shape.num_elements() == 1) {
            result = regular_shape;
            return Status::OK();
          }
        }
        int matches = CountShapeMatches(result, required_shape, dali_shape);
        if (matches != 1) {
          std::stringstream ss;
          ss << "The shape provided for output `" << output_idx << "` is not compatible with "
              << "the shape returned by DALI Pipeline in an umabigous way. Expected (output_shapes["
              << output_idx << "]): " << required_shape << ", got from Pipeline: "
              << dali_shape << ".";
          return errors::InvalidArgument(ss.str());
        }
        return Status::OK();
      }

      /**
       * @brief Check if the given dimension sizes match. Negative value represent `None`
       *        placeholder. `dali` values are always concrete
       */
      bool DimSizeMatch(int64_t required, int64_t dali) {
        return required < 0 || required == dali;
      }

      /**
       * @brief Calculate the matching shapes by recursively matching possible. Count the number
       *        of possible matches. 1-sized dimensions in `dali_shape` can be skipped.
       *
       * If there is only 1 match it will be stored in the `result`.
       */
      int CountShapeMatches(TensorShape &result, const PartialTensorShape &required_shape, const TensorShape &dali_shape,
                            int req_pos = 0, int dali_pos = 0) {
        // We went over the whole shapes and they matched on the way
        if (req_pos == required_shape.dims() && dali_pos == dali_shape.dims()) {
          return 1;
        }
        // We have only DALI shape elements left, if they are all `1` it's ok.
        if (req_pos == required_shape.dims() && dali_pos < dali_shape.dims()) {
          if (dali_shape.dim_size(dali_pos) == 1) {
            return CountShapeMatches(result, required_shape, dali_shape, req_pos, dali_pos + 1);
          } else {
            return 0;
          }
        }
        if (req_pos < required_shape.dims() && dali_pos < dali_shape.dims()) {
          int total = 0;
          // We match exactly or to a "None" position
          if (DimSizeMatch(required_shape.dim_size(req_pos), dali_shape.dim_size(dali_pos))) {
            int matches = CountShapeMatches(result, required_shape, dali_shape, req_pos + 1, dali_pos + 1);
            // If we are the only exact match when backing up from recursion, save the result
            if (matches == 1) {
              result.set_dim(req_pos, dali_shape.dim_size(dali_pos));
            }
            total += matches;
          }
          // If DALI returned 1, we can skip this position an try other match
          if (dali_shape.dim_size(dali_pos) == 1) {
            total += CountShapeMatches(result, required_shape, dali_shape, req_pos, dali_pos + 1);
          }
          return total;
        }
        return 0;
      }

      tensorflow::mutex mu_;
      daliPipelineHandle pipeline_handle_;
      bool enable_memory_stats_;
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
  .Attr("enable_memory_stats: bool = false")
  .Attr("output_shapes: list(shape) >= 1")
  .Attr("output_dtypes: list({bool, half, float, uint8, uint16, uint32, uint64, int8, int16, int32, int64}) >= 1")
  .Attr("fail_on_device_mismatch: bool = true")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape)
  .Doc(R"doc(
DALI Dataset plugin
Creates a DALI dataset compatible with tf.data.Dataset from a DALI pipeline.
`shapes` must match the shape of the corresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the corresponding DALI Pipeline output tensors type.
)doc");

}  // namespace dali_tf_impl

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15
