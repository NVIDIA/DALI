// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <chrono>
#include <sstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/public/version.h"

// for Eigen::GpuDevice
#define EIGEN_USE_GPU
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 16
#include "unsupported/Eigen/CXX11/Tensor"
#else
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

#define USE_TF_ALLOCATOR 0
#if USE_TF_ALLOCATOR
#include "dali_tf_plugin/tfallocator.h"
#endif

#include "dali/dali.h"
#include "dali/dali_cpp_wrappers.h"
#include "dali/core/common.h"
#include "dali/core/small_vector.h"
#include "dali_tf_plugin/dali_helper.h"

typedef std::chrono::high_resolution_clock Clock;

namespace tf = tensorflow;

namespace dali_tf_impl {
using namespace dali::c_api;  // NOLINT

namespace {
template <typename Context>
inline void SetDALIErrorStatus(
      Context *ctx,
      daliResult_t status,
      const char *message,
      const char *expression,
      const char *file,
      int line) {
  auto err = DALIException::MakeErrorString(status, message, expression, file, line);
  std::cout << err << std::endl;
  ctx->SetStatus(tf::errors::Internal(std::move(err)));
}
}  // namespace

#define TF_DALI_CALL(...)                                                                    \
  do {                                                                                       \
    auto status = (__VA_ARGS__);                                                             \
    if (status & DALI_ERROR) {                                                               \
      SetDALIErrorStatus(context, status, daliGetLastErrorMessage(), #__VA_ARGS__, __FILE__, \
                         __LINE__);                                                          \
      return;                                                                                \
    }                                                                                        \
  } while (0)

#define TF_TRANSLATE_EXCEPTION(...)                                                                \
  do {                                                                                             \
    try {                                                                                          \
      __VA_ARGS__;                                                                                 \
    } catch (const DALIException &e) {                                                             \
      std::cout << e.what() << std::endl;                                                          \
      context->SetStatus(tf::errors::Internal(e.what()));                                          \
      return;                                                                                      \
    } catch (const std::exception &e) {                                                            \
      std::string error = dali::make_string(                                                       \
          "Error ", e.what(), "\nwhile executing:\n" #__VA_ARGS__ "\nin " __FILE__ ":", __LINE__); \
      std::cout << error << std::endl;                                                             \
      context->SetStatus(tf::errors::Internal(error));                                             \
      return;                                                                                      \
    }                                                                                              \
  } while (0)

REGISTER_OP("Dali")
  .Attr("serialized_pipeline: string")
  .Attr("shapes: list(shape) >= 1")
  .Attr("num_threads: int = -1")
  .Attr("device_id: int = -1")
  .Attr("exec_separated: bool = false")
  .Attr("exec_dynamic: bool = false")
  .Attr("gpu_prefetch_queue_depth: int = 2")
  .Attr("cpu_prefetch_queue_depth: int = 2")
  .Attr("sparse: list(bool) = []")
  .Attr("batch_size: int = -1")
  .Attr("enable_memory_stats: bool = false")
  .Output("data: dtypes")
  .Attr("dtypes: list({half, float, uint8, int16, int32, int64}) >= 1")
  // To prevent replacing DALI op with constant tensor during TF constant folding process
  .SetIsStateful()
  .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
    std::vector<tf::PartialTensorShape> shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
    for (unsigned i = 0; i < shapes.size(); ++i) {
      if (shapes[i].dims() > 0) {
        tf::shape_inference::ShapeHandle passed_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
        TF_RETURN_IF_ERROR(
            c->WithRank(passed_shape, shapes[i].dims(), &passed_shape));
        c->set_output(i, passed_shape);
      }
    }
    return tf::Status();
  })
  .Doc(R"doc(
DALI TensorFlow plugin

Creates a DALI pipeline from a serialized pipeline, obtained from `serialized_pipeline` argument.
`shapes` must match the shape of the coresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the coresponding DALI Pipeline output tensors type.
 )doc");

inline int64_t EnumerateIndices(
      int64_t *indices, const int64_t *shape, int ndim, int64_t *pos, int d) {
  if (d == ndim) {
    for (int i = 0; i < ndim; i++)
      indices[i] = pos[i];
    return ndim;
  } else {
    int64_t extent = shape[d];
    int64_t offset = 0;
    for (int64_t i = 0; i < extent; i++) {
      pos[d] = i;
      offset += EnumerateIndices(indices + offset, shape, ndim, pos, d + 1);
    }
    return offset;
  }
}

inline int64_t EnumerateIndicesWithinSample(
      int64_t *out_indices,
      int sample_idx,
      const int64_t *sample_shape,
      int sample_ndim) {
  dali::SmallVector<int64_t, 8> pos;
  pos.resize(sample_ndim + 1);
  pos[0] = sample_idx;
  // HACK: We now start enumeration as if we had one more dimension, but we start at 1.
  //       To achieve this feat, we need to offset the shape pointer by -1, so all's fine
  //       when it's accessed starting at index 1.
  return EnumerateIndices(out_indices, sample_shape - 1, sample_ndim + 1, pos.data(), 1);
}

class DaliOp : public tf::OpKernel {
 public:
  explicit DaliOp(tf::OpKernelConstruction* context)
    : OpKernel(context) {

    std::string serialized_pipeline;
    OP_REQUIRES_OK(context, context->GetAttr("serialized_pipeline", &serialized_pipeline));

    int num_threads;
    int device_id;
    int max_batch_size;
    bool exec_separated;
    bool exec_dynamic;
    int cpu_prefetch_queue_depth;

    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &types_));
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads));
    OP_REQUIRES_OK(context, context->GetAttr("device_id", &device_id));
    OP_REQUIRES_OK(context, context->GetAttr("exec_separated", &exec_separated));
    OP_REQUIRES_OK(context, context->GetAttr("exec_dynamic", &exec_dynamic));
    // In exec_separated==false case, gpu_prefetch_queue_depth is the global prefetch_queue_depth_
    OP_REQUIRES_OK(context, context->GetAttr("gpu_prefetch_queue_depth", &prefetch_queue_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("sparse", &sparse_));
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &max_batch_size));
    OP_REQUIRES_OK(context, context->GetAttr("cpu_prefetch_queue_depth",
                                             &cpu_prefetch_queue_depth));
    OP_REQUIRES_OK(context, context->GetAttr("enable_memory_stats", &enable_memory_stats_));

    // TF doing constant propagation runs all operators on the CPU first, so we need to provide
    // ability to copy memory from the GPU pipeline to the CPU seamlessly
    this->device_type_ = (context->device_type() == "CPU") ?
                          DALI_STORAGE_CPU : DALI_STORAGE_GPU;
    if (std::any_of(sparse_.begin(), sparse_.end(), [] (const bool &v) {return v;}) &&
        this->device_type_ == DALI_STORAGE_GPU) {
      OP_REQUIRES_OK(context, tf::errors::Internal("Cannot output sparse tensors on the GPU"));
    }
    if (device_id >= 0)
      this->device_id_ = device_id;
    this->batch_size_ = max_batch_size;
    LOG_LINE << "Initializing...\n";

    if (max_batch_size < 0) {
      max_batch_size = shapes_[0].dim_size(0);
    }

    daliExecType_t exec_type = DALI_EXEC_ASYNC_PIPELINED;
    if (exec_dynamic)
      exec_type = exec_type | DALI_EXEC_IS_DYNAMIC;
    if (exec_separated)
      exec_type = exec_type | DALI_EXEC_IS_SEPARATED;

    daliPipelineParams_t params{};
    DALI_SET_PARAM(params, max_batch_size, max_batch_size);
    DALI_SET_PARAM(params, exec_type, exec_type);

    daliPrefetchQueueSizes_t queue_depths;
    queue_depths.cpu = exec_separated
      ? cpu_prefetch_queue_depth
      : prefetch_queue_depth_;
    queue_depths.gpu = prefetch_queue_depth_;
    DALI_SET_PARAM(params, prefetch_queue_depths, queue_depths);
    if (device_id_.has_value())
      DALI_SET_PARAM(params, device_id, *device_id_);
    if (num_threads >= 1)
      DALI_SET_PARAM(params, num_threads, num_threads);
    DALI_SET_PARAM(params, enable_memory_stats, enable_memory_stats_);

    daliPipeline_h handle{};
    TF_DALI_CALL(daliPipelineDeserialize(
        &handle,
        serialized_pipeline.c_str(),
        serialized_pipeline.length(),
        &params));
    pipe_handle_ = dali::c_api::PipelineHandle(handle);
    TF_DALI_CALL(daliPipelineBuild(pipe_handle_));


#if USE_TF_ALLOCATOR
    SetupTFAllocator(device_id_);
    UpdateTFAllocaterContext<tf::OpKernelConstruction>(context, device_id_);
#endif
    LOG_LINE << "Pipeline created\n";
    LOG_LINE << "Prefetching...\n";
    TF_DALI_CALL(daliPipelinePrefetch(pipe_handle_));
    LOG_LINE << "After first run\n";
  }

  ~DaliOp() override {
    pipe_handle_.reset();
    /* TODO(michalz): Remove or implement memory stats in C API 2.0
    if (pipe_handle_) {
      if (enable_memory_stats_) {
        size_t N;
        daliExecutorMetadata *meta;
        daliGetExecutorMetadata(&pipe_handle_, &meta, &N);
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
    }*/
  }

  void Compute(tf::OpKernelContext* context) override {
    TF_TRANSLATE_EXCEPTION(ComputeImpl(context));
  }

  void ComputeImpl(tf::OpKernelContext* context) {
    auto total_s = Clock::now();

#if USE_TF_ALLOCATOR
    UpdateTFAllocaterContext<tf::OpKernelContext>(context, device_id_);
    LOG_LINE << "Updated context\n";
#endif
    LOG_LINE << "Before output...\n";

    auto s = Clock::now();

    daliPipelineOutputs_h pipe_outputs_h;
    DALI_RETHROW(daliPipelinePopOutputs(pipe_handle_, &pipe_outputs_h));
    PipelineOutputsHandle pipe_outputs(pipe_outputs_h);

    int64_t output_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - s).count();
    LOG_LINE << "After output...\n";

    s = Clock::now();

    tf::OpOutputList outputs;
    std::vector<tf::Tensor*> data_output_tensors;
    // each sparse tensor need 3 tensors in total - values, indices and shape
    int additional_sparse_tensors = std::accumulate(sparse_.begin(), sparse_.end(), 0) * 2;
    int dali_num_out = 0;
    TF_DALI_CALL(daliPipelineGetOutputCount(pipe_handle_, &dali_num_out));
    data_output_tensors.resize(dali_num_out + additional_sparse_tensors);

    OP_REQUIRES_OK(context, context->output_list("data", &outputs));

    cudaStream_t stream = 0;
    if (this->device_type_ == DALI_STORAGE_GPU) {
      stream = context->eigen_device<Eigen::GpuDevice>().stream();
    }

    for (unsigned i = 0, j = 0; i < dali_num_out; ++i, ++j) {
      bool should_be_sparse_tensor = i < sparse_.size() && sparse_[i];
      std::vector<tf::int64> max_dims;

      daliTensorList_h output_tl_h;
      TF_DALI_CALL(daliPipelineOutputsGet(pipe_outputs, &output_tl_h, i));
      TensorListHandle output_tl(output_tl_h);

      int num_samples = 0, ndim = 0;
      const int64_t *out_tl_shape = nullptr;
      TF_DALI_CALL(daliTensorListGetShape(output_tl, &num_samples, &ndim, &out_tl_shape));
      int batch_ndim = ndim + 1;

      if (!should_be_sparse_tensor) {
        bool is_uniform = IsUniform(num_samples, ndim, out_tl_shape);
        if (!is_uniform) {
          OP_REQUIRES(
              context,
              is_uniform,
              tensorflow::errors::FailedPrecondition(
                  "Batch output at index '", i,
                  "' from DALI pipeline is not uniform - individual samples have different "
                  "shapes. This output cannot be represented as single, dense Tensor, which is "
                  "required by TensorFlow. Ensure that all the samples that you produce in given "
                  "batch have equal shape. Or use sparse output representation. Got shapes: ",
                  ShapeToString(output_tl)));
        }
        // build a shape of the output as if it was a single tensor, with leading num_samples
        std::vector<int64_t> shape_as_tensor(ndim + 1);
        shape_as_tensor[0] = num_samples;
        if (num_samples > 0) {
          for (int i = 0; i < ndim; i++)
            shape_as_tensor[i + 1] = out_tl_shape[i];
        }
        tf::TensorShape data_output_shape = ToTfShape(shape_as_tensor.data(), ndim + 1);

        // If tensor has shape provided it need to match
        OP_REQUIRES(context, shapes_[i].dims() <= 0 || data_output_shape == shapes_[i],
        tf::errors::InvalidArgument("DALI pipeline output shape at " + std::to_string(i) +
                                    " " + data_output_shape.DebugString() + " != "
                                    + shapes_[i].DebugString() + " plugin `shapes` argument"));
        OP_REQUIRES_OK(context, outputs.allocate(j, data_output_shape, &data_output_tensors[j]));
      } else {
        max_dims.resize(batch_ndim);
        // first dim is number of elements in the tensor list
        max_dims[0] = num_samples;
        ptrdiff_t total_elms = 0;

        tf::TensorShape data_output_shape;
        for (int i = 0; i < num_samples; i++) {
          ptrdiff_t sample_volume = 1;
          for (int d = 0; d < ndim; d++) {
            int extent = out_tl_shape[i * ndim + d];
            if (extent > max_dims[d + 1]) {
              max_dims[d + 1] = extent;
            }
            sample_volume *= extent;
          }
          total_elms += sample_volume;
        }
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({total_elms, batch_ndim}),
                                                 &data_output_tensors[j]));
        tf::Tensor* out_tensor = data_output_tensors[j];
        auto p_out_indices = out_tensor->flat<tf::int64>().data();
        assert(total_elms * batch_ndim * sizeof(int64_t) <= out_tensor->TotalBytes());
        int64_t idx_offset = 0;
        for (int s = 0; s < num_samples; s++) {
          auto n = EnumerateIndicesWithinSample(
              p_out_indices + idx_offset,
              s,
              out_tl_shape + s * ndim,
              ndim);
          idx_offset += n;
          assert(idx_offset <= total_elms * batch_ndim);
        }
        assert(idx_offset * sizeof(int64_t) <= out_tensor->TotalBytes());
        ++j;
        // allocate output
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({total_elms}),
                                                 &data_output_tensors[j]));
      }
      void *dst = nullptr;
      tf::Tensor* out_tensor = data_output_tensors[j];
      size_t dali_tensor_size = 0;
      TF_DALI_CALL(daliTensorListGetByteSize(output_tl, &dali_tensor_size));
      if (dali_tensor_size > out_tensor->TotalBytes()) {
        context->CtxFailure(__FILE__, __LINE__,
            tf::errors::InvalidArgument("Output " + std::to_string(i) +
              " has bigger size than allocated by TensorFlow - check if type requested matches" +
              " with one produced by the DALI pipeline"));
      }
      switch (types_[j]) {
        case tf::DT_HALF:
              dst = reinterpret_cast<void*>(out_tensor->flat<Eigen::half>().data());
          break;
        case tf::DT_FLOAT:
              dst = reinterpret_cast<void*>(out_tensor->flat<float>().data());
          break;
        case tf::DT_UINT8:
              dst = reinterpret_cast<void*>(out_tensor->flat<uint8_t>().data());
          break;
        case tf::DT_INT16:
              dst = reinterpret_cast<void*>(out_tensor->flat<int16_t>().data());
          break;
        case tf::DT_INT32:
              dst = reinterpret_cast<void*>(out_tensor->flat<int32_t>().data());
          break;
        case tf::DT_INT64:
              dst = reinterpret_cast<void*>(out_tensor->flat<tf::int64>().data());
          break;
        default:
          context->CtxFailure(__FILE__, __LINE__,
            tf::errors::InvalidArgument("Unsupported type: " + tf::DataTypeString(types_[i]) +
                                        "for tensor " + std::to_string(i)));
          break;
      }

      // Synchronize with the dataset()->stream_ when doing the last copy, so the outputs
      // are fully finished before we release the output buffers for reuse.
      // if the OP runs on the CPU the output memory is not pinned and we don't need to sync
      daliCopyFlags_t flags = this->device_type_ != DALI_STORAGE_CPU && (i == dali_num_out - 1)
        ? DALI_COPY_SYNC
        : DALI_COPY_DEFAULT;
      daliBufferPlacement_t dst_placement{device_type_, device_id_.value_or(0), false};
      daliBufferPlacement_t tl_placement{};
      TF_DALI_CALL(daliTensorListGetBufferPlacement(output_tl, &tl_placement));
      bool use_stream = device_type_ == DALI_STORAGE_GPU ||
                        tl_placement.device_type == DALI_STORAGE_GPU;
      TF_DALI_CALL(daliTensorListCopyOut(
          output_tl, dst, dst_placement, use_stream ? &stream : nullptr, flags));
      if (should_be_sparse_tensor) {
        ++j;
        // copy out shape
        assert(max_dims.size() == static_cast<size_t>(batch_ndim));
        OP_REQUIRES_OK(context, outputs.allocate(j, tf::TensorShape({batch_ndim}),
                                                 &data_output_tensors[j]));
        auto out_tensor = data_output_tensors[j];
        assert(out_tensor->TotalBytes() >= batch_ndim * sizeof(int64_t));
        auto out_shape = out_tensor->flat<tf::int64>().data();
        for (int k = 0; k < batch_ndim; ++k) {
          out_shape[k] = max_dims[k];
        }
      }
    }
    int64_t copy_time =  std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - s).count();

    pipe_outputs.reset();

    LOG_LINE << "Computing...\n";
    s = Clock::now();
    TF_DALI_CALL(daliPipelineRun(pipe_handle_));
    int64_t run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - s).count();

    int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - total_s).count();

    LOG_LINE << "[TIMES] TOTAL " << total_time << " RUN " << run_time
      << " - OUTPUT " << output_time << " - ALLOC + COPY " << copy_time << std::endl;
  }

 private:
  dali::c_api::PipelineHandle pipe_handle_;
  std::vector<tf::TensorShape> shapes_;
  tf::DataTypeVector types_;
  std::optional<int> device_id_;
  int batch_size_ = 0;
  int prefetch_queue_depth_ = -1;
  daliStorageDevice_t device_type_ = DALI_STORAGE_CPU;
  std::vector<bool> sparse_;
  bool enable_memory_stats_ = false;
};

using tf::int64;

REGISTER_KERNEL_BUILDER(Name("Dali").Device(tf::DEVICE_GPU), DaliOp)
REGISTER_KERNEL_BUILDER(Name("Dali").Device(tf::DEVICE_CPU), DaliOp)

}  // namespace dali_tf_impl
