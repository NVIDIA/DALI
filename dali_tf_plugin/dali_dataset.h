// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TF_PLUGIN_DALI_DATASET_H_
#define DALI_TF_PLUGIN_DALI_DATASET_H_

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"


#include "dali/core/common.h"
#include "dali/core/format.h"

namespace dali_tf_impl {

class DALIDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit DALIDatasetOp(tensorflow::OpKernelConstruction* context)
      : DatasetOpKernel(context),
        is_gpu_device_(context->device_type() == "GPU"),
        context_(context) {
    FillPipelineDef(context, pipeline_def_);
    FillInputDef(context, input_def_);
    OP_REQUIRES_OK(context, context->GetAttr(kOutputShapes, &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr(kOutputDtypes, &dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr(kDeviceMismatch, &fail_on_device_mismatch_));
  }

  void MakeDataset(tensorflow::OpKernelContext* context,
                   tensorflow::data::DatasetBase** output) override;

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

  // TODO(klecki): I know the name Inputs is not ideal, but we actually get them
  // in compute from Op inputs.
  struct Inputs {
    std::vector<tensorflow::data::DatasetBase *> inputs;
    std::vector<device_type_t> input_devices;
  };

  // Those are the static Attrs describing inputs
  struct InputDef {
    std::vector<std::string> input_names;
    std::vector<std::string> input_layouts;
    bool input_batch = false;
  };

  struct InputDesc : Inputs, InputDef {
    InputDesc(const Inputs& inputs, const InputDef& input_def)
        : Inputs(inputs), InputDef(input_def) {}
  };

  // DALI Pipeline arguments
  static constexpr const char* const kPipeline = "pipeline";
  static constexpr const char* const kBatchSize = "batch_size";
  static constexpr const char* const kNumThreads = "num_threads";
  static constexpr const char* const kDeviceId = "device_id";
  static constexpr const char* const kExecSeparated = "exec_separated";
  static constexpr const char* const kPrefetchQueueDepth = "prefetch_queue_depth";
  static constexpr const char* const kCpuPrefetchQueueDepth = "cpu_prefetch_queue_depth";
  static constexpr const char* const kGpuPrefetchQueueDepth = "gpu_prefetch_queue_depth";
  static constexpr const char* const kGpuMemoryStats = "enable_memory_stats";

  // Arguments describing inputs
  static constexpr const char* const kInputNames = "input_names";
  static constexpr const char* const kInputLayouts = "input_layouts";
  static constexpr const char* const kInputBatch = "input_batch";

  // Arguments describing outputs
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kOutputDtypes = "output_dtypes";

  // DatasetOp-specific arguments
  static constexpr const char* const kDeviceMismatch = "fail_on_device_mismatch";


  void FillPipelineDef(tensorflow::OpKernelConstruction* context, PipelineDef& def);
  void FillInputs(tensorflow::OpKernelContext *context, Inputs &def);
  void FillInputDef(tensorflow::OpKernelConstruction* context, InputDef& def);
  void ValidateInputs(tensorflow::OpKernelContext *context, Inputs &inputs, InputDef& input_def);

  PipelineDef pipeline_def_;
  InputDef input_def_;
  std::vector<tensorflow::PartialTensorShape> shapes_;
  tensorflow::DataTypeVector dtypes_;
  bool is_gpu_device_;
  bool fail_on_device_mismatch_;
  tensorflow::OpKernelConstruction* context_;

  class Dataset;
};


}  // namespace dali_tf_impl

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15

#endif  // DALI_TF_PLUGIN_DALI_DATASET_H_
