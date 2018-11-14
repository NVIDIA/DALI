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

#ifndef DALI_PIPELINE_OPERATORS_TENSORRTINFER_TENSORRTINFER_H_
#define DALI_PIPELINE_OPERATORS_TENSORRTINFER_TENSORRTINFER_H_

#include <cuda_runtime_api.h>
#include <common.h>
#include <dlfcn.h>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

typedef struct {
    // Binding Index
    int binding_index;
    // DataType for each bindings
    DALIDataType data_type;
    // Shape required for allocating memory to output tensorlist
    std::vector<Dims> dali_dimensions;
}BindingParam;

template <typename Backend> class TensorRTInfer : public Operator<Backend> {
 public:
  explicit inline TensorRTInfer(const OpSpec &spec) :
    Operator<Backend>(spec),
    input_blobs_(spec.GetRepeatedArgument<std::string>("input_blobs")),
    output_blobs_(spec.GetRepeatedArgument<std::string>("output_blobs")),
    plugins_(spec.GetRepeatedArgument<std::string>("plugins")),
    trt_batch_size_(spec.GetArgument<int64>("trt_batch_size")),
    num_outputs_(spec.GetArgument<int>("num_outputs")),
    engine_data_(spec.GetArgument<std::string>("engine")),
    use_dla_core_(spec.GetArgument<int>("use_dla_core")) {
    for (auto &s : plugins_) {
        void *dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh) {
          std::cout << "Error while performing dlopen on the" << s << " library"
                    << dlerror() << std::endl;
        }
     }
    initLibNvInferPlugins(&trt_logger_, "");
    runtime_ = nvinfer1::createInferRuntime(trt_logger_);
    if (-1 != use_dla_core_ && use_dla_core_ < MAX_ALLOWED_DLA_CORE) {
        runtime_->setDLACore(use_dla_core_);
    }
    engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size(), nullptr);
    context_ = engine_->createExecutionContext();

    DALI_ENFORCE(use_dla_core_ < MAX_ALLOWED_DLA_CORE,
         "DLA core id should be less than " + std::to_string(MAX_ALLOWED_DLA_CORE));
    DALI_ENFORCE(input_blobs_.size() > 0, "Input blob is missing");
    DALI_ENFORCE(output_blobs_.size() > 0, "Output blob is missing");
    DALI_ENFORCE((static_cast<size_t>(engine_->getNbBindings()) ==
         (input_blobs_.size() + output_blobs_.size())),
         "Number of bindings mismatch with the engine");
    DALI_ENFORCE(trt_batch_size_ <= engine_->getMaxBatchSize(),
         "Batch size is greater than MaxBatch size configured for the engine");

    for (size_t i = 0; i < input_blobs_.size(); i++) {
        std::string blobName = input_blobs_[i];
        BindingParam param;
        param.binding_index = engine_->getBindingIndex(blobName.c_str());
        binding_param_[blobName] = param;
    }

    for (size_t i = 0; i < output_blobs_.size(); i++) {
       std::string blobName = output_blobs_[i];
       BindingParam param;
       // Retrieve binding index from engine
       param.binding_index = engine_->getBindingIndex(blobName.c_str());

       // Retrieve binding Dimensions from engine
       nvinfer1::Dims dimensions =
          static_cast<nvinfer1::Dims&&>(engine_->getBindingDimensions(param.binding_index));
       std::vector<Index> dim;
       dim.push_back(trt_batch_size_);
       for (int j = 0; j < dimensions.nbDims; j++) {
          dim.push_back(dimensions.d[j]);
       }
       param.dali_dimensions.push_back(dim);
       nvinfer1::DataType datatype = engine_->getBindingDataType(param.binding_index);
       param.data_type = (datatype == nvinfer1::DataType::kHALF) ?
          DALI_FLOAT16 : (datatype == nvinfer1::DataType::kINT32) ?
          DALI_INT32 : DALI_FLOAT;
       binding_param_[blobName] = param;
    }
  }

  ~TensorRTInfer() {
      if (engine_) engine_->destroy();
      if (runtime_) runtime_->destroy();
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

 private:
  std::vector<std::string> input_blobs_;
  std::vector<std::string> output_blobs_;
  std::vector<std::string> plugins_;

  int trt_batch_size_;
  int num_outputs_;

  std::string engine_data_;
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  std::map<std::string, BindingParam> binding_param_;
  Logger trt_logger_;
  int use_dla_core_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_TENSORRTINFER_TENSORRTINFER_H_

