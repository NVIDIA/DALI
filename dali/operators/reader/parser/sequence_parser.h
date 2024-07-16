// Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_PARSER_SEQUENCE_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_SEQUENCE_PARSER_H_

#include "dali/operators/imgcodec/image_decoder.h"
#include "dali/operators/reader/loader/sequence_loader.h"
#include "dali/operators/reader/parser/parser.h"
namespace dali {

class SequenceParser : public Parser<TensorSequence> {
 public:
  explicit SequenceParser(const OpSpec& spec)
      : Parser<TensorSequence>(spec), image_type_(spec.GetArgument<DALIImageType>("image_type")) {
    EnforceMinimumNvimgcodecVersion();

    nvimgcodecInstanceCreateInfo_t instance_create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t),
        nullptr};

    const char* log_lvl_env = std::getenv("DALI_NVIMGCODEC_LOG_LEVEL");
    int log_lvl = log_lvl_env ? clamp(atoi(log_lvl_env), 1, 5) : 2;

    instance_create_info.load_extension_modules = 1;
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.extension_modules_path = nullptr;
    instance_create_info.create_debug_messenger = 1;
    instance_create_info.debug_messenger_desc = nullptr;
    instance_create_info.message_severity = imgcodec::verbosity_to_severity(log_lvl);
    instance_create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
    instance_ = imgcodec::NvImageCodecInstance::Create(&instance_create_info);

    exec_params_.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params_.backends = &backend_;
    exec_params_.num_backends = 1;
    exec_params_.device_allocator = nullptr;
    exec_params_.pinned_allocator = nullptr;
    exec_params_.executor = nullptr;
    exec_params_.max_num_cpu_threads = 1;
    exec_params_.pre_init = 1;
    decoder_ = imgcodec::NvImageCodecDecoder::Create(instance_, &exec_params_, {});
  }

  void Parse(const TensorSequence& data, SampleWorkspace* ws) override;

 private:
  DALIImageType image_type_;
  imgcodec::NvImageCodecInstance instance_ = {};
  imgcodec::NvImageCodecDecoder decoder_ = {};

  nvimgcodecBackend_t backend_{
      NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
      sizeof(nvimgcodecBackend_t),
      nullptr,
      NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
      {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}};
  nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                                           sizeof(nvimgcodecExecutionParams_t), nullptr};
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_SEQUENCE_PARSER_H_
