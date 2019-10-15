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

#ifndef DALI_OPERATORS_READER_PARSER_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_PARSER_H_

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

/**
 * @brief Base class for parsing data returned from a Loader
 *
 * @tparam ParseTarget Wrapper for data to be parsed, usually LoadTarget template param
 *                     forwarded from Loader by DataReader.
 */
template <typename ParseTarget>
class Parser {
 public:
  explicit Parser(const OpSpec& spec) {}
  virtual ~Parser() {}

  /**
   * Parse the information contained in data to
   * whatever is necessary to continue the pipeline.
   * e.g. Extracting (image, label) pairs from a protobuf
   * entry
   */
  virtual void Parse(const ParseTarget& data, SampleWorkspace* ws) = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_PARSER_H_
