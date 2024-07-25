// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/reader/loader/sequence_loader.h"
#include "dali/operators/reader/parser/parser.h"

namespace dali {

class SequenceParser : public Parser<TensorSequence> {
 public:
  explicit SequenceParser(const OpSpec& spec)
      : Parser<TensorSequence>(spec), image_type_(spec.GetArgument<DALIImageType>("image_type")) {}

  void Parse(const TensorSequence& data, SampleWorkspace* ws) override;

 private:
  DALIImageType image_type_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_SEQUENCE_PARSER_H_
