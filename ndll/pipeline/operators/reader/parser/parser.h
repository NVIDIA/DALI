// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_PARSER_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_PARSER_H_

#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

/**
 * Base class for parsing data returned from a Loader
 */
class Parser {
 public:
  explicit Parser(const OpSpec& spec) {}
  virtual ~Parser() {}

  /**
   * Parse the 'size' bytes of information contained in data to
   * whatever is necessary to continue the pipeline.
   * e.g. Extracting (image, label) pairs from a protobuf
   * entry
   */
  virtual void Parse(const uint8_t* data,
                     const size_t size,
                     SampleWorkspace* ws) = 0;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_PARSER_PARSER_H_
