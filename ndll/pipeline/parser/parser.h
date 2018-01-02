#ifndef NDLL_PIPELINE_PARSER_PARSER_H_
#define NDLL_PIPELINE_PARSER_PARSER_H_

#include "ndll/pipeline/sample_workspace.h"

namespace ndll {

/**
 * Base class for parsing data returned from a Loader
 */
class Parser {
 public:
  explicit Parser(const OpSpec& spec) {}

  /**
   * Parse the information contained in data to
   * whatever is necessary to continue the pipeline.
   * e.g. Extracting (image, label) pairs from a protobuf
   * entry
   */
  virtual void Parse(uint8_t* data, SampleWorkspace* ws) = 0;
};

}  // namespace ndll

#endif
