#ifndef NDLL_PIPELINE_PARSER_H_
#define NDLL_PIPELINE_PARSER_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/batch.h"

namespace ndll {

/**
 * @brief Defines the API for a Parser. Parser are a way for users to handle their
 * own custom data formats prior to processing in the pipeline.
 */
template <typename Backend,
          typename std::enable_if<std::is_base_of<CPUBackend, Backend>::value, int>::type = 0>
class Parser {
public:
  Parser() {}
  virtual ~Parser() = default;

  /**
   * @brief Parses the data from the input Datum into the output
   *
   * Note: The parser is responsible for resizing its output Datum
   * prior to accessing its data
   */
  virtual void Run(const Datum<Backend> &input, Datum<Backend> *output) = 0;

  virtual Parser* Clone() const = 0;
  
  DISABLE_COPY_MOVE_ASSIGN(Parser);
protected:
};

/**
 * @brief The default Parser used by the Pipeline if no user-defined Parser is
 * added. This Parser simply copies the input to the output. This is not ideal
 * for perf testing, if this ends up being used in real applications we should
 * devise a mechanism for zero-copy to avoid this unnesscessary overhead
 */
template <typename Backend>
class DefaultParser final : public Parser<Backend> {
public:
  DefaultParser() {}
  ~DefaultParser() {}

  void Run(const Datum<Backend> &input, Datum<Backend> *output) {
    output->Copy(input);
  }

  DefaultParser* Clone() const override {
    return new DefaultParser;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(DefaultParser);
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_PARSER_H_
