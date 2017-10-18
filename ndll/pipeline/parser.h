#ifndef NDLL_PIPELINE_PARSER_H_
#define NDLL_PIPELINE_PARSER_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/datum.h"
#include "ndll/pipeline/operator_factory.h"
#include "ndll/pipeline/op_spec.h"

namespace ndll {

/**
 * @brief Defines the API for a Parser. Parser are a way for users to handle their
 * own custom data formats prior to processing in the pipeline.
 */
class Parser {
public:
  inline Parser(const OpSpec &spec) :
    num_threads_(spec.GetSingleArgument<int>("num_threads", -1)),
    batch_size_(spec.GetSingleArgument<int>("batch_size", -1)) {
    NDLL_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }
  
  virtual ~Parser() = default;

  /**
   * @brief Parses the data from the input Datum into the output
   *
   * Note: The parser is responsible for resizing its output Datum
   * prior to accessing its data
   */
  virtual void Parse(const Datum<CPUBackend> &input, Datum<CPUBackend> *output,
      int data_idx, int thread_idx) = 0;

  DISABLE_COPY_MOVE_ASSIGN(Parser);
protected:
  int num_threads_, batch_size_;
};

// Create registries for DataReaders
NDLL_DECLARE_OPTYPE_REGISTRY(Parser, Parser);

#define NDLL_REGISTER_PARSER(OpName, OpType)      \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,   \
      ndll::Parser, ndll::Parser)

/**
 * @brief The default Parser used by the Pipeline if no user-defined Parser is
 * added. This Parser simply copies the input to the output. This is not ideal
 * for perf testing, if this ends up being used in real applications we should
 * devise a mechanism for zero-copy to avoid this unnesscessary overhead
 */
class DefaultParser final : public Parser {
public:
  inline DefaultParser(const OpSpec &spec) : Parser(spec) {}
  ~DefaultParser() = default;

  inline void Parse(const Datum<CPUBackend> &input, Datum<CPUBackend> *output,
      int /* unused */, int /* unused */) override {
    output->Copy(input);
  }

  DISABLE_COPY_MOVE_ASSIGN(DefaultParser);
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_PARSER_H_
