#ifndef NDLL_PIPELINE_OPERATOR_H_
#define NDLL_PIPELINE_OPERATOR_H_

#include <utility>

#include "ndll/common.h"

namespace ndll {

/**
 * @brief Baseclass for the basic unit of computation in the pipeline
 */
class Operator {
public:
  Operator() : id_(num_ops_) { ++num_ops_; }
  virtual ~Operator() = default;

  /**
   * Move constructor to allow transfer of ownership of the 
   * op from the user to the pipeline
   */
  Operator(Operator &&op) noexcept {
    std::swap(id_, op.id_);
  }

  int id() const { return id_; }
  
  Operator& operator=(Operator &&op) = delete;
  DISABLE_COPY_ASSIGN(Operator);
private:
  int id_;

  static int num_ops_;
};

class Decoder : public Operator {
public:
  Decoder() {}
  virtual ~Decoder() = default;
  DISABLE_COPY_ASSIGN(Decoder);
private:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATOR_H_
