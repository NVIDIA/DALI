#ifndef DALI_OPERATOR_ADAPTER_H
#define DALI_OPERATOR_ADAPTER_H

#include <string>
#include <dali/pipeline/operators/op_spec.h>

namespace dali {

namespace testing {

class OperatorAdapter {

 dali::OpSpec GenerateOpSpec() const noexcept {}

 private:
  std::string operator_name_;
};

}  // namespace testing

}  // namespace dali

#endif //DALI_OPERATOR_ADAPTER_H
