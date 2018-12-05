#ifndef DALI_OPERATOR_ARGUMENTS_H
#define DALI_OPERATOR_ARGUMENTS_H

#include <map>

namespace dali {

namespace testing {

class OperatorArguments {
 public:
  using Arguments = std::map<std::string, double>; // TODO generalization (boost:any? tagged union?)
  OperatorArguments(Arguments single_operator_arguments) {}
};

}  // namespace testing

}  // namespace dali

#endif //DALI_OPERATOR_ARGUMENTS_H
