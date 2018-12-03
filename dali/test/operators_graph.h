#ifndef DALI_OPERATORS_GRAPH_H
#define DALI_OPERATORS_GRAPH_H

#include <string>

namespace dali {

namespace testing {

class OperatorsGraph {

public:
  OperatorsGraph(const char* operator_name) {}
  OperatorsGraph(std::string operator_name) {}

private:
  std::vector<std::pair<std::string, std::set<size_t>>> graph_;
};

}  // namespace dali

}  // namespace testing

#endif //DALI_OPERATORS_GRAPH_H
