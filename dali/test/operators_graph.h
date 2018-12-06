#ifndef DALI_OPERATORS_GRAPH_H
#define DALI_OPERATORS_GRAPH_H

#include <string>

namespace dali {

namespace testing {

struct Operator{};

class OperatorsGraph {

public:
  OperatorsGraph(const char* operator_name) {}

private:
  std::vector<std::pair<Operator, std::set<size_t>>> graph_;
};

}  // namespace dali

}  // namespace testing

#endif //DALI_OPERATORS_GRAPH_H
