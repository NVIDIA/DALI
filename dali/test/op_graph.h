#ifndef DALI_OP_GRAPH_H
#define DALI_OP_GRAPH_H

#include <string>

namespace dali {
namespace testing {

class OpDag {
public:
  OpDag(std::string name) : name_(name) {}
  std::string get_op_name() { return name_; };
  std::string name_;
}; // PR #369


}  // namespace testing
}  // namespace dali

#endif //DALI_OP_GRAPH_H
