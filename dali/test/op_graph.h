#ifndef DALI_OP_GRAPH_H
#define DALI_OP_GRAPH_H

#include <string>

namespace dali {
namespace testing {

class OpDag {
public:
  virtual std::string get_op_name() = 0;
}; // PR #369

struct OpDagStub : public OpDag {
  OpDagStub(std::string name) : name_(name) {}


  std::string get_op_name() { return name_; }


  std::string name_;
};

}  // namespace testing
}  // namespace dali

#endif //DALI_OP_GRAPH_H
