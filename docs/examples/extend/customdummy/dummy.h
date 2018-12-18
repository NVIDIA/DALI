#ifndef EXAMPLE_DUMMY_H_
#define EXAMPLE_DUMMY_H_

#include "dali/pipeline/operators/operator.h"

namespace other_ns {

template <typename Backend>
class Dummy : public ::dali::Operator<Backend> {
 public:
  inline explicit Dummy(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {}

  virtual inline ~Dummy() = default;

  Dummy(const Dummy&) = delete;
  Dummy& operator=(const Dummy&) = delete;
  Dummy(Dummy&&) = delete;
  Dummy& operator=(Dummy&&) = delete;

 protected:
  void RunImpl(::dali::Workspace<Backend> *ws, const int idx) override;
};

}  // namespace other_ns

#endif  // EXAMPLE_DUMMY_H_
