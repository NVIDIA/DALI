// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_CAST_H_
#define NDLL_PIPELINE_OPERATORS_CAST_H_

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class Cast : public Operator<Backend> {
 public:
  explicit inline Cast(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<NDLLDataType>("dtype"))
    {}

  virtual inline ~Cast() = default;

  DISABLE_COPY_MOVE_ASSIGN(Cast);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

 private:
  template <typename IType, typename OType>
  inline void CPUHelper(OType * out, const IType * in, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      out[i] = static_cast<OType>(in[i]);
    }
  }

  NDLLDataType output_type_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_CAST_H_
