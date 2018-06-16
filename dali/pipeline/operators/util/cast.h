// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_UTIL_CAST_H_
#define DALI_PIPELINE_OPERATORS_UTIL_CAST_H_

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Cast : public Operator<Backend> {
 public:
  explicit inline Cast(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("dtype"))
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

  DALIDataType output_type_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_CAST_H_
