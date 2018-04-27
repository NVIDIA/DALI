// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_COPY_H_
#define NDLL_PIPELINE_OPERATORS_COPY_H_

#include <cstring>

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class Copy : public Operator<Backend> {
 public:
  inline explicit Copy(const OpSpec &spec) :
    Operator<Backend>(spec) {}

  virtual inline ~Copy() = default;

  DISABLE_COPY_MOVE_ASSIGN(Copy);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_COPY_H_
