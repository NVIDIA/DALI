// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
#define NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_

#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class DumpImage : public Operator<Backend> {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator<Backend>(spec),
    suffix_(spec.GetArgument<string>("suffix")) {
    NDLL_ENFORCE(spec.GetArgument<NDLLTensorLayout>("input_layout") == NDLL_NHWC,
        "CHW not supported yet.");
  }

  virtual inline ~DumpImage() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  const string suffix_;
};
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
