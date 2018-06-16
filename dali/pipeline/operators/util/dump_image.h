// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_

#include <string>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class DumpImage : public Operator<Backend> {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator<Backend>(spec),
    suffix_(spec.GetArgument<string>("suffix")) {
    DALI_ENFORCE(spec.GetArgument<DALITensorLayout>("input_layout") == DALI_NHWC,
        "CHW not supported yet.");
  }

  virtual inline ~DumpImage() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  const string suffix_;
};
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_
