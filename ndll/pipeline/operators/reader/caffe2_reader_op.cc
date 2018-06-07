// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/reader/caffe2_reader_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Caffe2Reader, Caffe2Reader, CPU);

NDLL_SCHEMA(Caffe2Reader)
  .DocStr("Read sample data from a Caffe2 LMDB")
  .NumInput(0)
  .OutputFn([](const OpSpec& spec) {
      auto label_type = static_cast<LabelType>(spec.GetArgument<int>("label_type"));

      int num_label_outputs = (label_type == MULTI_LABEL_SPARSE ||
                               label_type == MULTI_LABEL_WEIGHTED_SPARSE) ? 2 : 1;
      int additional_inputs = spec.GetArgument<int>("additional_inputs");
      int has_bbox = static_cast<int>(spec.GetArgument<bool>("bbox"));
    return 1 + num_label_outputs + additional_inputs + has_bbox;
  })
  .AddArg("path",
      R"code(`string`
      Path to Caffe2 LMDB directory)code")
  .AddOptionalArg("num_labels", "Foo", 1)
  .AddOptionalArg("label_type", "Foo", 0)
  .AddOptionalArg("additional_inputs", "Foo", 0)
  .AddOptionalArg("bbox", "Foo", false)
  .AddParent("LoaderBase");

}  // namespace ndll

