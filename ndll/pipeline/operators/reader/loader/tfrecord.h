// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_LOADER_TFRECORD_H_
#define NDLL_PIPELINE_LOADER_TFRECORD_H_

#include "ndll/common.h"
#include "ndll/pipeline/operators/reader/loader/loader.h"

namespace ndll {

class TFRecordLoader : public Loader<CPUBackend> {
 public:
  explicit TFRecordLoader(const OpSpec& options)
    : Loader(options) {
    }
  void ReadSample(Tensor<CPUBackend>* tensor) override {
  }

  Index Size() override {
    return 0;
  }

};

}  // namespace ndll

#endif  // NDLL_PIPELINE_LOADER_TFRECORD_H_
