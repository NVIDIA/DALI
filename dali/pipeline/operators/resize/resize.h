// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/fused/resize_crop_mirror.h"
#include "dali/pipeline/operators/resize/resize_base.h"
#include "dali/kernels/context.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {

class ResizeAttr : protected ResizeCropMirrorAttr {
 public:
  explicit inline ResizeAttr(const OpSpec &spec) : ResizeCropMirrorAttr(spec) {}

  void SetBatchSize(int batch_size) {
    per_sample_meta_.reserve(batch_size);
  }

 protected:
  uint ResizeInfoNeeded() const override { return 0; }

  // store per-thread data for same resize on multiple data
  std::vector<TransformMeta> per_sample_meta_;
};

template <typename Backend>
class Resize : public Operator<Backend>
             , protected ResizeAttr
             , protected ResizeBase {
 public:
  explicit Resize(const OpSpec &spec);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  kernels::ResamplingParams2D GetResamplingParams(const TransformMeta &meta) const {
    kernels::ResamplingParams2D params;
    params[0].output_size = meta.rsz_h;
    params[1].output_size = meta.rsz_w;
    params[0].min_filter = params[1].min_filter = min_filter_;
    params[0].mag_filter = params[1].mag_filter = mag_filter_;
    return params;
  }

  USE_OPERATOR_MEMBERS();
  bool save_attrs_;
  int outputs_per_idx_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_
