// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
class Resize : public Operator<Backend>, protected ResizeAttr {
 public:
  explicit Resize(const OpSpec &spec);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  void SetupResamplingParams() {
    DALIInterpType interp_min = DALIInterpType::DALI_INTERP_TRIANGULAR;
    DALIInterpType interp_mag = DALIInterpType::DALI_INTERP_LINEAR;

    const OpSpec &spec = this->spec_;  // avoid dependent name hell
    if (spec.HasArgument("min_filter"))
      interp_min = spec.GetArgument<DALIInterpType>("min_filter");
    else if (spec.HasArgument("interp_type"))
      interp_min = spec.GetArgument<DALIInterpType>("interp_type");

    if (spec.HasArgument("mag_filter"))
      interp_mag = spec.GetArgument<DALIInterpType>("mag_filter");
    else if (spec.HasArgument("interp_type"))
      interp_mag = spec.GetArgument<DALIInterpType>("interp_type");

    min_filter_ = { interp2resample(interp_min), 0 };
    mag_filter_ = { interp2resample(interp_mag), 0 };
  }

  kernels::ResamplingParams2D GetResamplingParams(const TransformMeta &meta) const {
    kernels::ResamplingParams2D params;
    params[0].output_size = meta.rsz_h;
    params[1].output_size = meta.rsz_w;
    params[0].min_filter = params[1].min_filter = min_filter_;
    params[0].mag_filter = params[1].mag_filter = mag_filter_;
    return params;
  }

  kernels::ResamplingFilterType interp2resample(DALIInterpType interp) {
#define DALI_MAP_INTERP_TO_RESAMPLE(interp, resample) case DALI_INTERP_##interp:\
  return kernels::ResamplingFilterType::resample;

    switch (interp) {
      DALI_MAP_INTERP_TO_RESAMPLE(NN, Nearest);
      DALI_MAP_INTERP_TO_RESAMPLE(LINEAR, Linear);
      DALI_MAP_INTERP_TO_RESAMPLE(CUBIC, Cubic);
      DALI_MAP_INTERP_TO_RESAMPLE(LANCZOS3, Lanczos3);
      DALI_MAP_INTERP_TO_RESAMPLE(GAUSSIAN, Gaussian);
      DALI_MAP_INTERP_TO_RESAMPLE(TRIANGULAR, Triangular);
    default:
      DALI_FAIL("Unknown interpolation type");
    }
#undef DALI_MAP_INTERP_TO_RESAMPLE
  }

  struct KernelData {
    kernels::KernelContext context;
    kernels::KernelRequirements requirements;
    kernels::ScratchpadAllocator scratch_alloc;
  };
  std::vector<KernelData> kernel_data_;

  kernels::FilterDesc min_filter_{ kernels::ResamplingFilterType::Triangular, 0 };
  kernels::FilterDesc mag_filter_{ kernels::ResamplingFilterType::Linear, 0 };

  vector<kernels::ResamplingParams2D> resample_params_;
  USE_OPERATOR_MEMBERS();
  bool save_attrs_;
  int outputs_per_idx_;
  std::vector<Dims> out_shape_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_
