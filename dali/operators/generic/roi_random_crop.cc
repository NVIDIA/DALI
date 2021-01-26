// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <random>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(ROIRandomCrop)
    .DocStr(R"code(Produces a fixed shape cropping window, randomly placed so that as much of the
provided region of interest (ROI) is contained in it.
If the ROI is bigger than the cropping window, the cropping window will be a subwindow of the ROI.
If the ROI is smaller than the cropping window, the whole ROI should be contained in the cropping window.

If an input shape (``in_shape``) is given, the resulting cropping window is selected to be within the
bounds of that input shape. Alternatively, the input data subject to cropping can be fed directly to
the operator.

If no input shape is provided, the resulting cropping window is unbounded, potentially resulting in out 
of bounds cropping.

The cropping window dimensions should be explicitly provided (``crop_shape``), and the ROI should be
either specified with ``roi_start``/``roi_end`` or ``roi_start``/``roi_shape``.

The operator produces an output representing the cropping window start coordinates.
)code")
    .AddArg("crop_shape",
      R"code(Cropping window dimensions.)code", DALI_INT_VEC, true)
    .AddArg("roi_start",
      R"code(ROI start coordinates.)code", DALI_INT_VEC, true)
    .AddOptionalArg<std::vector<int>>("roi_end",
      R"code(ROI end coordinates.

Note: Using ``roi_end`` is mutually exclusive with ``roi_shape``.)code", nullptr, true)
    .AddOptionalArg<std::vector<int>>("roi_shape",
      R"code(ROI shape.

Note: Using ``roi_shape`` is mutually exclusive with ``roi_end``.)code", nullptr, true)
    .AddOptionalArg<std::vector<int>>("in_shape",
      R"code(Shape of the input data.

If provided, the cropping window start will be selected so that the cropping window is within the
bounds of the input.

Note: Providing ``in_shape`` is incompatible with feeding the input data directly as a positional input.
)code", nullptr, true)
    .NumInput(0, 1)
    .NumOutput(1);

class ROIRandomCropCPU : public Operator<CPUBackend> {
 public:
  explicit ROIRandomCropCPU(const OpSpec &spec);
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  BatchRNG<std::mt19937> rngs_;

  ArgValue<int, 1> roi_start_;
  ArgValue<int, 1> roi_end_;
  ArgValue<int, 1> roi_shape_;
  ArgValue<int, 1> crop_shape_;
  ArgValue<int, 1> in_shape_arg_;

  TensorListShape<> in_shape_;

  USE_OPERATOR_MEMBERS();
};

ROIRandomCropCPU::ROIRandomCropCPU(const OpSpec &spec)
    : Operator<CPUBackend>(spec),
      rngs_(spec.GetArgument<int64_t>("seed"), spec.GetArgument<int64_t>("max_batch_size")),
      roi_start_("roi_start", spec),
      roi_end_("roi_end", spec),
      roi_shape_("roi_shape", spec),
      crop_shape_("crop_shape", spec),
      in_shape_arg_("in_shape", spec) {
  if ((roi_end_.IsDefined() + roi_shape_.IsDefined()) != 1)
    DALI_FAIL("Either ROI end or ROI shape should be defined, but not both");
}

bool ROIRandomCropCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<CPUBackend> &ws) {
  int nsamples = spec_.HasTensorArgument("crop_shape") ?
                     ws.ArgumentInput("crop_shape").size() :
                     ws.GetRequestedBatchSize(0);
  crop_shape_.Acquire(spec_, ws, nsamples, true);
  int ndim = crop_shape_[0].shape[0];

  TensorShape<1> sh{ndim};
  roi_start_.Acquire(spec_, ws, nsamples, sh);
  if (roi_end_.IsDefined()) {
    roi_end_.Acquire(spec_, ws, nsamples, sh);
  } else {
    assert(roi_shape_.IsDefined());
    roi_shape_.Acquire(spec_, ws, nsamples, sh);
  }

  in_shape_.shapes.clear();
  if (in_shape_arg_.IsDefined() || ws.NumInput() == 1) {
    if (in_shape_arg_.IsDefined() && (ws.NumInput() == 1)) {
      DALI_FAIL("``in_shape`` argument is incompatible with providing an input.")
    }
    if (in_shape_arg_.IsDefined()) {
      in_shape_.resize(nsamples, ndim);
      in_shape_arg_.Acquire(spec_, ws, nsamples, sh);
      for (int s = 0; s < nsamples; s++) {
        auto sample_sh = in_shape_.tensor_shape_span(s);
        for (int d = 0; d < ndim; d++) {
          sample_sh[d] = in_shape_arg_[s].data[d];
        }
      }
    } else {
      auto &in = ws.template InputRef<CPUBackend>(0);
      in_shape_ = in.shape();
    }

    for (int s = 0; s < nsamples; s++) {
      auto sample_sh = in_shape_.tensor_shape_span(s);
      for (int d = 0; d < ndim; d++) {
        DALI_ENFORCE(sample_sh[d] >= crop_shape_[s].data[d],
                     "Cropping shape can't be bigger than the input shape.");
      }
      if (roi_shape_.IsDefined()) {
        for (int d = 0; d < ndim; d++) {
          DALI_ENFORCE(roi_start_[s].data[d] >= 0 &&
                       sample_sh[d] >= (roi_start_[s].data[d] + roi_shape_[s].data[d]),
                       "ROI can't be out of bounds.");
        }
      } else {
        for (int d = 0; d < ndim; d++) {
          DALI_ENFORCE(roi_start_[s].data[d] >= 0 && sample_sh[d] >= roi_end_[s].data[d],
                       "ROI can't be out of bounds.");
        }
      }
    }
  }

  output_desc.resize(1);
  output_desc[0].shape = uniform_list_shape(nsamples, sh);
  output_desc[0].type = TypeTable::GetTypeInfo(DALI_INT64);
  return true;
}

void ROIRandomCropCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  auto &out_crop_start = ws.template OutputRef<CPUBackend>(0);
  auto crop_start = view<int64_t, 1>(out_crop_start);

  int nsamples = crop_start.shape.size();
  int ndim = crop_start[0].shape[0];

  auto& thread_pool = ws.GetThreadPool();

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    thread_pool.AddWork(
      [&, ndim, sample_idx](int thread_id) {
        int64_t* sample_sh = nullptr;
        if (!in_shape_.empty())
          sample_sh = in_shape_.tensor_shape_span(sample_idx).data();
        for (int d = 0; d < ndim; d++) {
          int64_t roi_extent = -1;
          int64_t roi_start = roi_start_[sample_idx].data[d];
          int64_t crop_extent = crop_shape_[sample_idx].data[d];
          if (roi_end_.IsDefined()) {
            roi_extent = roi_end_[sample_idx].data[d] - roi_start;
          } else {
            roi_extent = roi_shape_[sample_idx].data[d];
          }

          if (roi_extent == crop_extent) {
            crop_start[sample_idx].data[d] = roi_start;
          } else if (roi_extent > crop_extent) {
            int64_t range_end = roi_start + roi_extent - crop_extent;
            if (sample_sh)
              range_end = std::min<int64_t>(sample_sh[d] - crop_extent, range_end);
            auto dist = std::uniform_int_distribution<int64_t>(roi_start, range_end);
            crop_start[sample_idx].data[d] = dist(rngs_[sample_idx]);
          } else {  // roi_extent < crop_extent
            int64_t range_start = roi_start + roi_extent - crop_extent;
            if (sample_sh)
              range_start = std::max<int64_t>(0, range_start);
            auto dist = std::uniform_int_distribution<int64_t>(range_start, roi_start);
            crop_start[sample_idx].data[d] = dist(rngs_[sample_idx]);
          }
        }
      }, sample_idx);
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(ROIRandomCrop, ROIRandomCropCPU, CPU);

}  // namespace dali
