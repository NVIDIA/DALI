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

#include "dali/operators/bbox/bb_flip.h"
#include <iterator>
#include "dali/core/geom/box.h"
#include "dali/pipeline/util/bounding_box_utils.h"

namespace dali {

DALI_REGISTER_OPERATOR(BbFlip, BbFlipCPU, CPU);

DALI_SCHEMA(BbFlip)
    .DocStr(R"code(Flips bounding boxes horizontaly or verticaly (mirror).

The bounding box coordinates for the  input are in the [x, y, width, height] - ``xywh`` or
[left, top, right, bottom] - ``ltrb`` format. All coordinates are in the image coordinate
system, that is 0.0-1.0)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("ltrb",
                    R"code(True for ``ltrb`` or False for ``xywh``.)code",
                    false, false)
    .AddOptionalArg("horizontal",
                    R"code(Flip horizontal dimension.)code",
                    1, true)
    .AddOptionalArg("vertical",
                    R"code(Flip vertical dimension.)code",
                    0, true);

void BbFlipCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_view = view<const float>(input);
  auto out_view = view<float>(output);
  auto nsamples = in_view.shape.size();
  auto &tp = ws.GetThreadPool();
  constexpr int box_size = Box<2, float>::size;
  TensorLayout layout = ltrb_ ? "xyXY" : "xyWH";

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    tp.AddWork(
      [&, sample_idx](int thread_id) {
        bool vertical = vert_[sample_idx].data[0];
        bool horizontal = horz_[sample_idx].data[0];
        std::vector<Box<2, float>> bboxes;
        const auto &in_sample = in_view[sample_idx];
        const auto &out_sample = out_view[sample_idx];
        assert(in_sample.shape.num_elements() % box_size == 0);
        int nboxes = in_sample.shape.num_elements() / box_size;
        bboxes.resize(nboxes);
        ReadBoxes(make_span(bboxes),
                  make_cspan(in_sample.data, in_sample.shape.num_elements()),
                  layout, {});
        for (auto &bbox : bboxes) {
          if (horizontal)
            HorizontalFlip(bbox);
          if (vertical)
            VerticalFlip(bbox);
        }
        WriteBoxes(make_span(out_sample.data, out_sample.shape.num_elements()),
                   make_cspan(bboxes),
                   layout);
      }, in_view.shape.tensor_size(sample_idx));
  }
  tp.RunAll();
}

}  // namespace dali
