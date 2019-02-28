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


#include <vector>
#include <random>
#include <utility>
#include <algorithm>

#include "dali/pipeline/operators/detection/random_crop.h"
#include "dali/pipeline/operators/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(SSDRandomCrop)
  .DocStr(R"code(Perform a random crop with bounding boxes
where IoU meets randomly selected threshold between 0-1.
When IoU falls below threshold new random crop is generated up to num_attempts.
As an input, it accepts image, bounding boxes and labels. At the output
cropped image, cropped and valid bounding boxes and valid labels are returned.)code")
  .NumInput(3)   // [img, bbox, label]
  .NumOutput(3)  // [img, bbox, label]
  .AddOptionalArg("num_attempts", R"code(Number of attempts.)code", 1);

/*
 * # This function is from https://github.com/kuangliu/pytorch-ssd.
 * def calc_iou_tensor(box1, box2):
 *    """ Calculation of IoU based on two boxes tensor,
 *        Reference to https://github.com/kuangliu/pytorch-ssd
 *        input:
 *            box1 (N, 4)
 *            box2 (M, 4)
 *        output:
 *            IoU (N, M)
 *    """
 *    N = box1.size(0)
 *    M = box2.size(0)
 *
 *    be1 = box1.unsqueeze(1).expand(-1, M, -1)
 *    be2 = box2.unsqueeze(0).expand(N, -1, -1)
 *
 *    # Left Top & Right Bottom
 *    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
 *    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
 *
 *    delta = rb - lt
 *    delta[delta < 0] = 0
 *    intersect = delta[:,:,0]*delta[:,:,1]
 *
 *    delta1 = be1[:,:,2:] - be1[:,:,:2]
 *    area1 = delta1[:,:,0]*delta1[:,:,1]
 *    delta2 = be2[:,:,2:] - be2[:,:,:2]
 *    area2 = delta2[:,:,0]*delta2[:,:,1]
 *
 *    iou = intersect/(area1 + area2 - intersect)
 *    return iou
 */

namespace detail {

// input in ltrb format
// box1 is [N, 4], box2 is [M, 4]
// calculate IoU of every box1 vs. every box2 : O(N^2)
Tensor<CPUBackend> cpu_iou(const Tensor<CPUBackend>& box1,
                           const Tensor<CPUBackend>& box2) {
  Tensor<CPUBackend> ious;
  ious.set_pinned(false);

  int N = box1.dim(0);
  // Note: We know M=1 in this use-case
  // int M = box2.dim(0);
  const int M = 1;

  const float* box1_data = box1.data<float>();
  const float* box2_data = box2.data<float>();

  ious.Resize({N, 1});
  float *ious_data = ious.mutable_data<float>();

  std::vector<std::pair<float, float>> lt, rb;

  /*
   * # Left Top & Right Bottom
   * lt = torch.max(be1[:,:,:2], be2[:,:,:2])
   * rb = torch.min(be1[:,:,2:], be2[:,:,2:])
   */
  for (int i = 0; i < N; ++i) {
    const float *b1 = box1_data + i * box1.dim(1);
    const float *b2 = box2_data;

    // want the maximum top, left
    float l = std::max(b1[0], b2[0]);
    float t = std::max(b1[1], b2[1]);

    // minimum bottom, right
    float r = std::min(b1[2], b2[2]);
    float b = std::min(b1[3], b2[3]);

    lt.push_back(std::make_pair(l, t));
    rb.push_back(std::make_pair(r, b));
  }

  // delta = rb - lt
  // delta[delta < 0] = 0
  // intersect = delta[:,:,0] * delta[:,:, 1]
  vector<float> intersect(N);
  for (int i = 0; i < N; ++i) {
    float first_elem = rb[i].first - lt[i].first;
    float second_elem = rb[i].second - lt[i].second;

    // delta[delta < 0] = 0
    first_elem = (first_elem < 0) ? 0 : first_elem;
    second_elem = (second_elem < 0) ? 0 : second_elem;
    float inter = first_elem * second_elem;

    intersect[i] = inter;
  }

  // delta1 = be1[:, :, 2:] - be1[:, :, :2]
  // area1 = delta1[:, :, 0] * delta[:, :, 1]
  vector<float> area1(N);
  for (int i = 0; i < N; ++i) {
    const float* box = box1_data + i * 4;
    // area is (b-t) * (r-l)
    area1[i] = (box[3] - box[1]) * (box[2] - box[0]);
  }
  // delta2 = be2[:, :, 2:] - be2[:, :, :2]
  // area2 = delta2[:, :, 0] * delta[:, :, 2]
  const float* box = box2_data;
  // area is (b-t) * (r-l)
  auto area2 = (box[3] - box[1]) * (box[2] - box[0]);

  // iou = intersect / (area1 + area2 - intersect)
  for (int i = 0; i < N; ++i) {
    // index into N*M arrays
    auto idx = i;
    ious_data[idx] = intersect[idx] / (area1[i] + area2 - intersect[idx]);
  }
  return ious;
}

// img is [H, W, C], bounds [l, t, r, b]
// output [r-l, b-t, C]
void crop(const Tensor<CPUBackend>& img, vector<int> bounds, Tensor<CPUBackend>& out) {
  // output dimensions
  const int width = bounds[2] - bounds[0];
  const int height = bounds[3] - bounds[1];
  // input dimensions
  const int H = img.dim(0);
  const int W = img.dim(1);
  const int C = img.dim(2);

  out.Resize({height, width, C});
  uint8_t *out_data = out.mutable_data<uint8_t>();

  int out_idx = 0;
  for (int h = bounds[1]; h < bounds[3]; ++h) {
    const int idx = h * W * C + bounds[0] * C;
    memcpy(out_data + out_idx, img.data<uint8_t>() + idx, (bounds[2] - bounds[0]) * C);
    out_idx += (bounds[2] - bounds[0]) * C;
  }
}

}  // namespace detail

template <>
void SSDRandomCrop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  // [H, W, C], dtype=uint8_t
  const auto& img = ws->Input<CPUBackend>(0);
  // [N] : [ltrb, ... ], dtype=float
  const auto& bboxes = ws->Input<CPUBackend>(1);
  const auto& labels = ws->Input<CPUBackend>(2);

  auto N = bboxes.dim(0);
  const float* bbox_data = bboxes.data<float>();

  const int* label_data = labels.data<int>();

  // [1x4]
  Tensor<CPUBackend> crop_attempt;
  crop_attempt.set_pinned(false);
  crop_attempt.Resize({1, 4});
  float *crop_ptr = crop_attempt.mutable_data<float>();

  // iterate until a suitable crop has been found
  while (true) {
    auto opt_idx = int_dis_(gen_);
    auto option = sample_options_[opt_idx];

    if (option.no_crop()) {
      // copy directly to output without modification
      ws->Output<CPUBackend>(0).Copy(img, 0);
      ws->Output<CPUBackend>(1).Copy(bboxes, 0);
      ws->Output<CPUBackend>(2).Copy(labels, 0);
      return;
    }

    // input is HWC ordering
    auto htot = img.dim(0);
    auto wtot = img.dim(1);

    auto min_iou = option.min_iou();

    // make num_attempts_ tries to get a valid crop
    for (int i = 0; i < num_attempts_; ++i) {
      auto w = float_dis_(gen_);
      auto h = float_dis_(gen_);
      // aspect ratio check
      if ((w / h < 0.5) || (w / h > 2.)) {
        continue;
      }

      // need RNG generators for left, top
      std::uniform_real_distribution<float> l_dis(0., 1. - w), t_dis(0., 1. - h);
      auto left = l_dis(gen_);
      auto top = t_dis(gen_);

      auto right = left + w;
      auto bottom = top + h;

      crop_ptr[0] = left;
      crop_ptr[1] = top;
      crop_ptr[2] = right;
      crop_ptr[3] = bottom;

      // returns ious : [N, M]

      Tensor<CPUBackend> ious = detail::cpu_iou(bboxes, crop_attempt);
      const float *ious_data = ious.data<float>();

      // make sure all the calculated IoUs are in the range (min_iou, max_iou)
      // Note: ious has size N*M, but M = 1 in this case
      bool fail = false;
      for (int j = 0; j < N * 1; ++j) {
        if (ious_data[j] <= min_iou) fail = true;
      }
      // generate a new crop
      if (fail) {
        continue;
      }

      // discard any bboxes whose center is not in the cropped image
      int valid_bboxes = 0;
      std::vector<int> mask;
      for (int j = 0; j < N; ++j) {
        const auto* bbox = bbox_data + j * 4;
        auto xc = 0.5*(bbox[0] + bbox[2]);
        auto yc = 0.5*(bbox[1] + bbox[3]);

        bool valid = (xc > left) && (xc < right) && (yc > top) && (yc < bottom);
        if (valid) {
          mask.push_back(j);
          valid_bboxes += 1;
        }
      }
      // If we don't have any valid boxes, generate a new crop
      if (!valid_bboxes) {
        continue;
      }

      // now we know how many output bboxes there will be, we can allocate
      // the output.
      auto &img_out = ws->Output<CPUBackend>(0);
      auto &bbox_out = ws->Output<CPUBackend>(1);
      auto &label_out = ws->Output<CPUBackend>(2);

      bbox_out.Resize({valid_bboxes, 4});
      auto *bbox_out_data = bbox_out.mutable_data<float>();

      label_out.Resize({valid_bboxes, 1});
      auto *label_out_data = label_out.mutable_data<int>();

      // copy valid bboxes to output and transform them
      for (int j = 0; j < valid_bboxes; ++j) {
        int idx = mask[j];

        // this bbox is being preserved
        const auto *bbox_i = bbox_data + idx * 4;
        auto *bbox_o = bbox_out_data + j * 4;

        // copy across (with clamping)
        bbox_o[0] = (bbox_i[0] < left) ? left : bbox_i[0];
        bbox_o[1] = (bbox_i[1] < top) ? top : bbox_i[1];
        bbox_o[2] = (bbox_i[2] > right) ? right : bbox_i[2];
        bbox_o[3] = (bbox_i[3] > bottom) ? bottom : bbox_i[3];
        // bbox_o[4] = bbox_i[4];
        label_out_data[j] = label_data[idx];

        // scaling
        float minus[] = {left, top, left, top};
        float scale[] = {w, h, w, h};
        for (int k = 0; k < 4; ++k) {
          bbox_o[k] = (bbox_o[k] - minus[k]) / scale[k];
        }
      }  // end bbox copy

      // everything is good, generate the crop parameters
      const int left_idx = static_cast<int>(left * wtot);
      const int top_idx = static_cast<int>(top * htot);
      const int right_idx = static_cast<int>(right * wtot);
      const int bottom_idx = static_cast<int>(bottom * htot);

      // perform the crop
      detail::crop(img, {left_idx, top_idx, right_idx, bottom_idx},
                   ws->Output<CPUBackend>(0));

      return;
    }  // end num_attempts loop
  }  // end sample loop
}

template <>
void SSDRandomCrop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  return;
}

DALI_REGISTER_OPERATOR(SSDRandomCrop, SSDRandomCrop<CPUBackend>, CPU);

}  // namespace dali
