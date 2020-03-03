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

#include "dali/operators/image/crop/bbox_crop.h"

#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/pipeline/util/bounding_box.h"

namespace dali {

namespace {

void CollectShape(std::vector<TensorShape<>> &v,
                  const std::string &name,
                  const OpSpec& spec,
                  const workspace_t<CPUBackend>& ws,
                  int ndim,
                  bool required = false) {
  int batch_size = spec.GetArgument<int>("batch_size");
  v.clear();
  v.reserve(batch_size);

  if (spec.HasTensorArgument(name)) {
    auto arg_view = view<const int>(ws.ArgumentInput(name));
    DALI_ENFORCE(arg_view.num_samples() == batch_size, make_string(
      "Unexpected number of samples in argument `", name, "`: ", arg_view.num_samples(),
      ", expected: ", batch_size));
    DALI_ENFORCE(arg_view.sample_dim() == ndim, make_string(
      "Unexpected number of dimensions in argument `", name, "`: ", arg_view.sample_dim(),
      ", expected: ", ndim));

    std::vector<int64_t> tmp(ndim);
    for (int sample = 0; sample < batch_size; sample++) {
      const auto* sample_data = arg_view.tensor_data(sample);
      for (int d = 0; d < ndim; d++) {
        tmp[d] = static_cast<int64_t>(sample_data[d]);
      }
      v.emplace_back(tmp);
    }
  } else if (spec.HasArgument(name)) {
    auto tmp = spec.GetArgument<std::vector<int>>(name);
    DALI_ENFORCE(static_cast<int>(tmp.size()) == ndim,
      make_string("Argument `", name, "` must be a ", ndim, "D vector"));

    TensorShape<> sh(std::vector<int64_t>(tmp.begin(), tmp.end()));
    v.resize(batch_size, sh);
  } else {
    if (required)
      DALI_FAIL(make_string("Argument `", name, "` is required"));
  }
}

struct SampleOption {
  bool no_crop = false;
  float min_iou = 0.0f;
  SampleOption(bool _no_crop, float _min_iou)
      : no_crop(_no_crop), min_iou(_min_iou) {}
};

struct Range {
  Range(float _min, float _max)
      : min(_min), max(_max) {
    DALI_ENFORCE(min >= 0.0f,
      make_string("Min should be at least 0.0. Received: ", min));
    DALI_ENFORCE(min <= max,
      make_string("Range should be provided as: [min, max]. Received: [", min, max, "]"));
  }

  explicit Range(const std::vector<float>& range)
      : Range(range.size() == 2 ? range[0] : -1.0f,
              range.size() == 2 ? range[1] : -1.0f) {
  }

  bool Contains(float k) const {
    return k >= min && k <= max;
  }

  const float min, max;
};

struct ProspectiveCrop {
  bool success = false;
  BoundingBox crop{};
  std::vector<BoundingBox> boxes;
  std::vector<int> labels;

  ProspectiveCrop(bool success, const BoundingBox &crop, const std::vector<BoundingBox> &boxes,
                  const std::vector<int> &labels)
      : success(success), crop(crop), boxes(boxes), labels(labels) {}
  ProspectiveCrop() = default;
};

}  // namespace

DALI_SCHEMA(RandomBBoxCrop)
    .DocStr(
        R"code(Perform a prospective crop to an image while keeping bounding boxes and labels consistent. Inputs must be supplied as
two Tensors: `BBoxes` containing bounding boxes represented as `[l,t,r,b]` or `[x,y,w,h]`, and `Labels` containing the
corresponding label for each bounding box. Resulting prospective crop is provided as two Tensors: `Begin` containing the starting
coordinates for the `crop` in `(x,y)` format, and 'Size' containing the dimensions of the `crop` in `(w,h)` format.
Bounding boxes are provided as a `(m*4)` Tensor, where each bounding box is represented as `[l,t,r,b]` or `[x,y,w,h]`. Resulting
labels match the boxes that remain, after being discarded with respect to the minimum accepted intersection threshold.
Be advised, when `allow_no_crop` is `false` and `thresholds` does not contain `0` it is good to increase `num_attempts` as otherwise
it may loop for a very long time.)code")
    .NumInput(1, 2)  // [boxes, labels (optional),]
    .NumOutput(3)    // [anchor, shape, bboxes, labels (optional),]
    .AdditionalOutputsFn(
      [](const OpSpec& spec) {
        return spec.NumInput() - 1;  // +1 if labels are provided
      })
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum overlap (Intersection over union) of the bounding boxes with respect to the prospective crop.
Selected at random for every sample from provided values. Default imposes no restrictions on Intersection over Union for boxes and crop.)code",
        std::vector<float>{0.f})
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Range `[min, max]` of valid aspect ratio values for new crops. Value for `min` should be greater or equal to `0.0`.
Default values disallow changes in aspect ratio.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "scaling",
        R"code(Range `[min, max]` for crop size with respect to original image dimensions. Value for `min` should be greater or equal to `0.0`.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "ltrb",
        R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height].)code",
        true)
    .AddOptionalArg(
        "num_attempts",
        R"code(Number of attempts to get a crop window that matches the desired parameters.)code",
        1)
    .AddOptionalArg(
        "allow_no_crop",
        R"code(If true, includes no cropping as one of the random options.)code",
        true)
    .AddOptionalArg<int>(
        "crop_shape",
         R"code()code",
         std::vector<int>{},
         true)
    .AddOptionalArg<int>(
        "input_shape",
         R"code()code",
         std::vector<int>{},
         true);

template <>
class RandomBBoxCrop<CPUBackend>::Impl {
 public:
  ~Impl() = default;

  explicit Impl(const OpSpec &spec)
      : spec_(spec)
      , ltrb_{spec.GetArgument<bool>("ltrb")}
      , num_attempts_{spec.GetArgument<int>("num_attempts")}
      , has_labels_(spec.NumInput() > 1)
      , has_crop_shape_(spec.ArgumentDefined("crop_shape"))
      , has_input_shape_(spec.ArgumentDefined("input_shape"))
      , scale_range_{Range(spec.GetRepeatedArgument<float>("scaling"))}
      , aspect_ratio_range_{Range(spec.GetRepeatedArgument<float>("aspect_ratio"))}
      , rngs_(spec.GetArgument<int64_t>("seed"), spec.GetArgument<int>("batch_size")) {
    DALI_ENFORCE(has_crop_shape_ == has_input_shape_,
      "`crop_shape` and `input_shape` should be provided together or not provided");

    if (has_crop_shape_) {
      DALI_ENFORCE(!spec.HasArgument("allow_no_crop"),
        "`allow_no_crop` is incompatible with providing the crop shape explicitly");
      DALI_ENFORCE(!spec.HasArgument("aspect_ratio"),
        "`aspect_ratio` is incompatible with providing the crop shape explicitly");
      DALI_ENFORCE(!spec.HasArgument("scaling"),
        "`scaling` is incompatible with providing the crop shape explicitly");
    }

    auto thresholds = spec.GetRepeatedArgument<float>("thresholds");
    DALI_ENFORCE(!thresholds.empty(),
      "At least one threshold value must be provided");
    DALI_ENFORCE(num_attempts_ > 0,
      "Minimum number of attempts must be greater than zero");
    for (const auto &threshold : thresholds) {
      DALI_ENFORCE(0.0 <= threshold && threshold <= 1.0,
        make_string("Threshold value must be within the range [0.0, 1.0]. Received: ", threshold));
      sample_options_.emplace_back(false, threshold);
    }

    if (spec.GetArgument<bool>("allow_no_crop")) {
      sample_options_.emplace_back(true, 0.0f);
    }
  }

  bool Setup(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) {
    const auto &boxes = ws.template InputRef<CPUBackend>(0);
    auto tl_shape = boxes.shape();
    DALI_ENFORCE(tl_shape.sample_dim() == 2, make_string(
      "Unexpected number of dimensions for bounding boxes input: ", tl_shape.sample_dim()));
    // first dim is number of boxes, second is number of coordinates on each box
    auto ncoords = tl_shape[0][1];  // first sample, second dimension
    for (int sample = 0; sample < tl_shape.num_samples(); sample++) {
      auto sh = tl_shape[sample];
      DALI_ENFORCE(sh[1] == ncoords,
        make_string("Unexpected number of coordinates for sample ", sample, ". Expected ",
                    ncoords, ", got ", sh[1]));
    }
    DALI_ENFORCE(ncoords % 2 == 0,
      make_string("Unexpected number of coordinates for bounding boxes: ", ncoords));
    ndim_ = ncoords / 2;

    DALI_ENFORCE(ndim_ == 2 || ndim_ == 3,
      make_string("Unexpected number of dimensions: ", ndim_));

    if (spec_.ArgumentDefined("input_shape") && spec_.ArgumentDefined("crop_shape")) {
      CollectShape(input_shape_, "input_shape", spec_, ws, ndim_, true);
      CollectShape(crop_shape_, "crop_shape", spec_, ws, ndim_, true);
    }
    return false;
  }

  void Run(SampleWorkspace &ws) {
    const auto &boxes_tensor = ws.Input<CPUBackend>(0);
    auto nboxes = boxes_tensor.dim(0);
    auto ncoords = ndim_ * 2;
    std::vector<BoundingBox> bounding_boxes;
    bounding_boxes.reserve(nboxes);
    for (int i = 0; i < nboxes; i++) {
      const auto *box_data = boxes_tensor.data<float>() + i * ncoords;
      RelBounds bbox_coords;
      bbox_coords.resize(ncoords);
      for (int j = 0; j < ncoords; j++) {
        bbox_coords[j] = box_data[j];
      }

      auto box = ltrb_ ? BoundingBox::FromStartAndEnd(bbox_coords)
                       : BoundingBox::FromStartAndShape(bbox_coords);
      bounding_boxes.emplace_back(box);
    }

    std::vector<int> labels;
    if (has_labels_) {
      const auto &labels_tensor = ws.Input<CPUBackend>(1);
      auto nlabels = labels_tensor.dim(0);
      DALI_ENFORCE(nlabels == nboxes,
        make_string("Unexpected number of labels. Expected: ", nboxes, ", got ", nlabels));
      labels.resize(nlabels);
      const auto *label_data = labels_tensor.data<int>();
      for (int i = 0; i < nlabels; i++) {
        labels[i] = label_data[i];
      }
    }

    int sample = ws.data_idx();
    ProspectiveCrop prospective_crop;
    while (!prospective_crop.success) {
      prospective_crop = FindProspectiveCrop(bounding_boxes, labels, sample);
    }

    WriteCropToOutput(ws, prospective_crop.crop);

    const auto &selected_boxes  = prospective_crop.boxes;
    WriteBoxesToOutput(ws, selected_boxes);

    if (has_labels_) {
      const auto &selected_labels = prospective_crop.labels;
      DALI_ENFORCE(selected_boxes.size() == selected_labels.size(),
        make_string("Expected boxes.size() == labels.size(). Received: ", selected_boxes.size(),
          " != ", selected_labels.size()));
      WriteLabelsToOutput(ws, selected_labels);
    }
  }

 private:
  const ProspectiveCrop FindProspectiveCrop(const std::vector<BoundingBox> &bounding_boxes,
                                            const std::vector<int> &labels,
                                            int sample) {
    auto &rng = rngs_[sample];
    std::uniform_int_distribution<> idx_dist(0, sample_options_.size() - 1);
    SampleOption option = sample_options_[idx_dist(rng)];

    assert(has_crop_shape_ == has_input_shape_);
    bool absolute_crop_dims = has_crop_shape_;

    if (option.no_crop) {
      RelBounds no_crop;
      no_crop.resize(ndim_ * 2);
      for (int d = 0; d < ndim_; d++) {
        no_crop[d] = 0.0f;
        no_crop[ndim_ + d] = 1.0f;
      }
      if (absolute_crop_dims) {
        auto &input_shape = input_shape_[sample];
        for (int d = 0; d < ndim_; d++)
          no_crop[ndim_ + d] *= input_shape[d];
      }
      return ProspectiveCrop(true, BoundingBox::FromStartAndEnd(no_crop), bounding_boxes, labels);
    }

    RelBounds shape, anchor, out_bounds, rel_bounds;
    shape.resize(ndim_);
    anchor.resize(ndim_);
    out_bounds.resize(2 * ndim_);
    rel_bounds.resize(2 * ndim_);

    for (int i = 0; i < num_attempts_; i++) {
      if (absolute_crop_dims) {
        auto &crop_shape = crop_shape_[sample];
        auto &input_shape = input_shape_[sample];

        DALI_ENFORCE(crop_shape.sample_dim() == ndim_,
                     make_string("Unexpected number of dimensions. Expected ", ndim_, ", got ",
                                 crop_shape.sample_dim()));

        for (int d = 0; d < ndim_; d++) {
          shape[d] = static_cast<float>(crop_shape[d]);
          out_bounds[ndim_ + d] = shape[d];
          rel_bounds[ndim_ + d] = shape[d] / input_shape[d];
        }

        for (int d = 0; d < ndim_; d++) {
          assert(input_shape[d] >= crop_shape[d]);
          std::uniform_int_distribution<> anchor_dist(0, input_shape[d] - crop_shape[d]);
          anchor[d] = static_cast<float>(anchor_dist(rng));
          out_bounds[d] = anchor[d];
          rel_bounds[d] = anchor[d] / input_shape[d];
        }
      } else {  // relative dimensions
        std::uniform_real_distribution<float> extent_dist(scale_range_.min, scale_range_.max);
        for (int d = 0; d < ndim_; d++) {
          shape[d] = extent_dist(rng);
          out_bounds[ndim_ + d] = rel_bounds[ndim_ + d] = shape[d];
        }

        if (!ValidAspectRatio(shape)) {
          continue;
        }

        for (int d = 0; d < ndim_; d++) {
          std::uniform_real_distribution<float> anchor_dist(0.0f, 1.0f - shape[d]);
          anchor[d] = anchor_dist(rng);
          out_bounds[d] = rel_bounds[d] = anchor[d];
        }
      }

      const auto rel_crop = BoundingBox::FromStartAndShape(rel_bounds);
      const auto out_crop = BoundingBox::FromStartAndShape(out_bounds);

      if (!ValidOverlap(rel_crop, bounding_boxes, option.min_iou)) {
        continue;
      }

      auto filtered_bboxes = bounding_boxes;
      auto filtered_labels = labels;
      FilterByCentroid(rel_crop, filtered_bboxes, filtered_labels);
      if (filtered_bboxes.empty()) {
        continue;
      }

      for (auto &box : filtered_bboxes) {
        box = box.RemapTo(rel_crop);
      }

      return ProspectiveCrop(true, out_crop, filtered_bboxes, filtered_labels);
    }

    return ProspectiveCrop();
  }

  bool ValidAspectRatio(RelBounds shape) {
    assert(static_cast<int>(shape.size()) == ndim_);
    for (int i = 0; i < ndim_; i++) {
      for (int j = i + 1; j < ndim_; j++) {
        if (!aspect_ratio_range_.Contains(shape[i] / shape[j]))
          return false;
      }
    }
    return true;
  }

  bool ValidOverlap(const BoundingBox &crop, const std::vector<BoundingBox> &boxes,
                    float threshold) {
    return std::all_of(boxes.begin(), boxes.end(),
      [&crop, threshold](const BoundingBox &box) {
        return crop.IntersectionOverUnion(box) >= threshold;
      });
  }

  void FilterByCentroid(const BoundingBox &crop,
                        std::vector<BoundingBox> &bboxes,
                        std::vector<int> &labels) {
    std::vector<BoundingBox> new_bboxes;
    std::vector<int> new_labels;
    bool process_labels = !labels.empty();
    assert(labels.empty() || labels.size() == bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i++) {
      if (crop.Contains(bboxes[i].Centroid())) {
        new_bboxes.push_back(bboxes[i]);
        if (process_labels)
          new_labels.push_back(labels[i]);
      }
    }
    std::swap(bboxes, new_bboxes);
    if (process_labels)
      std::swap(labels, new_labels);
  }

  void WriteCropToOutput(SampleWorkspace &ws, const BoundingBox &crop) {
    const int ndim = crop.ndim();
    const auto coordinates = crop.AsStartAndShape();

    // output0 : anchor, output1 : shape
    auto &anchor_out = ws.Output<CPUBackend>(0);
    anchor_out.Resize({ndim});
    auto *anchor_out_data = anchor_out.mutable_data<float>();

    auto &shape_out = ws.Output<CPUBackend>(1);
    shape_out.Resize({ndim});
    auto *shape_out_data = shape_out.mutable_data<float>();

    for (int dim = 0; dim < ndim; dim++) {
      anchor_out_data[dim] = coordinates[dim];
      shape_out_data[dim]  = coordinates[ndim + dim];
    }
  }

  void WriteBoxesToOutput(SampleWorkspace &ws, const std::vector<BoundingBox> &bounding_boxes) {
    int box_size = 2 * ndim_;
    auto &bbox_out = ws.Output<CPUBackend>(2);
    bbox_out.Resize({static_cast<int64_t>(bounding_boxes.size()), box_size});
    auto *bbox_out_data = bbox_out.mutable_data<float>();

    for (size_t i = 0; i < bounding_boxes.size(); ++i) {
      auto *output = bbox_out_data + i * box_size;
      assert(bounding_boxes[i].ndim() == ndim_);
      auto coordinates =
          ltrb_ ? bounding_boxes[i].AsStartAndEnd() : bounding_boxes[i].AsStartAndShape();
      for (int j = 0; j < box_size; j++)
        output[j] = coordinates[j];
    }
  }

  void WriteLabelsToOutput(SampleWorkspace &ws, const std::vector<int> &labels) {
    auto &labels_out = ws.Output<CPUBackend>(3);
    labels_out.Resize({static_cast<Index>(labels.size()), 1});
    auto *labels_out_data = labels_out.mutable_data<int>();
    for (size_t i = 0; i < labels.size(); i++) {
      labels_out_data[i] = labels[i];
    }
  }

 private:
  OpSpec spec_;
  bool ltrb_;
  int num_attempts_;
  bool has_labels_;
  bool has_crop_shape_;
  bool has_input_shape_;

  Range scale_range_;
  Range aspect_ratio_range_;
  BatchRNG<std::mt19937> rngs_;

  std::vector<SampleOption> sample_options_;

  std::vector<TensorShape<>> crop_shape_;
  std::vector<TensorShape<>> input_shape_;
  int ndim_ = -1;
};

template <>
RandomBBoxCrop<CPUBackend>::~RandomBBoxCrop() = default;

template <>
RandomBBoxCrop<CPUBackend>::RandomBBoxCrop(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {
  impl_ = std::make_unique<Impl>(spec);
}

template <>
bool RandomBBoxCrop<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                           const workspace_t<CPUBackend> &ws) {
  assert(impl_ != nullptr);
  return impl_->Setup(output_desc, ws);
}

template <>
void RandomBBoxCrop<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  assert(impl_ != nullptr);
  impl_->Run(ws);
}

DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<CPUBackend>, CPU);

}  // namespace dali
