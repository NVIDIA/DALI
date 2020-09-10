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
#include "dali/core/geom/box.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/pipeline/util/bounding_box_utils.h"

namespace dali {

namespace {

// This is the default shape layout that the operator uses internally
TensorLayout InternalShapeLayout(int ndim) {
  assert(ndim == 3 || ndim == 2);
  return ndim == 3 ? "WHD" : "WH";
}

void CollectShape(std::vector<TensorShape<>> &v,
                  const std::string &name,
                  const OpSpec& spec,
                  const workspace_t<CPUBackend>& ws,
                  int ndim) {
  int batch_size = spec.GetArgument<int>("batch_size");
  v.clear();
  v.reserve(batch_size);

  if (spec.HasTensorArgument(name)) {
    auto arg_view = view<const int>(ws.ArgumentInput(name));
    DALI_ENFORCE(arg_view.num_samples() == batch_size, make_string(
      "Unexpected number of samples in argument `", name, "`: ", arg_view.num_samples(),
      ", expected: ", batch_size));

    std::vector<int64_t> tmp(ndim);
    for (int sample = 0; sample < batch_size; sample++) {
      auto shape_len = volume(arg_view.tensor_shape(sample));
      DALI_ENFORCE(shape_len == ndim, make_string(
        "Unexpected number of elements in argument `", name, "`: ", shape_len,
        ", expected: ", ndim));

      const auto* sample_data = arg_view.tensor_data(sample);
      for (int d = 0; d < ndim; d++) {
        tmp[d] = static_cast<int64_t>(sample_data[d]);
      }
      v.emplace_back(tmp);
    }
  } else if (spec.HasArgument(name)) {
    auto tmp = spec.GetRepeatedArgument<int>(name);
    DALI_ENFORCE(static_cast<int>(tmp.size()) == ndim,
      make_string("Argument `", name, "` must be a ", ndim, "D vector"));

    TensorShape<> sh(std::vector<int64_t>(tmp.begin(), tmp.end()));
    v.resize(batch_size, sh);
  } else {
    DALI_FAIL(make_string("Argument `", name, "` was not found"));
  }
}

struct SampleOption {
  bool no_crop = false;
  float threshold = 0.0f;
};

struct Range {
  bool Contains(float k) const {
    assert(min <= max);
    return k >= min && k <= max;
  }
  float min = 0.0f, max = 0.0f;
};

}  // namespace

DALI_SCHEMA(RandomBBoxCrop)
    .DocStr(
        R"code(Applies a prospective random crop to an image coordinate space while keeping
bounding boxes (and optionally labels) consistent. That is, after applying the random crop operator
to the image coordinate space, the bounding boxes will be adjusted or filtered out to match the
cropped region of interest. The applied random crop operation is constrained by the arguments
provided to the operator.

The cropping window candidates are selected randomly until one matches the overlap restrictions
specified by ``thresholds`` argument. Thresholds values represent a minimum overlap metric, specified
by ``threshold_type``, such as intersection-over-union of the cropping window and the bounding boxes
or relative overlap as the ratio of the intersection area and the area of the bounding box.

Alternatively, we can allow not to crop as one of the valid outcomes of the random process
(use ``allow_no_crop`` for that).

Two modes of random crop are available:

Randomly shaped window, randomly placed within the original input space. The random crop window
dimensions are selected according to the provided ``aspect_ratio`` and relative area restrictions.

Fixed size window, randomly placed within the original input space. The random crop window
dimensions are taken from the ``crop_shape`` argument and the anchor is selected randomly.
When providing ``crop_shape``, a second argument ``input_shape`` should be provided, specifying the
original image dimensions, which is required to scale the output bounding boxes.

The argument ``num_attempts`` can be used to control the maximum number of attempts that we try to
produce a valid crop to match a single minimum overlap metric value from ``thresholds``. Be advised that when
``allow_no_crop`` is ``False`` and ``thresholds`` does not contain ``0.0``, it is good to increase the
``num_attempts`` value as otherwise it may loop for a very long time.

Inputs: 0: ``bboxes``, (1:``labels``, )

The first input, ``bboxes``, refers to bounding boxes that are provided as a 2-dimensional tensor where
the first dimension refers to the index of the bounding box and the second dimension refers to the
index of the coordinate.
Coordinates are relative to the original image dimensions (i.e. range [0.0, 1.0]), representing
either ``start`` and ``end`` of the region or ``start`` and shape` depending on the value of ``bbox_layout``.
E.g. ``bbox_layout="xyXY"`` means the bounding box coordinates are following the order
``start_x, start_y, end_x, end_y``, while ``bbox_layout="xyWH"`` indicates the order is
``start_x, start_y, width, height``. See more in ``bbox_layout`` argument description.

A second input ``labels`` can be optionally provided, representing the labels associated with each of
the bounding boxes.

Outputs: 0:``anchor``, 1:``shape``, 2:``bboxes``, (3:``labels``,)

The resulting crop parameters are provided as two separate outputs ``anchor`` and
``shape``, that can be fed directly to ``Slice`` operator to perform the actual cropping of the
original image. ``anchor`` and ``shape`` contain starting coordinates and dimensions for the crop, as
``[x, y, (z,)]`` and ``[w, h, (d,)]`` respectively. The coordinates can be represented in absolute or
relative terms, depending of whether the fixed ``crop_shape`` was used or not.

The third and fourth output correspond to the adjusted bounding boxes and optionally
their corresponding labels. Bounding boxes are always specified in relative coordinates.)code")
    .NumInput(1, 2)  // [boxes, labels (optional),]
    .InputDox(
        0, "boxes", "2D TensorList of float",
        "Relative coordinates of the bounding boxes represented as a 2D tensor where the first "
        "dimension refers to the index of the bounding box and the second dimension refers to the "
        "index of the coordinate.")
    .InputDox(1, "labels", "1D TensorList of integers",
              "(optional) Labels associated with each of the bounding boxes.")
    .NumOutput(3)  // [anchor, shape, bboxes, labels (optional),]
    .AdditionalOutputsFn([](const OpSpec &spec) {
      return spec.NumRegularInput() - 1;  // +1 if labels are provided
    })
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum intersection-over-union (IoU), or a different metric if specified by ``threshold_type``,
of the bounding boxes with respect to the cropping window. Each sample will select one of the ``thresholds``
randomly, and the operator will try a given number of attempts (see ``num_attempts``) to produce a random
crop window that has the metric above the selected threshold.)code",
        std::vector<float>{0.f})
    .AddOptionalArg(
        "threshold_type",
        R"code(Determines the meaning of ``thresholds``. By default refers to intersection-over-union (IoU) of
the bounding boxes with respect to the cropping window. Alternatively, it could be set to ``overlap``
to specify the fraction (by area) of the the bounding box which will fall inside the crop window
(e.g. a threshold ``1.0`` means the whole bounding box must be contained in the resulting cropping window))code",
        "iou")
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Single or multiple valid aspect ratio ranges for cropping windows.

For 2D bounding boxes, a single aspect ratio ``x/y`` range should be provided (i.e. ``[min_xy, max_xy]``)

For 3D bounding boxes, three aspect ratio ranges are expected (``x/y``, ``x/z`` and ``y/z``) ranges, i.e
``[min_xy, max_xy, min_xz, max_xz, min_yz, max_yz]``.
Alternatively, if a single aspect ratio range is provided, the valid aspect ratio range will be the
same for the three ratios.

Value for ``min`` should be ``> 0.0`` and ``min <= max``.

By default, square windows are generated.

Note: Providing ``aspect_ratio`` and ``scaling`` is incompatible with specifying ``crop_shape``
explicitly.)code",
                    std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "scaling",
        R"code(Range ``[min, max]`` for crop size with respect to original image dimensions.

Value for ``min`` should satisfy ``0.0 <= min <= max``.

Note: Providing ``aspect_ratio`` and ``scaling`` is incompatible with specifying `crop_shape`
explicitly)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "ltrb",
        R"code(If true, bboxes are returned as ``[left, top, right, bottom]``,
otherwise they are provided as ``[left, top, width, height]``.

WARNING: This argument is deprecated. Use ``bbox_layout`` instead to specify the bbox encoding.
E.g ``ltrb=True`` is equivalent to ``bbox_layout="xyXY"`` and ``ltrb=False`` corresponds to
``bbox_layout="xyWH"``)code",
        true)
    .AddOptionalArg(
        "num_attempts",
        R"code(Number of attempts to get a crop window that matches the ``aspect_ratio`` and a selected
value from ``thresholds``.

After each ``num_attempts``, a different threshold will be picked, until either reaching a maximum of
``total_num_attempts`` (if provided), or indefinitely otherwise.)code",
        1)
    .AddOptionalArg(
        "total_num_attempts",
        R"code(If provided, it indicates the total maximum number of attempts to get a crop window that
matches the ``aspect_ratio`` and any selected value from ``thresholds``.

After ``total_num_attempts`` the best candidate will be chosen as a default.

If not specified, the crop search will continue indefinitely until finding a valid crop.

WARNING: Not providing ``total_num_attempts`` could lead to an infinite loop if the provided arguments
can't be statisfied.)code",
        -1)
    .AddOptionalArg(
        "all_boxes_above_threshold",
         R"code(If true, all bounding boxes in a sample should overlap with the cropping window as specified by
``thresholds``, otherwise the cropping window is considered invalid. If false, a cropping window will be considered
valid if any bounding box overlaps it sufficiently.)code",
         true)
    .AddOptionalArg(
        "allow_no_crop",
        R"code(If true, not cropping will be one of the possible outcomes of the random process, as
if it was one more ``thresholds`` value to choose from.)code",
        true)
    .AddOptionalArg<int>(
        "crop_shape",
        R"code(If provided, the random crop window dimensions will be fixed to this shape.

The order of dimensions is determined by the layout provided in ``shape_layout``.

Note: ``crop_shape`` and ``input_shape`` should be provided together, and providing those is
incompatible with using ``scaling`` and ``aspect_ratio`` arguments.)code",
        std::vector<int>{}, true)
    .AddOptionalArg<int>(
        "input_shape",
        R"code(Specifies the shape of the original input image.

The order of dimensions is determined by the layout provided in ``shape_layout``.

Note: ``crop_shape`` and ``input_shape`` should be provided together, and providing those is
incompatible with using ``scaling`` and ``aspect_ratio`` arguments.)code",
        std::vector<int>{}, true)
    .AddOptionalArg<TensorLayout>(
        "bbox_layout",
        R"code(Determines the meaning of the coordinates of the bounding boxes.
  Possible values are:

  ``x`` (horizontal start anchor), ``y`` (vertical start anchor), ``z`` (depthwise start anchor),

  ``X`` (horizontal end anchor),   ``Y`` (vertical end anchor),   ``Z`` (depthwise end anchor),

  ``W`` (width),                   ``H`` (height),                ``D`` (depth).

Note: If left empty, ``"xyXY"`` or ``"xyzXYZ"`` will be assumed, depending on the number of dimensions.
)code",
        TensorLayout{""})
    .AddOptionalArg<TensorLayout>(
        "shape_layout",
        R"code(Determines the meaning of the dimensions provided in `crop_shape` and `input_shape`.
Possible values are:

  ``W`` (width), ``H`` (height), ``D`` (depth).

Note: If left empty, ``"WH"`` or ``"WHD"`` will be assumed, depending on the number of dimensions.
)code",
        TensorLayout{""});

template <int ndim>
class RandomBBoxCropImpl : public OpImplBase<CPUBackend> {
 public:
  static constexpr int coords_size = ndim * 2;

  enum OverlapMetric {
    IoU = 1,
    Overlap = 2
  };

  ~RandomBBoxCropImpl() = default;

  explicit RandomBBoxCropImpl(const OpSpec &spec)
      : spec_(spec),
        num_attempts_{spec.GetArgument<int>("num_attempts")},
        has_labels_(spec.NumRegularInput() > 1),
        has_crop_shape_(spec.ArgumentDefined("crop_shape")),
        has_input_shape_(spec.ArgumentDefined("input_shape")),
        bbox_layout_(spec.GetArgument<TensorLayout>("bbox_layout")),
        shape_layout_(spec.GetArgument<TensorLayout>("shape_layout")),
        all_boxes_above_threshold_(spec.GetArgument<bool>("all_boxes_above_threshold")),
        rngs_(spec.GetArgument<int64_t>("seed"), spec.GetArgument<int>("batch_size")) {
    auto scaling_arg = spec.GetRepeatedArgument<float>("scaling");
    DALI_ENFORCE(scaling_arg.size() == 2,
                 make_string("`scaling` must be a range `[min, max]`. Got ",
                             scaling_arg.size(), " values"));
    scale_range_.min = scaling_arg[0];
    scale_range_.max = scaling_arg[1];
    DALI_ENFORCE(
        scale_range_.min >= 0 && scale_range_.min <= scale_range_.max,
        make_string("`scaling` range must be positive and min <= max. Got: ", scale_range_.min,
                    ", ", scale_range_.max));

    auto aspect_ratio_arg = spec.GetRepeatedArgument<float>("aspect_ratio");
    DALI_ENFORCE(aspect_ratio_arg.size() == 2 || aspect_ratio_arg.size() == 6,
        make_string(
            "`aspect_ratio` range argument should have 2 elements, or 6 elements in case of "
            "3D bounding boxes. Got ",
            aspect_ratio_arg.size(), " elements"));
    aspect_ratio_ranges_.resize(aspect_ratio_arg.size() / 2);
    int k = 0;
    for (auto &range : aspect_ratio_ranges_) {
      range.min = aspect_ratio_arg[k++];
      range.max = aspect_ratio_arg[k++];
      DALI_ENFORCE(range.min >= 0 && range.min <= range.max,
                   make_string("`aspect_ratio` range must be positive and min <= max. Got: ",
                               range.min, ", ", range.max));
    }

    DALI_ENFORCE(has_crop_shape_ == has_input_shape_,
      "`crop_shape` and `input_shape` should be provided together or not provided");

    if (spec.ArgumentDefined("ltrb")) {
      if (spec.ArgumentDefined("bbox_layout")) {
        DALI_FAIL(
            "`ltrb` and `bbox_layout` can't be provided at the same time. `ltrb` was deprecated in "
            "favor of `bbox_layout`.");
      }
      DALI_WARN(
          "WARNING: `ltrb` is deprecated. Please use `bbox_layout` to specify the format of the "
          "bounding box. E.g. For 2D bounding boxes, `ltrb=True`` is equivalent to "
          "`bbox_layout=\"xyXY\"`, and `ltrb=False` is equivalent to `bbox_layout=\"xyWH\"`");
    }

    bool allow_no_crop = spec.GetArgument<bool>("allow_no_crop");
    if (has_crop_shape_) {
      // If it was left default but a crop_shape was provided, disallow no crop silently
      if (!spec.HasArgument("allow_no_crop")) {
        DALI_WARN("Using explicit `crop_shape`, `allow_no_crop` will not take effect.");
        allow_no_crop = false;
      }

      DALI_ENFORCE(!allow_no_crop,
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
      sample_options_.push_back({false, threshold});
    }

    if (spec.HasArgument("threshold_type")) {
      auto threshold_type = spec.GetArgument<std::string>("threshold_type");
      if (threshold_type == "iou") {
        overlap_metric_ = OverlapMetric::IoU;
      } else  if (threshold_type == "overlap") {
        overlap_metric_ = OverlapMetric::Overlap;
      } else {
        DALI_FAIL(make_string("Not supported ``threshold_type`` value: \"", threshold_type,
                              "\". Supported values are: \"iou\", \"overlap\"."));
      }
    }

    if (allow_no_crop) {
      sample_options_.push_back({true, 0.0f});
    }

    total_num_attempts_ = -1;
    if (spec.HasArgument("total_num_attempts")) {
      total_num_attempts_ = spec.GetArgument<int>("total_num_attempts");
      DALI_ENFORCE(total_num_attempts_ > 0,
        "Minimum total number of attempts must be greater than zero");
    }

    auto default_bbox_layout_start_end = DefaultBBoxLayout<ndim>();
    auto default_bbox_layout_start_shape = DefaultBBoxAnchorAndShapeLayout<ndim>();
    if (bbox_layout_.empty()) {
      auto ltrb = spec_.GetArgument<bool>("ltrb");
      bbox_layout_ = ltrb ? default_bbox_layout_start_end : default_bbox_layout_start_shape;
    }
    DALI_ENFORCE(bbox_layout_.is_permutation_of(default_bbox_layout_start_end) ||
                 bbox_layout_.is_permutation_of(default_bbox_layout_start_shape),
      make_string("`bbox_layout` should be a permutation of `", default_bbox_layout_start_end,
                  "` or `", default_bbox_layout_start_shape, "`. Got: `", bbox_layout_, "`"));
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override {
    if (spec_.ArgumentDefined("input_shape") && spec_.ArgumentDefined("crop_shape")) {
      CollectShape(input_shape_, "input_shape", spec_, ws, ndim);
      CollectShape(crop_shape_, "crop_shape", spec_, ws, ndim);

      // Converting the shapes to "WHD" or "WH" if necessary
      auto default_shape_layout = InternalShapeLayout(ndim);
      if (!shape_layout_.empty() && shape_layout_ != default_shape_layout) {
        DALI_ENFORCE(shape_layout_.is_permutation_of(default_shape_layout),
                     make_string("`shape_layout` should be a permutation of ", default_shape_layout,
                                 "` for the provided inputs"));
        auto perm = GetDimIndices(shape_layout_, default_shape_layout);
        for (int sample = 0; sample < static_cast<int>(input_shape_.size()); sample++) {
          TensorShape<> in_shape = input_shape_[sample];
          TensorShape<> crop_shape = crop_shape_[sample];

          DALI_ENFORCE(crop_shape.sample_dim() == ndim,
                    make_string("Unexpected number of dimensions. Expected ", ndim, ", got ",
                                crop_shape.sample_dim()));

          for (int i = 0; i < ndim; i++) {
            int axis = perm[i];
            DALI_ENFORCE(
                crop_shape[axis] <= in_shape[axis],
                make_string("Crop shape can't exceed input shape dimensions. Got crop_shape=",
                            crop_shape, ", input_shape=", in_shape));
            input_shape_[sample][i] = in_shape[axis];
            crop_shape_[sample][i]  = crop_shape[axis];
          }
        }
      }
    }
    return false;
  }

  void RunImpl(SampleWorkspace &ws) override {
    const auto &boxes_tensor = ws.Input<CPUBackend>(0);
    auto nboxes = boxes_tensor.dim(0);
    auto ncoords = ndim * 2;

    std::vector<Box<ndim, float>> bounding_boxes(nboxes);
    ReadBoxes(make_span(bounding_boxes),
              make_cspan(boxes_tensor.data<float>(), boxes_tensor.size()), bbox_layout_);

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
    ProspectiveCrop prospective_crop =
        FindProspectiveCrop(make_cspan(bounding_boxes), make_cspan(labels), sample);

    WriteCropToOutput(ws, prospective_crop.crop);
    WriteBoxesToOutput(ws, make_cspan(prospective_crop.boxes));

    if (has_labels_) {
      DALI_ENFORCE(
          prospective_crop.boxes.size() == prospective_crop.labels.size(),
          make_string("Expected boxes.size() == labels.size(). Received: ",
                      prospective_crop.boxes.size(), " != ", prospective_crop.labels.size()));
      WriteLabelsToOutput(ws, make_cspan(prospective_crop.labels));
    }
  }

 private:
  struct ProspectiveCrop {
    bool success = false;
    Box<ndim, float> crop{};
    std::vector<Box<ndim, float>> boxes;
    std::vector<int> labels;

    ProspectiveCrop(bool success,
                    const Box<ndim, float>& crop,
                    span<const Box<ndim, float>> boxes_data,
                    span<const int> labels_data,
                    bool has_labels)
        : success(success), crop(crop) {
      assert(boxes_data.size() == labels_data.size() || !has_labels);
      boxes.resize(boxes_data.size());
      labels.resize(labels_data.size());
      for (int i = 0; i < boxes_data.size(); i++) {
        boxes[i] = boxes_data[i];
        if (has_labels) labels[i] = labels_data[i];
      }
    }
    ProspectiveCrop() = default;
  };

  /**
   * @brief Fixes shape dimensions to follow aspect ratio constraints in case of ar_min == ar_max
   * @remarks The dimensions are fixed on a random order
   */
  void FixAspectRatios(vec<ndim, float>& shape) {
    // If aspect ratio is fixed, fix the required dimensions
    std::array<float, ndim*ndim> fixed_aspect_ratios;
    int k = 0;

    bool need_fix = false;
    for (int d0 = 0; d0 < ndim; d0++) {
      for (int d1 = d0 + 1; d1 < ndim; d1++) {
        // to be used later when min==max
        if (aspect_ratio_ranges_[k].min == aspect_ratio_ranges_[k].max) {
          fixed_aspect_ratios[d0*ndim+d1] = aspect_ratio_ranges_[k].min;
          fixed_aspect_ratios[d1*ndim+d0] = 1.0 / aspect_ratio_ranges_[k].min;
          need_fix = true;
        } else {
          fixed_aspect_ratios[d0*ndim+d1] = 0.0f;
          fixed_aspect_ratios[d1*ndim+d0] = 0.0f;
        }
        k = (k + 1) % aspect_ratio_ranges_.size();
      }
    }

    if (!need_fix)
      return;

    std::array<int, ndim> order;
    std::iota(order.begin(), order.end(), 0);
    std::random_shuffle(order.begin(), order.end());

    float max_extent = 0.0f;
    for (int d = 0; d < ndim; d++) {
      max_extent = std::max(max_extent, shape[d]);
    }

    for (int i0 = 0; i0 < ndim; i0++) {
      for (int i1 = i0 + 1; i1 < ndim; i1++) {
        int d0 = order[i0], d1 = order[i1];
        auto fixed_ar = fixed_aspect_ratios[d1*ndim+d0];
        if (fixed_ar > 0) {
          shape[d1] = shape[d0] * fixed_ar;
        }
      }
    }

    // Re-scale so that largest extent matches the previous max extent
    float new_max_extent = 0.0;
    for (int d = 0; d < ndim; d++)
      new_max_extent = std::max(new_max_extent, shape[d]);

    for (auto &extent : shape)
      extent = max_extent * extent / new_max_extent;
  }

  ProspectiveCrop FindProspectiveCrop(span<const Box<ndim, float>> bounding_boxes,
                                      span<const int> labels, int sample) {
    ProspectiveCrop crop;
    int count = 0;
    float best_metric = -1.0;

    while (!crop.success && (total_num_attempts_ < 0 || count < total_num_attempts_)) {
      auto &rng = rngs_[sample];
      std::uniform_int_distribution<> idx_dist(0, sample_options_.size() - 1);
      SampleOption option = sample_options_[idx_dist(rng)];
      bool absolute_crop_dims = has_crop_shape_;

      if (option.no_crop) {
        Box<ndim, float> no_crop = Uniform<ndim>(0.0f, 1.0f);
        if (absolute_crop_dims) {
          auto &input_shape = input_shape_[sample];
          for (int d = 0; d < ndim; d++)
            no_crop.hi[d] *= input_shape[d];
        }
        crop = ProspectiveCrop(true, no_crop, bounding_boxes, labels, has_labels_);
        break;
      }

      vec<ndim, float> shape, anchor;
      Box<ndim, float> rel_crop, out_crop;
      std::array<float, coords_size> out_bounds, rel_bounds;
      for (int i = 0; i < num_attempts_; i++, count++) {
        if (absolute_crop_dims) {
          auto &crop_shape = crop_shape_[sample];
          auto &input_shape = input_shape_[sample];

          for (int d = 0; d < ndim; d++) {
            shape[d] = static_cast<float>(crop_shape[d]);
            out_crop.hi[d] = shape[d];
            rel_crop.hi[d] = shape[d] / input_shape[d];
          }

          for (int d = 0; d < ndim; d++) {
            std::uniform_int_distribution<> anchor_dist(0, input_shape[d] - crop_shape[d]);
            anchor[d] = static_cast<float>(anchor_dist(rng));
            out_crop.lo[d] = anchor[d];
            rel_crop.lo[d] = anchor[d] / input_shape[d];
          }
          out_crop.hi += out_crop.lo;
          rel_crop.hi += rel_crop.lo;
        } else {  // relative dimensions
          std::uniform_real_distribution<float> extent_dist(scale_range_.min, scale_range_.max);
          for (int d = 0; d < ndim; d++) {
            shape[d] = extent_dist(rng);
          }

          FixAspectRatios(shape);

          if (!ValidAspectRatio(shape)) {
            continue;
          }

          for (int d = 0; d < ndim; d++) {
            std::uniform_real_distribution<float> anchor_dist(0.0f, 1.0f - shape[d]);
            anchor[d] = anchor_dist(rng);
            rel_crop.lo[d] = anchor[d];
            rel_crop.hi[d] = anchor[d] + shape[d];
          }
          out_crop = rel_crop;
        }

        float min_overlap = 0.0, max_overlap = 0.0;
        std::tie(min_overlap, max_overlap) =
            OverlapMetricRange(rel_crop, make_cspan(bounding_boxes));
        float metric = all_boxes_above_threshold_ ? min_overlap : max_overlap;
        bool is_valid_overlap = metric >= option.threshold;
        if (metric <= best_metric && !is_valid_overlap)
          continue;

        best_metric = metric;

        crop = {};
        crop.crop = out_crop;

        crop.boxes.resize(bounding_boxes.size());
        for (int i = 0; i < bounding_boxes.size(); i++)
          crop.boxes[i] = bounding_boxes[i];

        if (!labels.empty()) {
          assert(labels.size() == bounding_boxes.size());
          crop.labels.resize(labels.size());
          for (int i = 0; i < labels.size(); i++)
            crop.labels[i] = labels[i];
        }

        FilterByCentroid(rel_crop, crop.boxes, crop.labels);
        for (auto &box : crop.boxes) {
          box = RemapBox(box, rel_crop);
        }

        bool at_least_one_box = !crop.boxes.empty();
        crop.success = is_valid_overlap && at_least_one_box;
      }
    }

    if (!crop.success) {
      DALI_WARN(make_string(
        "Could not find a valid cropping window to satisfy the specified requirements (attempted ",
        count, " times). Using the best cropping window so far (best_metric=", best_metric, ")"));
      crop.success = true;
    }
    return crop;
  }

  bool ValidAspectRatio(vec<ndim, float> shape) {
    assert(static_cast<int>(shape.size()) == ndim);
    int k = 0;
    assert(!aspect_ratio_ranges_.empty());
    for (int i = 0; i < ndim; i++) {
      for (int j = i + 1; j < ndim; j++) {
        if (!aspect_ratio_ranges_[k].Contains(shape[i] / shape[j])) return false;
        k = (k + 1) % aspect_ratio_ranges_.size();
      }
    }
    return true;
  }

  template <typename Metric>
  std::pair<float, float> MinMaxRange(const Box<ndim, float> &crop,
                                      span<const Box<ndim, float>> boxes,
                                      Metric&& metric_f) {
    float best_metric = 0.0, worst_metric = 0.0;
    if (boxes.empty())
      return {0.0f, 0.0f};
    float metric = metric_f(crop, boxes[0]);
    best_metric = metric;
    worst_metric = metric;
    for (int i = 1; i < boxes.size(); i++) {
      metric = metric_f(crop, boxes[i]);
      best_metric = std::max(metric, best_metric);
      worst_metric = std::min(metric, worst_metric);
    }
    return {worst_metric, best_metric};
  }

  std::pair<float, float> OverlapMetricRange(const Box<ndim, float> &crop,
                                             span<const Box<ndim, float>> boxes) {
    if (overlap_metric_ == OverlapMetric::Overlap) {
      auto f =
          [](const Box<ndim, float> &crop, const Box<ndim, float> &box) {
            return volume(intersection(crop, box)) / static_cast<float>(volume(box));
          };
      return MinMaxRange(crop, boxes, f);
    } else {  // IoU
      auto f =
          [](const Box<ndim, float> &crop, const Box<ndim, float> &box) {
            return intersection_over_union(crop, box);
          };
      return MinMaxRange(crop, boxes, f);
    }
  }

  void FilterByCentroid(const Box<ndim, float> &crop,
                        std::vector<Box<ndim, float>> &bboxes,
                        std::vector<int> &labels) {
    std::vector<Box<ndim, float>> new_bboxes;
    std::vector<int> new_labels;
    bool process_labels = !labels.empty();
    assert(labels.empty() || labels.size() == bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i++) {
      if (crop.contains(bboxes[i].centroid())) {
        new_bboxes.push_back(bboxes[i]);
        if (process_labels)
          new_labels.push_back(labels[i]);
      }
    }
    std::swap(bboxes, new_bboxes);
    if (process_labels)
      std::swap(labels, new_labels);
  }

  void WriteCropToOutput(SampleWorkspace &ws, const Box<ndim, float> &crop) {
    // output0 : anchor, output1 : shape
    auto &anchor_out = ws.Output<CPUBackend>(0);
    anchor_out.Resize({ndim});
    auto *anchor_out_data = anchor_out.mutable_data<float>();

    auto &shape_out = ws.Output<CPUBackend>(1);
    shape_out.Resize({ndim});
    auto *shape_out_data = shape_out.mutable_data<float>();

    auto extent = crop.extent();
    for (int d = 0; d < ndim; d++) {
      anchor_out_data[d] = crop.lo[d];
      shape_out_data[d] = extent[d];
    }
  }

  void WriteBoxesToOutput(SampleWorkspace &ws,
                          span<const Box<ndim, float>> bounding_boxes) {
    auto &bbox_out = ws.Output<CPUBackend>(2);
    bbox_out.Resize({static_cast<int64_t>(bounding_boxes.size()), coords_size});
    auto *bbox_out_data = bbox_out.mutable_data<float>();
    WriteBoxes(make_span(bbox_out_data, bbox_out.size()), make_cspan(bounding_boxes),
               bbox_layout_);
  }

  void WriteLabelsToOutput(SampleWorkspace &ws, span<const int> labels) {
    auto &labels_out = ws.Output<CPUBackend>(3);
    labels_out.Resize({static_cast<Index>(labels.size()), 1});
    auto *labels_out_data = labels_out.mutable_data<int>();
    for (int i = 0; i < labels_out.size(); i++)
      labels_out_data[i] = labels[i];
  }

 private:
  OpSpec spec_;
  int num_attempts_;
  int total_num_attempts_;
  bool has_labels_;
  bool has_crop_shape_;
  bool has_input_shape_;

  TensorLayout bbox_layout_;
  TensorLayout shape_layout_;

  OverlapMetric overlap_metric_ = OverlapMetric::IoU;
  bool all_boxes_above_threshold_ = true;

  BatchRNG<std::mt19937> rngs_;

  std::vector<SampleOption> sample_options_;

  std::vector<TensorShape<>> crop_shape_;
  std::vector<TensorShape<>> input_shape_;

  Range scale_range_;
  std::vector<Range> aspect_ratio_ranges_;
};

template <>
RandomBBoxCrop<CPUBackend>::~RandomBBoxCrop() = default;

template <>
RandomBBoxCrop<CPUBackend>::RandomBBoxCrop(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , spec_(spec) {}

template <>
bool RandomBBoxCrop<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                           const workspace_t<CPUBackend> &ws) {
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
  auto num_dims = ncoords / 2;

  DALI_ENFORCE(num_dims == 2 || num_dims == 3,
    make_string("Unexpected number of dimensions: ", num_dims));

  if (impl_ == nullptr || impl_ndim_ != num_dims) {
    VALUE_SWITCH(num_dims, ndim, (2, 3),
      (impl_ = std::make_unique<RandomBBoxCropImpl<ndim>>(spec_);),
      (DALI_FAIL(make_string("Not supported number of dimensions", num_dims));));
    impl_ndim_ = num_dims;
  }
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void RandomBBoxCrop<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop<CPUBackend>, CPU);

}  // namespace dali
