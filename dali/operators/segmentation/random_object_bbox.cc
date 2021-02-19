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

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/imgproc/structure/label_bbox.h"
#include "dali/operators/segmentation/random_object_bbox.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/imgproc/structure/connected_components.h"

namespace dali {

using dali::kernels::OutTensorCPU;
using dali::kernels::InTensorCPU;
using dali::kernels::OutListCPU;
using dali::kernels::InListCPU;

using kernels::connected_components::LabelConnectedRegions;
using kernels::label_bbox::GetLabelBoundingBoxes;

DALI_SCHEMA(segmentation__RandomObjectBBox)
  .DocStr(R"(Randomly selects an object from a mask and returns its bounding box.

This operator takes a labeled segmentation map as its input. With probability ``foreground_prob``
it randomly selects a label (uniformly or according to the distribution given as ``class_weights``),
extracts connected blobs of pixels with the selected label and randomly selects one of the blobs.
The blobs may be further filtered according to ``k_largest`` and ``threshold``.
The output is a bounding box of the selected blob in one of the formats described in ``format``.

With probability 1-foreground_prob, the entire area of the input is returned.)")
  .NumInput(1)
  .OutputFn([](const OpSpec& spec) {
    int separate_corners = spec.GetArgument<string>("format") != "box";
    int output_class = spec.GetArgument<bool>("output_class");
    return 1 + separate_corners + output_class;
  })
  .AddOptionalArg("ignore_class", R"(If True, all objects are picked with equal probability,
regardless of the class they belong to. Otherwise, a class is picked first and then an object is
randomly selected from this class.

This argument is incompatible with ``classes``, ``class_weights`` or ``output_class``.

.. note::
  This flag only affects the probability with which blobs are selected. It does not cause
  blobs of different classes to be merged.)", false)
  .AddOptionalArg("output_class", R"(If True, an additional output is produced which contains the
label of the class to which the selected box belongs, or background label if the selected box
is not an object bounding box.

The output may not be an object bounding box when any of the following conditions occur:
  - the sample was randomly (according to ``foreground_prob``) chosen not be be a foreground one
  - the sample contained no foreground objects
  - no bounding box met the required size threshold.)", false)
  .AddOptionalArg("foreground_prob", "Probability of selecting a foreground bounding box.", 1.0f,
    true)
  .AddOptionalArg<vector<int>>("classes", R"(List of labels considered as foreground.

If left unspecified, all labels not equal to ``background`` are considered foreground.)",
    nullptr, true)
  .AddOptionalArg("background", R"(Background label.

If left unspecified, it's either 0 or any value not in ``classes``.)", 0, true)
  .AddOptionalArg<vector<float>>("class_weights", R"(Relative probabilities of foreground classes.

Each value corresponds to a class label in ``classes``. If ``classes`` are not specified,
consecutive 1-based labels are assigned.

The sum of the weights doesn't have to be equal to 1 - if it isn't the weights will be
normalized .)", nullptr, true)
  .AddOptionalArg<int>("k_largest", R"(If specified, the boxes are sorted by decreasing volume
and only ``k_largest`` are considered.

If ``ignore_class`` is True, ``k_largest`` referes to all boxes; otherwise it refers to the
selected class.)",
    nullptr)
  .AddOptionalArg<vector<int>>("threshold", R"(Per-axis minimum size of the bounding boxes
to return.

If the selected class doesn't contain any bounding box that meets this condition, it is rejected
and another class is picked. If no class contains a satisfactory box, the entire input area
is returned.)", nullptr, true)
  .AddOptionalArg("format", R"(Format in which the data is returned.

Possible choices are::
  * "anchor_shape" (the default) - there are two outputs: anchor and shape
  * "start_end" - there are two outputs: bounding box start and one-past-end coordinates
  * "box" - there is one output that contains concatenated start and end coordinates
)", "anchor_shape");


bool RandomObjectBBox::SetupImpl(vector<OutputDesc> &out_descs, const HostWorkspace &ws) {
  out_descs.resize(spec_.NumOutput());
  auto &input = ws.InputRef<CPUBackend>(0);
  int ndim = input.sample_dim();
  int N = input.ntensor();

  DALI_ENFORCE(N == 0 || (ndim >= 1 && ndim <= 6),
      make_string("Unsuported number of dimensions ", ndim, "; must be 1..6"));
  AcquireArgs(ws, N, ndim);
  out_descs[0].type = TypeTable::GetTypeInfo(DALI_INT32);
  out_descs[0].shape = uniform_list_shape<DynamicDimensions>(
      N, TensorShape<1>{ format_ == Out_Box ? 2*ndim : ndim });
  if (format_ != Out_Box)
    out_descs[1] = out_descs[0];
  if (HasClassLabelOutput()) {
    out_descs[class_output_idx_].type = TypeTable::GetTypeInfo(DALI_INT32);
    out_descs[class_output_idx_].shape.resize(N, 0);
  }

  return true;
}

void RandomObjectBBox::AcquireArgs(const HostWorkspace &ws, int N, int ndim) {
  background_.Acquire(spec_, ws, N);
  if (classes_.IsDefined())
    classes_.Acquire(spec_, ws, N);
  foreground_prob_.Acquire(spec_, ws, N);
  if (weights_.IsDefined())
    weights_.Acquire(spec_, ws, N);
  if (threshold_.IsDefined())
    threshold_.Acquire(spec_, ws, N, TensorShape<1>{ndim});

  if (weights_.IsDefined() && classes_.IsDefined()) {
    DALI_ENFORCE(weights_.get().shape == classes_.get().shape, make_string(
      "If both ``classes`` and ``class_weights`` are provided, their shapes must match. Got:"
      "\n  classes.shape  = ", classes_.get().shape,
      "\n  weights.shape  = ", weights_.get().shape));
  }
}


#define INPUT_TYPES (bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t)

template <typename T>
void RandomObjectBBox::FindLabels(std::unordered_set<int> &labels, const T *data, int64_t N) {
  if (!N)
    return;
  T prev = data[0];
  labels.insert(prev);
  for (int64_t i = 1; i < N; i++) {
    if (data[i] == prev)
      continue;  // skip runs of equal labels
    labels.insert(data[i]);
    prev = data[i];
  }
}

template <typename Out, typename In>
void FilterByLabel(Out *out, const In *in, int64_t N, In label) {
  for (int64_t i = 0; i < N; i++) {
    out[i] = in[i] == label;
  }
}

template <typename Out, typename In>
void FilterByLabel(const OutTensorCPU<Out> &out, const InTensorCPU<In> &in, same_as_t<In> label) {
  assert(out.shape == in.shape);
  int64_t N = in.num_elements();
  FilterByLabel(out.data, in.data, N, label);
}


template <typename Lo, typename Hi>
void StoreBox(const OutListCPU<int, 1> &out1,
              const OutListCPU<int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Lo &&start, Hi &&end) {
  assert(size(start) == size(end));
  int ndim = size(start);
  switch (format) {
    case RandomObjectBBox::Out_Box:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out1.data[sample_idx][i + ndim] = end[i];
      }
      break;
    case RandomObjectBBox::Out_AnchorShape:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out2.data[sample_idx][i] = end[i] - start[i];
      }
      break;
    case RandomObjectBBox::Out_StartEnd:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out2.data[sample_idx][i] = end[i];
      }
      break;
    default:
      assert(!"Unreachable code");
  }
}

template <typename Box>
void StoreBox(const OutListCPU<int, 1> &out1,
              const OutListCPU<int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Box &&box) {
  StoreBox(out1, out2, format, sample_idx, box.lo, box.hi);
}

void RandomObjectBBox::GetBgFgAndWeights(
      ClassVec &classes, WeightVec &weights, int &background, int sample_idx) {
  background = background_[sample_idx].data[0];
  if (ignore_class_)
    return;  // we don't care about classes at all

  if (classes_.IsDefined()) {
    const auto &cls_tv = classes_[sample_idx];
    int ncls = cls_tv.shape[0];
    classes.resize(ncls);
    if (!weights_.IsDefined()) {
      weights.clear();
      weights.resize(ncls, 1.0f);
    }

    int min_class_label = ncls ? cls_tv.data[0] : 0;
    bool choose_different_background = false;
    for (int i = 0; i < ncls; i++) {
      classes[i] = cls_tv.data[i];
      if (classes[i] < min_class_label)
        min_class_label = classes[i];

      if (classes[i] == background) {
        if (background_.IsDefined()) {
          // Background was explicitly specified this way - that's an error
          DALI_FAIL(make_string("Class label ", classes[i],
            " coincides with background label - please specify a different background label or "
            " remove it from your list of foreground classes."));
        } else {
          // Not specified? Pick a different one.
          choose_different_background = true;
        }
      }
    }
    if (choose_different_background) {
      if (min_class_label == std::numeric_limits<int>::min()) {
        background = std::numeric_limits<int>::max();  // wrap around

        std::unordered_set<int> cls_set;
        for (auto cls : classes)
          cls_set.insert(cls);

        while (cls_set.count(background))
          background--;
      } else {
        background = min_class_label - 1;
      }
    }
  }
  if (weights_.IsDefined()) {
    const auto &cls_tv = weights_[sample_idx];
    int ncls = cls_tv.shape[0];
    weights.resize(ncls);
    for (int i = 0; i < ncls; i++)
      weights[i] = cls_tv.data[i];
    if (!classes_.IsDefined()) {
      classes.resize(ncls);
      int cls = 0;
      for (int i = 0; i < ncls; i++, cls++) {
        if (cls == background)
          cls++;
        classes[i] = cls;
      }
    }
  }
}

template <int ndim>
int RandomObjectBBox::PickBox(span<Box<ndim, int>> boxes, int sample_idx) {
  auto beg = boxes.begin();
  auto end = boxes.end();
  if (threshold_.IsDefined()) {
    vec<ndim, int> threshold;
    const int *thresh = threshold_[sample_idx].data;
    assert(threshold_.get().shape[sample_idx] == TensorShape<1>{ ndim });
    for (int i = 0; i < ndim; i++)
      threshold[i] = thresh[i];
    end = std::remove_if(beg, end, [threshold](const Box<ndim, int> &box) {
      return any_coord(box.extent() < threshold);
    });
  }
  int n = end - beg;
  if (n <= 0)
    return -1;

  if (k_largest_ > 0 && k_largest_ < n) {
    SmallVector<std::pair<int64_t, int>, 32> vol_idx;
    vol_idx.resize(n);
    for (int i = 0; i < n; i++) {
      vol_idx[i] = { -volume(boxes[i]), i };
    }
    std::sort(vol_idx.begin(), vol_idx.end());
    std::uniform_int_distribution<int> dist(0, std::min(n, k_largest_)-1);
    return vol_idx[dist(rngs_[sample_idx])].second;
  } else {
    std::uniform_int_distribution<int> dist(0, n-1);
    return dist(rngs_[sample_idx]);
  }
}

bool RandomObjectBBox::PickBlob(SampleContext &ctx, int nblobs) {
  if (!nblobs)
    return false;

  int ndim = ctx.blobs.dim();
  ctx.box_data.clear();
  ctx.box_data.resize(2*ndim*nblobs);

  VALUE_SWITCH(ndim, static_ndim, (1, 2, 3, 4, 5, 6),
    (
      auto *box_data = reinterpret_cast<Box<static_ndim, int>*>(ctx.box_data.data());
      auto boxes = make_span(box_data, nblobs);
      GetLabelBoundingBoxes(boxes, ctx.blobs.to_static<static_ndim>(), -1);
      int box_idx = PickBox(boxes, ctx.sample_idx);
      if (box_idx >= 0) {
        ctx.SelectBox(box_idx);
        return true;
      }
    ), (  // NOLINT
      DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim, "; must be 1..6"));
    )  // NOLINT
  );  // NOLINT
  return false;
}

template <typename T>
bool RandomObjectBBox::PickForegroundBox(
      SampleContext &context, const InTensorCPU<T> &input) {
  GetBgFgAndWeights(context.classes, context.weights, context.background, context.sample_idx);
  context.class_label = context.background;
  if (ignore_class_) {
    int nblobs = LabelConnectedRegions(context.blobs, input, -1, context.background);
    return PickBlob(context, nblobs);
  } else {
    FindLabels(context.labels, input);

    context.labels.erase(context.background);

    if (!classes_.IsDefined() && !weights_.IsDefined()) {
      context.classes.clear();
      context.weights.clear();
      for (auto cls : context.labels) {
        context.classes.push_back(cls);
        context.weights.push_back(1);
      }
      // We need to sort, because the order in `context.labels` depends on its previous contents
      // (it changes with the number of bins in the hastable).
      // We want don't need any particular order here, but it must be deterministic
      // - thus, sort.
      std::sort(context.classes.begin(), context.classes.end());
    } else {
      for (int i = 0; i < static_cast<int>(context.classes.size()); i++) {
        if (!context.labels.count(context.classes[i]))
          context.weights[i] = 0;  // label not present - reduce its weight to 0
      }
    }

    while (context.CalculateCDF()) {
      if (!context.PickClassLabel(rngs_[context.sample_idx]))
        return false;

      assert(context.class_label != context.background);
      FilterByLabel(context.filtered, input, context.class_label);

      int nblobs = LabelConnectedRegions<int64_t, uint8_t, -1>(
          context.blobs, context.filtered, -1, 0);

      if (PickBlob(context, nblobs))
        return true;

      // we couldn't find a satisfactory blob in this class, so let's exclude it and try again
      context.weights[context.class_idx] = 0;
      context.class_label = context.background;
    }
    // we've run out of classes and still there's no good blob
    return false;
  }
}

bool RandomObjectBBox::PickForegroundBox(SampleContext &context) {
  bool ret = false;
  TYPE_SWITCH(context.input->type().id(), type2id, T, INPUT_TYPES,
    (ret = PickForegroundBox(context, view<const T>(*context.input));),
    (DALI_FAIL(make_string("Unsupported input type: ", context.input->type().id())))
  );  // NOLINT
  return ret;
}

void RandomObjectBBox::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int N = input.ntensor();
  if (N == 0)
    return;

  int ndim = input.sample_dim();
  auto &tp = ws.GetThreadPool();

  OutListCPU<int, 1> out1 = view<int, 1>(ws.OutputRef<CPUBackend>(0));
  OutListCPU<int, 1> out2;
  if (format_ != Out_Box)
    out2 = view<int, 1>(ws.OutputRef<CPUBackend>(1));
  OutListCPU<int, 0> class_label_out;
  if (HasClassLabelOutput())
    class_label_out = view<int, 0>(ws.OutputRef<CPUBackend>(class_output_idx_));

  TensorShape<> default_anchor;
  default_anchor.resize(ndim);

  contexts_.resize(tp.NumThreads()+1);

  std::uniform_real_distribution<> foreground(0, 1);
  for (int i = 0; i < N; i++) {
    bool fg = foreground(rngs_[i]) < foreground_prob_[i].data[0];
    if (!fg) {
      StoreBox(out1, out2, format_, i, default_anchor, input.tensor_shape(i));
      if (HasClassLabelOutput()) {
        SampleContext &ctx = contexts_[0];
        GetBgFgAndWeights(ctx.classes, ctx.weights, ctx.background, i);
        class_label_out.data[i][0] = ctx.background;
      }
    } else {
      tp.AddWork([&, i](int thread_idx) {
        SampleContext &ctx = contexts_[thread_idx+1];
        ctx.Init(i, &input[i]);
        ctx.out1 = out1[i];
        if (out2.num_samples() > 0)
          ctx.out2 = out2[i];

        if (PickForegroundBox(ctx)) {
          assert(ctx.class_label != ctx.background || ignore_class_);
          StoreBox(out1, out2, format_, i, ctx.selected_box);
        } else {
          assert(ctx.class_label == ctx.background);
          StoreBox(out1, out2, format_, i, default_anchor, input.tensor_shape(i));
        }

        if (HasClassLabelOutput())
          class_label_out.data[i][0] = ctx.class_label;
      }, input.tensor_shape(i).num_elements());
    }
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(segmentation__RandomObjectBBox, RandomObjectBBox, CPU);

}  // namespace dali
