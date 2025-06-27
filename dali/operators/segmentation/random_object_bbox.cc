// Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This operator takes a labeled segmentation map as its input. With probability `foreground_prob`
it randomly selects a label (uniformly or according to the distribution given as `class_weights`),
extracts connected blobs of pixels with the selected label and randomly selects one of the blobs.
The blobs may be further filtered according to `k_largest` and `threshold`.
The output is a bounding box of the selected blob in one of the formats described in `format`.

With probability 1-foreground_prob, the entire area of the input is returned.)")
  .NumInput(1)
  .OutputFn([](const OpSpec& spec) {
    int separate_corners = spec.GetArgument<string>("format") != "box";
    int output_class = spec.GetArgument<bool>("output_class");
    return 1 + separate_corners + output_class;
  })
  .AddRandomSeedArg()
  .AddOptionalArg("ignore_class", R"(If True, all objects are picked with equal probability,
regardless of the class they belong to. Otherwise, a class is picked first and then an object is
randomly selected from this class.

This argument is incompatible with `classes`, `class_weights` or `output_class`.

.. note::
  This flag only affects the probability with which blobs are selected. It does not cause
  blobs of different classes to be merged.)", false)
  .AddOptionalArg("output_class", R"(If True, an additional output is produced which contains the
label of the class to which the selected box belongs, or background label if the selected box
is not an object bounding box.

The output may not be an object bounding box when any of the following conditions occur:
  - the sample was randomly (according to `foreground_prob`) chosen not be be a foreground one
  - the sample contained no foreground objects
  - no bounding box met the required size threshold.)", false)
  .AddOptionalArg("foreground_prob", "Probability of selecting a foreground bounding box.", 1.0f,
    true)
  .AddOptionalArg<vector<int>>("classes", R"(List of labels considered as foreground.

If left unspecified, all labels not equal to `background` are considered foreground.)",
    nullptr, true)
  .AddOptionalArg("background", R"(Background label.

If left unspecified, it's either 0 or any value not in `classes`.)", 0, true)
  .AddOptionalArg<vector<float>>("class_weights", R"(Relative probabilities of foreground classes.

Each value corresponds to a class label in `classes`. If `classes` are not specified,
consecutive 1-based labels are assigned.

The sum of the weights doesn't have to be equal to 1 - if it isn't the weights will be
normalized .)", nullptr, true)
  .AddOptionalArg<int>("k_largest", R"(If specified, the boxes are sorted by decreasing volume
and only `k_largest` are considered.

If `ignore_class` is True, `k_largest` referes to all boxes; otherwise it refers to the
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
)", "anchor_shape")
  .AddOptionalArg("cache_objects", R"(Cache object bounding boxes to avoid the computational cost
of finding object blobs in previously seen inputs.

Searching for blobs of connected pixels and finding boxes can take a long time. When the dataset
has few items, but item size is big, you can use caching to save the boxes and reuse them when
the same input is seen again. The inputs are compared based on 256-bit hash, which is much faster
to compute than to recalculate the object boxes.)", false);

bool RandomObjectBBox::SetupImpl(vector<OutputDesc> &out_descs, const Workspace &ws) {
  out_descs.resize(spec_.NumOutput());
  auto &input = ws.Input<CPUBackend>(0);
  int ndim = input.sample_dim();
  int N = input.num_samples();

  DALI_ENFORCE(N == 0 || (ndim >= 1 && ndim <= 6),
      make_string("Unsuported number of dimensions ", ndim, "; must be 1..6"));
  AcquireArgs(ws, N, ndim);
  out_descs[0].type = DALI_INT32;
  out_descs[0].shape = uniform_list_shape<DynamicDimensions>(
      N, TensorShape<1>{ format_ == Out_Box ? 2*ndim : ndim });
  if (format_ != Out_Box)
    out_descs[1] = out_descs[0];
  if (HasClassLabelOutput()) {
    out_descs[class_output_idx_].type = DALI_INT32;
    out_descs[class_output_idx_].shape.resize(N, 0);
  }

  return true;
}

void RandomObjectBBox::AcquireArgs(const Workspace &ws, int N, int ndim) {
  background_.Acquire(spec_, ws, N);
  if (classes_.HasExplicitValue())
    classes_.Acquire(spec_, ws, N);
  foreground_prob_.Acquire(spec_, ws, N);
  if (weights_.HasExplicitValue())
    weights_.Acquire(spec_, ws, N);
  if (threshold_.HasExplicitValue())
    threshold_.Acquire(spec_, ws, N, TensorShape<1>{ndim});

  if (weights_.HasExplicitValue() && classes_.HasExplicitValue()) {
    DALI_ENFORCE(weights_.get().shape == classes_.get().shape, make_string(
      "If both ``classes`` and ``class_weights`` are provided, their shapes must match. Got:"
      "\n  classes.shape = ", classes_.get().shape,
      "\n  weights.shape = ", weights_.get().shape));
  }
}


#define INPUT_TYPES (bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t)

namespace detail {

template <typename T>
void FindLabels(std::unordered_set<int> &labels, const T *data, int64_t N) {
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

}  // namespace detail

template <typename BlobLabel>
template <typename T>
void RandomObjectBBox::SampleContext<BlobLabel>::FindLabels(const InTensorCPU<T> &in) {
  labels.clear();
  int64_t N = in.num_elements();
  if (!N)
    return;

  constexpr int64_t min_chunk_size = 1<<16;
  int num_chunks = std::min<int>(2*thread_pool->NumThreads(), div_ceil(N, min_chunk_size));
  const T *data = in.data;

  tmp_labels.resize(num_chunks);

  for (int i = 0; i < num_chunks; i++) {
    int64_t start = N * i / num_chunks;
    int64_t end = N * (i + 1)  / num_chunks;
    thread_pool->AddWork([=, this](int) {
      auto &lbl = tmp_labels[i];
      lbl.clear();
      detail::FindLabels(lbl, data + start, end - start);
    });
  }
  thread_pool->RunAll();

  for (auto &tmp : tmp_labels) {
    for (auto l : tmp)
      labels.insert(l);
  }
}


template <typename Out, typename In>
void FilterByLabel(Out *out, const In *in, int64_t N, In label) {
  for (int64_t i = 0; i < N; i++) {
    out[i] = in[i] == label;
  }
}

template <typename Out, typename In>
void FilterByLabel(ThreadPool *tp,
                   const OutTensorCPU<Out> &out, const InTensorCPU<In> &in, same_as_t<In> label) {
  assert(out.shape == in.shape);
  int64_t N = in.num_elements();
  const int64_t min_chunk_size = 1<<16;
  int num_chunks = std::min<int>(tp->NumThreads(), div_ceil(N, min_chunk_size));
  if (num_chunks == 1) {
    FilterByLabel(out.data, in.data, N, label);
  } else {
    for (int i = 0; i < num_chunks; i++) {
      int64_t start = N * i / num_chunks;
      int64_t end = N * (i + 1) / num_chunks;
      auto *out_start = out.data + start;
      auto *in_start = in.data + start;
      int64_t n = end - start;
      tp->AddWork([=](int) {
        FilterByLabel(out_start, in_start, n, label);
      });
    }
    tp->RunAll();
  }
}


template <typename Lo, typename Hi>
void StoreBox(const OutListCPU<int, 1> &out1,
              const OutListCPU<int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Lo &&start, Hi &&end) {
  assert(dali::size(start) == dali::size(end));
  int ndim = dali::size(start);
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

void RandomObjectBBox::ClassInfo::Init(const int *bg_ptr,
                                       const InTensorCPU<int, 1> &cls_tv,
                                       const InTensorCPU<float, 1> &weight_tv) {
  Reset();
  background = bg_ptr ? *bg_ptr : 0;

  if (cls_tv.data) {
    int ncls = cls_tv.shape[0];
    classes.resize(ncls);
    if (!weight_tv.data) {
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
        if (bg_ptr) {
          // Background was explicitly specified this way - that's an error
          DALI_FAIL(make_string("Class label ", classes[i],
            " coincides with background label - please specify a different background label or"
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
  if (weight_tv.data) {
    int ncls = weight_tv.shape[0];
    weights.resize(ncls);
    for (int i = 0; i < ncls; i++)
      weights[i] = weight_tv.data[i];
    if (!cls_tv.data) {
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

void RandomObjectBBox::InitClassInfo(int sample_idx) {
  const int *bg = background_.HasExplicitValue() ? background_[sample_idx].data : nullptr;
  InTensorCPU<int, 1> class_tv;
  InTensorCPU<float, 1> weight_tv;
  if (classes_.HasExplicitValue() && !ignore_class_) class_tv  = classes_[sample_idx];
  if (weights_.HasExplicitValue() && !ignore_class_) weight_tv = weights_[sample_idx];
  class_info_.Init(bg, class_tv, weight_tv);
}

template <int ndim>
int RandomObjectBBox::PickBox(span<Box<ndim, int>> boxes, int sample_idx) {
  auto beg = boxes.begin();
  auto end = boxes.end();
  if (threshold_.HasExplicitValue()) {
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
    return vol_idx[dist(rng_[sample_idx])].second;
  } else {
    std::uniform_int_distribution<int> dist(0, n-1);
    return dist(rng_[sample_idx]);
  }
}

template <typename BlobLabel>
void RandomObjectBBox::GetBoxes(SampleContext<BlobLabel> &ctx, int nblobs) {
  ctx.box_data.clear();
  if (!nblobs)
    return;

  int ndim = ctx.blobs.dim();
  ctx.box_data.resize(2*ndim*nblobs);

  VALUE_SWITCH(ndim, static_ndim, (1, 2, 3, 4, 5, 6),
    (
      auto *box_data = reinterpret_cast<Box<static_ndim, int>*>(ctx.box_data.data());
      auto boxes = make_span(box_data, nblobs);
      GetLabelBoundingBoxes(boxes, ctx.blobs.template to_static<static_ndim>(), -1,
                            *ctx.thread_pool);
    ), (  // NOLINT
      DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim, "; must be 1..6"));
    )  // NOLINT
  );  // NOLINT
}

template <typename BlobLabel>
bool RandomObjectBBox::PickBox(SampleContext<BlobLabel> &ctx) {
  int ndim = ctx.blobs.dim();
  int nblobs = ctx.box_data.size() / (2 * ndim);
  if (!nblobs)
    return false;

  VALUE_SWITCH(ndim, static_ndim, (1, 2, 3, 4, 5, 6),
    (
      auto *box_data = reinterpret_cast<Box<static_ndim, int>*>(ctx.box_data.data());
      auto boxes = make_span(box_data, nblobs);
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

void RandomObjectBBox::ClassInfo::Reset() {
  classes.clear();
  weights.clear();
  cdf.clear();
}

void RandomObjectBBox::ClassInfo::FromLabels(const LabelSet &labels) {
  classes.clear();
  weights.clear();
  for (auto cls : labels) {
    classes.push_back(cls);
    weights.push_back(1);
  }
  // We need to sort, because the order in `context.labels` depends on its previous contents
  // (it changes with the number of bins in the hastable).
  // We want don't need any particular order here, but it must be deterministic
  // - thus, sort.
  std::sort(classes.begin(), classes.end());
}

void RandomObjectBBox::ClassInfo::DisableAbsentClasses(const LabelSet &labels) {
  for (int i = 0; i < static_cast<int>(classes.size()); i++) {
    if (!labels.count(classes[i]))
      weights[i] = 0;  // label not present - reduce its weight to 0
  }
}

template <typename BlobLabel, typename T>
bool RandomObjectBBox::PickForegroundBox(
      SampleContext<BlobLabel> &context, const InTensorCPU<T> &input) {
  InitClassInfo(context.sample_idx);
  context.class_label = class_info_.background;
  auto &tp = *context.thread_pool;

  CacheEntry *cache_entry = nullptr;
  kernels::fast_hash_t hash = {};
  if (use_cache_) {
    fast_hash(hash, input.data, input.num_elements() * sizeof(T));
    cache_entry = &cache_[hash];
  }

  if (ignore_class_) {
    if (!cache_entry || !cache_entry->Get(context.box_data, class_info_.background)) {
      int nblobs = LabelConnectedRegions(context.blobs, input, tp, -1, class_info_.background);
      GetBoxes(context, nblobs);
      if (cache_entry)
        cache_entry->Put(class_info_.background, context.box_data);
    }
    return PickBox(context);
  } else {
    if (cache_entry && !cache_entry->labels.empty()) {
      context.labels = cache_entry->labels;
    } else {
      context.FindLabels(input);
      context.labels.erase(class_info_.background);
      if (cache_entry)
        cache_entry->labels = context.labels;
    }

    if (!classes_.HasExplicitValue() && !weights_.HasExplicitValue()) {
      class_info_.FromLabels(context.labels);
    } else {
      class_info_.DisableAbsentClasses(context.labels);
    }

    while (class_info_.CalculateCDF()) {
      if (!context.PickClassLabel(class_info_, rng_[context.sample_idx]))
        return false;

      assert(context.class_label != class_info_.background);

      if (!cache_entry || !cache_entry->Get(context.box_data, context.class_label)) {
        FilterByLabel(context.thread_pool, context.filtered, input, context.class_label);
        int nblobs = LabelConnectedRegions<BlobLabel, uint8_t, -1>(
                context.blobs, context.filtered, tp, -1, 0);
        GetBoxes(context, nblobs);
        if (cache_entry)
          cache_entry->Put(context.class_label, context.box_data);
      }

      if (PickBox(context))
        return true;

      // we couldn't find a satisfactory blob in this class, so let's exclude it and try again
      class_info_.weights[context.class_idx] = 0;
      context.class_label = class_info_.background;
    }
    // we've run out of classes and still there's no good blob
    return false;
  }
}

template <typename BlobLabel>
bool RandomObjectBBox::PickForegroundBox(SampleContext<BlobLabel> &context) {
  bool ret = false;
  TYPE_SWITCH(context.input.type(), type2id, T, INPUT_TYPES,
    (ret = PickForegroundBox(context, view<const T>(context.input));),
    (DALI_FAIL(make_string("Unsupported input type: ", context.input.type())))
  );  // NOLINT
  return ret;
}

void RandomObjectBBox::AllocateTempStorage(const TensorList<CPUBackend> &input) {
  int64_t max_blob_bytes = 0;
  int64_t max_filtered_bytes = 0;
  int N = input.num_samples();
  for (int i = 0; i < N; i++) {
    int64_t vol = input[i].shape().num_elements();
    int label_size = vol > 0x80000000 ? 8 : 4;
    int64_t blob_bytes = vol * label_size;
    if (blob_bytes > max_blob_bytes)
      max_blob_bytes = blob_bytes;
    if (vol > max_filtered_bytes)
      max_filtered_bytes = vol;
  }
  auto grow = [](Tensor<CPUBackend> &tensor, int64_t bytes) {
    int64_t cap = tensor.capacity();
    if (cap < bytes) {
      tensor.reserve(std::max(2*cap, bytes));
    }
  };
  grow(tmp_blob_storage_, max_blob_bytes);
  grow(tmp_filtered_storage_, max_filtered_bytes);
}

void RandomObjectBBox::RunImpl(Workspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  int N = input.num_samples();
  if (N == 0)
    return;

  int ndim = input.sample_dim();
  auto &tp = ws.GetThreadPool();

  OutListCPU<int, 1> out1 = view<int, 1>(ws.Output<CPUBackend>(0));
  OutListCPU<int, 1> out2;
  if (format_ != Out_Box)
    out2 = view<int, 1>(ws.Output<CPUBackend>(1));
  OutListCPU<int, 0> class_label_out;
  if (HasClassLabelOutput())
    class_label_out = view<int, 0>(ws.Output<CPUBackend>(class_output_idx_));

  TensorShape<> default_anchor;
  default_anchor.resize(ndim);

  default_context_.thread_pool = &tp;
  huge_context_.thread_pool = &tp;

  AllocateTempStorage(input);

  std::uniform_real_distribution<> foreground(0, 1);
  for (int i = 0; i < N; i++) {
    bool fg = foreground(rng_[i]) < foreground_prob_[i].data[0];
    if (!fg) {
      StoreBox(out1, out2, format_, i, default_anchor, input.tensor_shape(i));
      if (HasClassLabelOutput()) {
        InitClassInfo(i);
        class_label_out.data[i][0] = class_info_.background;
      }
    } else {
      // Blobl labeling uses disjoint sets as an implementation - as such,
      // the label type must be large enough to store flattened element indices within
      // the input tensor.
      // We want to limit the size of this auxiliary storage to limit memory traffic.
      // To that end, when the indices fit in int32_t, we use that type for the labels,
      // otherwise we fall back to int64_t.
      auto blob_label = (input[i].shape().num_elements() > 0x80000000) ? DALI_INT64 : DALI_INT32;
      TYPE_SWITCH(blob_label, type2id, BlobLabel, (int32_t, int64_t), (
        auto &ctx = GetContext(BlobLabel());
        ctx.Init(i, input[i], &tp, tmp_filtered_storage_, tmp_blob_storage_);
        ctx.out1 = out1[i];
        if (out2.num_samples() > 0)
          ctx.out2 = out2[i];

        if (PickForegroundBox(ctx)) {
          assert(ctx.class_label != class_info_.background || ignore_class_);
          StoreBox(out1, out2, format_, i, ctx.selected_box);
        } else {
          assert(ctx.class_label == class_info_.background);
          StoreBox(out1, out2, format_, i, default_anchor, input.tensor_shape(i));
        }

        if (HasClassLabelOutput())
          class_label_out.data[i][0] = ctx.class_label;
      ), (assert("!Internal error");));  // NOLINT
    }
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(segmentation__RandomObjectBBox, RandomObjectBBox, CPU);

}  // namespace dali
