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

#include <cassert>
#include <random>
#include <utility>
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/imgproc/structure/label_bbox.h"
#include "dali/operators/segmentation/random_object_bbox.h"

namespace dali {

DALI_SCHEMA(segmentation__RandomObjectBBox)
  .DocStr(R"(Randomly selects an object from a mask and returns its bounding box.

This operator takes a labeled segmentation map as its input. With probability ``foreground_prob``
it randomly selects a label (uniformly or according to the distribution given as ``weights``),
extracts connected blobs of pixels with the selected label and randomly selects one of them
(with additional constraints given as ``k_largest`` and ``threshold``).
With probability 1-foreground_prob, entire area of the input is returned.)")
  .NumInput(1)
  .OutputFn([](const OpSpec& spec) {
    return spec.GetArgument<string>("format") == "box" ? 1 : 2;
  })
  .AddOptionalArg("ignore_class", R"(If True, all objects are picked with equal probability,
regardless of the class they belong to. Otherwise, a class is picked first and then object is
randomly selected from this class.

This argument is incompatible with ``classes`` or ``class_weights``.

.. note::
  This flag only affects the probability with which blobs are selected. It does not cause
  blobs of different classes to be merged.)", false)
  .AddOptionalArg("foreground_prob", "Probability of selecting a foreground bounding box.", 1.0f,
    true)
  .AddOptionalArg<vector<int>>("classes", R"(List of labels considered as foreground.

If left unspecified, all labels not equal to ``background`` are considered foreground)",
    nullptr, true)
  .AddOptionalArg("background", R"(Background label.

If left unspecified, it's either 0 or any value not in ``classes``.)", 0, true)
  .AddOptionalArg<vector<float>>("class_weights", R"(Relative probabilities of foreground classes.

Each value corresponds to a class label in ``classes`` or a 1-based number if ``classes`` are
not specified.
The values are normalized so that they sum to 1.)", nullptr, true)
  .AddOptionalArg<int>("k_largest", "If specified, at most k largest bounding boxes are consider",
    nullptr)
  .AddOptionalArg<vector<int>>("threshold", R"(Minimum extent(s) of the bounding boxes to return.

If current class doesn't contain any bounding box that meets this condition, the largest one
is returned.)", nullptr)
  .AddOptionalArg("format", R"(Format in which the data is returned.

Possible choices are::
  * "anchor_shape" (the default) - there are two outputs: anchor and shape
  * "start_end" - there are two outputs - bounding box start and one-past-end coordinates
  * "box" - ther'es one output that contains concatenated start and end coordinates
)", "anchor_shape");


bool RandomObjectBBox::SetupImpl(vector<OutputDesc> &out_descs, const HostWorkspace &ws) {
  out_descs.resize(format_ == Out_Box ? 1 : 2);
  auto in_shape = ws.InputRef<CPUBackend>(0).shape();
  int ndim = in_shape.sample_dim();
  int N = in_shape.num_samples();
  AcquireArgs(ws, N, ndim);
  out_descs[0].type = TypeTable::GetTypeInfo(DALI_INT32);
  out_descs[0].shape = uniform_list_shape<DynamicDimensions>(
      N, TensorShape<1>{ format_ == Out_Box ? 2*ndim : ndim });
  for (size_t i = 1; i < out_descs.size(); i++)
    out_descs[i] = out_descs[0];
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
      "If both ``classes`` and ``weights`` are provided, their shapes must match. Got:"
      "\n  classes.shape  = ", classes_.get().shape,
      "\n  weights.shape  = ", weights_.get().shape));
  }
}


#define INPUT_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t)

template <typename T>
void RandomObjectBBox::FindLabels(std::unordered_set<int> &labels, const T *data, size_t N) {
  if (!N)
    return;
  T prev = data[0];
  labels.insert(prev);
  for (size_t i = 1; i < N; i++) {
    if (data[i] == prev)
      continue;  // skip runs of equal labels
    labels.insert(data[i]);
    prev = data[i];
  }
}

template <typename Corner>
void StoreBox(const TensorListView<StorageCPU, int, 1> &out1,
              const TensorListView<StorageCPU, int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Corner &&start, Corner &&end) {
  assert(size(start) == size(end));
  int ndim = size(start);
  switch (format) {
    case Out_Box:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out1.data[sample_idx][i + ndim] = end[i];
      }
      break;
    case Out_AnchorShape:
      for (int i = 0; i < ndim; i++) {
        out1.data[sample_idx][i] = start[i];
        out2.data[sample_idx][i] = end[i] - start[i];
      }
      break;
    case Out_StartEnd:
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
void StoreBox(const TensorListView<StorageCPU, int, 1> &out1,
              const TensorListView<StorageCPU, int, 1> &out2,
              RandomObjectBBox::OutputFormat format,
              int sample_idx, Box &&box) {
  StoreBox(out1, out2, format, sample_idx, box.lo, box.hi);
}

template <typename T>
void RandomObjectBBox::ProcessSample(
      Context &context, const TensorView<StorageCPU, const T> &input) {
  FindLabels(context.labels, input);
  context.labels.erase(*background_[i].data);
  if (classes_.IsDefined()) {
    context.tmp_labels.clear();
    context.ordered_labels.clear();
    auto &cls_tv = classes_[context.sample_id];
    for (int i = 0; i < cls_tv.shape[0]; i++)
      if (context.labels.count(cls_tv.data[i])) {
        context.tmp_labels.insert(cls_tv.data[i]));
        cotnext.ordered_labels.push_back(i);
      }
    std::swap(context.labels, context.tmp_labels);
  }
  if (weights_.IsDefined()) {
    context.weight_dist = std::discrete_distribution(weights);

  }
}

void RandomObjectBBox::ProcessSample(Context &context) {
  TYPE_SWITCH(context, input.type().id(), type2id, input_type, INPUT_TYPES,
    (ProcessSample(context, view<iput_type>(context.input));),
    (DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))
  );  // NOLINT
}

void RandomObjectBBox::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  const auto &in_shape = input.shape();
  int N = in_shape.num_samples();
  int ndim = in_shape.sample_dim();
  auto &tp = ws.GetThreadPool();

  TensorListView<StorageCPU, int, 1> out1 = view<int, 1>(ws.OutputRef<CPUBackend>(0));
  TensorListView<StorageCPU, int, 1> out2;
  if (ws.NumOutput() > 1)
    out2 = view<int, 1>(ws.OutputRef<CPUBackend>(1));

  TensorShape<> default_anchor;
  default_anchor.resize(ndim);

  contexts_.resize(tp.size());

  std::uniform_real_distribution<> foreground(0, 1);
  for (int i = 0; i < N; i++) {
    bool fg = foreground(rngs_[i]) < *foreground_prob_[i].data;
    if (!fg) {
      StoreBox(out1, out2, format_, i, default_anchor, in_shape[i]);
    } else {
      tp.AddWork([&, i](int thread_idx) {
        Context &ctx = contexts[thread_idx];
        ctx.Init(sample_idx, input[i]);
        ctx.out1 = out1[sample_idx];
        if (out2.num_samples() > 0)
          ctx.out2 = out2[sample_idx];

        ProcessSample(context);
      }, volume(in_shape.tensor_shape_span(i)));
    }
  }
  tp.RunAll();
}


}  // namespace dali
