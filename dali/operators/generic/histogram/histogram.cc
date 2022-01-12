// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/generic/histogram/histogram.h"

#include "dali/kernels/common/copy.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/pipeline/operator/op_schema.h"


#include "dali/core/tensor_shape_print.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/core/core_c.h"

#include <iostream>

using namespace dali;
using namespace dali::hist_detail;

#define id_(x) x

#define HistogramOpName id_(histogram__Histogram)
#define UniformHistogramOpName id_(histogram__UniformHistogram)

#define str_next_(x) #x
#define str_(x) str_next_(x)

namespace {

constexpr const char histogramOpString[] = str_(HistogramOpName);
constexpr const char unifromHistogramOpString[] = str_(UniformHistogramOpName);

static constexpr int HISTOGRAM_MAX_CH = CV_MAX_DIM;

std::vector<float> GetFlattenedRanges(int sample, const workspace_t<CPUBackend> &ws) {
  std::vector<float> ranges;

  for (int r = 1; r < ws.NumInput(); ++r) {
    auto &dim_ranges = ws.template Input<CPUBackend>(r);
    auto range_view = view<const float>(dim_ranges);
    for (int i = 0; i < range_view[sample].num_elements(); ++i) {
      ranges.push_back(range_view.tensor_data(sample)[i]);
    }
  }
  return ranges;
}

template <typename Ty_>
struct CVMatType {
  static int get(int) {
    DALI_ENFORCE(false, "Unreachable - invalid type");
  }
};

template <>
struct CVMatType<uint8_t> {
  static int get(int nchannel) noexcept {
    return CV_MAKETYPE(CV_8U, nchannel);
  }
};

template <>
struct CVMatType<uint16_t> {
  static int get(int nchannel) noexcept {
    return CV_MAKETYPE(CV_16U, nchannel);
  }
};

template <>
struct CVMatType<float> {
  static int get(int nchannel) noexcept {
    return CV_MAKETYPE(CV_32F, nchannel);
  }
};

template <>
struct CVMatType<double> {
  static int get(int nchannel) noexcept {
    return CV_MAKETYPE(CV_64F, nchannel);
  }
};

template <typename Type, typename ScratchAlloc, typename Coll>
TensorListView<StorageCPU, const Type> transpose_view(
    dali::ThreadPool &thread_pool, ScratchAlloc &scratch,
    const TensorListView<StorageCPU, const Type> &in_view, const Coll &transpose_axes_order) {
  const auto &in_shapes = in_view.shape;

  TensorListShape<> transposed_shapes;
  permute_dims(transposed_shapes, in_shapes, transpose_axes_order);
  std::vector<Type *> tmp_pointers;
  tmp_pointers.reserve(transposed_shapes.num_samples());

  for (int i = 0; i < transposed_shapes.num_samples(); ++i) {
    auto tmp = scratch.template AllocTensor<mm::memory_kind::host, Type>(transposed_shapes[i]);
    tmp_pointers.push_back(tmp.data);
  }

  TensorListView<StorageCPU, Type> transpose_out_view(std::move(tmp_pointers),
                                                      std::move(transposed_shapes));

  for (int i = 0; i < transpose_out_view.num_samples(); ++i) {
    thread_pool.AddWork([&, i](int thread_id) {
      auto perm = make_span(transpose_axes_order);
      kernels::Transpose(transpose_out_view[i], in_view[i], perm);
    });
  }
  thread_pool.RunAll(true);
  return reinterpret<const Type>(transpose_out_view, transpose_out_view.shape);
}

template <typename Type>
void run_identity(ThreadPool &thread_pool, const TensorListView<StorageCPU, const Type> &in_view,
                  TensorListView<StorageCPU, Type> &out_view) {
  for (int i = 0; i < in_view.shape.num_samples(); ++i) {
    thread_pool.AddWork([&, i](int thread_id) { kernels::copy(out_view[i], in_view[i]); });
  }
  thread_pool.RunAll(true);
}

void ValidateInputShape(const TensorListShape<> &in_sh, int hist_dim, int channel_axis) {
  if (channel_axis < 0) {
    assert(hist_dim == 1);
    return;
  }
  for (int i=0; i<in_sh.num_samples(); ++i) {
    auto sh = in_sh.tensor_shape_span(i);
    assert(channel_axis < sh.size());
    int num_channels = sh[channel_axis];
    DALI_ENFORCE(hist_dim == num_channels,
      make_string("Number of channels in dimension specified as channel axis (", channel_axis,
        ") doesn't match histogram dimensionality, (", num_channels, " vs ", hist_dim, ")"));
  }
}

}  // namespace

HistReductionAxesHelper::HistReductionAxesHelper(const OpSpec &spec) : detail::AxesHelper(spec) {
  has_channel_axis_arg_ = spec.TryGetArgument(channel_axis_, "channel_axis");
  has_channel_axis_name_arg_ = spec.TryGetArgument(channel_axis_name_, "channel_axis_name");

  DALI_ENFORCE(!has_channel_axis_arg_ || !has_channel_axis_name_arg_,
               "Arguments `channel_axis` and `channel_axis_name` are mutually exclusive");
}

void HistReductionAxesHelper::PrepareChannelAxisArg(const TensorLayout &layout,
                                                    const SmallVector<bool, 6> &reduction_axes_mask,
                                                    int hist_dim, bool implicit_axes) {
  const int sample_dim = reduction_axes_mask.size();
  const bool has_channel_axis = (has_channel_axis_name_arg_ || has_channel_axis_arg_);

  if (hist_dim > 1) {
    DALI_ENFORCE(has_channel_axis,
                 "One of arguments `channel_axis` and `channel_axis_name` should be specified for "
                 "multidimensional histograms!");

    if (has_channel_axis_name_arg_) {
      auto indices = GetDimIndices(layout, channel_axis_name_);
      DALI_ENFORCE(indices.size() == 1,
                   "Single axis name should be specified as `channel_axis_name`");
      channel_axis_ = indices[0];
    } else {
      DALI_ENFORCE(channel_axis_ < sample_dim,
                   make_string("Invalid axis specified for argument `channel_axis` (is ",
                               channel_axis_, " and should be less than ", sample_dim, ")"));
    }
    DALI_ENFORCE(
        !reduction_axes_mask[channel_axis_] || implicit_axes,
        make_string("Axis ", channel_axis_,
                    " can be eigther reduction axis `axes` or `channel_axis`, not both"));
  } else if (has_channel_axis) {
    if (has_channel_axis_name_arg_ && channel_axis_ == -1) {
      auto indices = GetDimIndices(layout, channel_axis_name_);
      if (!indices.empty())
        channel_axis_ = indices[0];
    }
    DALI_ENFORCE(hist_dim != 1 || (channel_axis_ < 0 && hist_dim == 1),
                 "None of `channel_axis` and `channel_axis_name` arguments should be specified for "
                 "single dimensional histograms!");
    channel_axis_ = -1;
  }
}

void HistReductionAxesHelper::PrepareReductionAxes(const TensorLayout &layout, int sample_dim,
                                                   int hist_dim) {
  assert(hist_dim > 0);

  PrepareAxes(layout, sample_dim);

  SmallVector<bool, 6> reduction_axes_mask;
  reduction_axes_mask.resize(sample_dim, false);

  for (int axis : axes_) {
    reduction_axes_mask[axis] = true;
  }

  bool explicit_axes = has_axes_arg_ || has_axis_names_arg_;
  PrepareChannelAxisArg(layout, reduction_axes_mask, hist_dim, !explicit_axes);

  // If axes were not specified explicitly, we consider all but channel_axis_ reduction axes.
  if (!explicit_axes) {
    for (int i = 0; i < sample_dim; ++i) {
      if (i != channel_axis_) {
        reduction_axes_mask[i] = true;
      }
      else {
        reduction_axes_mask[i] = false;
      }
    }
  }

  axes_order_.clear();
  axes_order_.reserve(sample_dim);

  // Collect non-reduction axes as outer tensor dimension
  for (int i = 0; i < sample_dim; ++i) {
    if (!reduction_axes_mask[i] && i!=channel_axis_) {
      axes_order_.push_back(i);
    }
  }

  size_t num_non_reduction = axes_order_.size();

  // Collect reduction axes as inner tensor dimension
  for (int i = 0; i < sample_dim; ++i) {
    if (reduction_axes_mask[i]) {
      axes_order_.push_back(i);
    }
  }

  size_t num_reduction = axes_order_.size() - num_non_reduction;

  // For multi-dimensional histogram channel axis is most inner dimension
  if (channel_axis_ >= 0) {
    axes_order_.push_back(channel_axis_);
  }

  assert(axes_order_.size() == static_cast<size_t>(sample_dim));

  non_reduction_axes_ = span<int>(axes_order_.data(), num_non_reduction);
  reduction_axes_ = span<int>(axes_order_.data() + num_non_reduction, num_reduction);

  // TODO: verify first part of condition
  is_identity_ = non_reduction_axes_.size() == sample_dim && has_empty_axes_arg_;
}

bool HistReductionAxesHelper::NeedsTranspose() const {
  for (size_t i = 0; i < axes_order_.size(); ++i) {
    if (axes_order_[i] != i) {
      return true;
    }
  }
  return false;
}

HistogramCPU::HistogramCPU(const OpSpec &spec)
    : Operator<CPUBackend>(spec),
      hist_detail::HistReductionAxesHelper(spec),
      param_num_bins_("num_bins", spec) {
  uniform_ = spec.name() == unifromHistogramOpString;
  assert(spec.name() == unifromHistogramOpString || spec.name() == histogramOpString);
}

TensorListShape<> HistogramCPU::GetBinShapes(int num_samples) const {
  TensorListShape<> ret;
  ret.resize(num_samples, hist_dim_);

  for (int i = 0; i < num_samples; ++i) {
    TensorShape<> bin_shape{make_span(batch_bins_[i].data(), hist_dim_)};
    ret.set_tensor_shape(i, bin_shape);
  }
  return ret;
}

void HistogramCPU::PrepareReductionShapes(const TensorListShape<> &in_sh, OutputDesc &output_desc) {
  // Prepare input shapes, if reduction axes followd by channel axis are not inner-most
  // permutate axes to create transposed shape
  TensorListShape<> transposed_shapes;
  const bool needs_transpose = NeedsTranspose();

  if (needs_transpose) {
    permute_dims(transposed_shapes, in_sh, axes_order_);
  }

  const TensorListShape<> &input_shapes = needs_transpose ? transposed_shapes : in_sh;

  // Prepare output tensors shapes
  TensorListShape<> output_shapes, bin_shapes = GetBinShapes(input_shapes.num_samples());

  if (!non_reduction_axes_.empty()) {
    auto non_reduction_axes_shape = input_shapes.first(non_reduction_axes_.size());
    output_shapes.resize(input_shapes.num_samples(), non_reduction_axes_.size() + hist_dim_);

    for (int i = 0; i < input_shapes.num_samples(); ++i) {
      auto out_sh = shape_cat(non_reduction_axes_shape[i], bin_shapes[i]);
      output_shapes.set_tensor_shape(i, out_sh);
    }
  } else {
    output_shapes = std::move(bin_shapes);
  }

  // Simplify shapes so histogram of per reduction can be calculated easily.
  SubdivideTensorsShapes(input_shapes, output_shapes, output_desc);
}

void HistogramCPU::SubdivideTensorsShapes(const TensorListShape<> &input_shapes,
                                          const TensorListShape<> &output_shapes,
                                          OutputDesc &output_desc) {
  SmallVector<std::pair<int, int>, 2> in_collapse_groups, out_collapse_groups;
  TensorListShape<> norm_inputs, norm_outputs;

  int reduced_start = 1;
  if (non_reduction_axes_.empty()) {
    // Add unitary outer dimension for unfold
    norm_inputs.resize(input_shapes.num_samples(), input_shapes.sample_dim() + 1);
    norm_outputs.resize(output_shapes.num_samples(), output_shapes.sample_dim() + 1);
    for (int i = 0; i < input_shapes.num_samples(); ++i) {
      norm_inputs.set_tensor_shape(i, shape_cat(1, input_shapes[i]));
      norm_outputs.set_tensor_shape(i, shape_cat(1, output_shapes[i]));
    }
  } else {
    if (non_reduction_axes_.size() != 1) {
      // Add collapse group to collapse outer (non-reduction) dimensions
      in_collapse_groups.push_back(std::make_pair(0, non_reduction_axes_.size()));
      out_collapse_groups.push_back(std::make_pair(0, non_reduction_axes_.size()));
      reduced_start = non_reduction_axes_.size();
    }
    norm_inputs = input_shapes;
    norm_outputs = output_shapes;
  }

  if (reduction_axes_.size() > 1) {
    // Add collapse group to collapse inner (reduction) dimensions, possibly ommiting channels
    // dimension.
    in_collapse_groups.push_back(std::make_pair(reduced_start, reduction_axes_.size()));
  }

  norm_inputs = collapse_dims(norm_inputs, in_collapse_groups);
  norm_outputs = collapse_dims(norm_outputs, out_collapse_groups);

  auto splited_input_shapes = unfold_outer_dim(norm_inputs);
  auto splited_output_shapes = unfold_outer_dim(norm_outputs);

  std::vector<int> split_mapping;
  split_mapping.reserve(splited_input_shapes.num_samples());
  auto norm_non_reduced = norm_inputs.first(1);
  for (int i = 0; i < norm_non_reduced.num_samples(); ++i) {
    for (int j = 0; j < norm_non_reduced[i][0]; ++j) {
      split_mapping.push_back(i);
    }
  }

  split_mapping_ = std::move(split_mapping);
  splited_input_shapes_ = std::move(splited_input_shapes);
  splited_output_shapes_ = std::move(splited_output_shapes);

  output_desc.shape = std::move(output_shapes);
  output_desc.type = DALI_FLOAT;
}

bool HistogramCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                             const workspace_t<CPUBackend> &ws) {
  output_desc.resize(1);

  auto &input = ws.template Input<CPUBackend>(0);
  const size_t ndims = input.shape().sample_dim();

  hist_dim_ = ValidateRangeArguments(ws, input.num_samples());

  PrepareReductionAxes(input.GetLayout(), ndims, hist_dim_);
  ValidateInputShape(input.shape(), hist_dim_, channel_axis_);

  // If an empty reduction axes were specified, histogram calculation becomes identity operation
  if (is_identity_) {
    output_desc[0].type = input.type();
    output_desc[0].shape = input.shape();
  } else {
    PrepareReductionShapes(input.shape(), output_desc[0]);
  }

  return true;
}

void HistogramCPU::ValidateBinsArgument(const workspace_t<CPUBackend> &ws, int num_samples, int hist_dim) {
  assert(uniform_ && "Specified only for uniform histogram");
  param_num_bins_.Acquire(this->spec_, ws, num_samples, TensorShape<1>(hist_dim));
  auto bins_view = param_num_bins_.get();

  // This always passes meaning that num_bins is broadcasted when it shouldn't for ND-histogram
  DALI_ENFORCE(bins_view.shape.num_elements()/num_samples == hist_dim);

  DALI_ENFORCE(bins_view.num_elements() % num_samples == 0,
               make_string("Histogram bins should be an array of bins per sample",
                           bins_view.num_elements()));

  batch_bins_.resize(0);
  batch_bins_.reserve(num_samples);
  for (int i=0; i<num_samples; ++i) {
    SmallVector<int, 3> bins;
    for (int j=0; j<hist_dim; ++j) {
      bins.push_back(bins_view.tensor_data(i)[j]);
    }
    batch_bins_.push_back(std::move(bins));
  }
}

void HistogramCPU::InferBinsArgument(const workspace_t<CPUBackend> &ws, int num_samples, int hist_dim) {
  assert(!uniform_ && "Can infer only for non-uniform histograms");

  SmallVector<int, 3> bins;
  for (int i=0; i < hist_dim; ++i) {
    const auto &dim_ranges = ws.template Input<CPUBackend>(1 + i);
    bins.push_back(dim_ranges.shape().num_samples());
  }
  batch_bins_.reserve(num_samples);

  for (int i=0; i<num_samples; ++i) {
    batch_bins_.push_back(bins);
  }
}

int HistogramCPU::ValidateUniformRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples) {
  assert(ws.NumInput() == 3);

  const auto &ranges_lo = ws.template Input<CPUBackend>(1);
  const auto &ranges_hi = ws.template Input<CPUBackend>(2);

  auto lo_view = view<const float>(ranges_lo);
  auto hi_view = view<const float>(ranges_hi);
  
  const auto &lo_sh = lo_view.shape;
  const auto &hi_sh = hi_view.shape;

  DALI_ENFORCE(is_uniform(lo_sh), "Histogram bins ranges must be uniform across batch!");
  DALI_ENFORCE(lo_sh == hi_sh, "Expected matching histogram bin upper and lower range shapes!");

  int hist_dim = ranges_lo.shape().num_elements()/num_samples;

  DALI_ENFORCE(hist_dim <= HISTOGRAM_MAX_CH,
               make_string("Unsupported histogram dimensionality, should be not greater than ",
                           HISTOGRAM_MAX_CH));
  DALI_ENFORCE(hist_dim >= 1, "Expected histogram bin ranges for at least one histogram dimension");

  ValidateBinsArgument(ws, num_samples, hist_dim);

  batch_ranges_.reserve(num_samples);

  for (int i=0; i<num_samples; ++i) {
    std::vector<float> dim_lo_hi(size_t(2*hist_dim));
    for (int j=0; j<hist_dim; ++j) {
      dim_lo_hi[2*j] = lo_view[i].data[j];
      dim_lo_hi[2*j + 1] = hi_view[i].data[j];
    }
    batch_ranges_.push_back(std::move(dim_lo_hi));
  }

  return hist_dim;
}

int HistogramCPU::ValidateNonUniformRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples) {
  assert(!uniform_);

  int hist_dim = ws.NumInput() - 1;

  DALI_ENFORCE(hist_dim <= HISTOGRAM_MAX_CH,
               make_string("Histogram dimensionality should not be greater than ", HISTOGRAM_MAX_CH));
  DALI_ENFORCE(hist_dim >= 1, "Ranges for at least one histogram should be specified");

  InferBinsArgument(ws, num_samples, hist_dim);

  int nsamples = ws.template Input<CPUBackend>(0).shape().num_samples();

  for (int r = 1; r < ws.NumInput(); ++r) {
    auto &dim_ranges = ws.template Input<CPUBackend>(r);
    DALI_ENFORCE(dim_ranges.type() == DALI_FLOAT,
                 make_string("Histogram bin ranges should be 32 bit floating-point numbers"));
    auto sh_ranges = dim_ranges.shape();
    DALI_ENFORCE(sh_ranges.sample_dim(),
                 make_string("Histogram bin ranges for ", r,
                               " dimension should be one-dimensional array"));

    DALI_ENFORCE(
      sh_ranges.num_elements() == 2 * nsamples,
        make_string("Bin ranges for uniform histogram should consist of lower and upper bound",
                    dim_ranges.num_samples()));

    for (int i = 0; i < nsamples; ++i) {
      batch_ranges_.push_back(GetFlattenedRanges(i, ws));
    }
  }
  return hist_dim;
}

int HistogramCPU::ValidateRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples) {
  if (uniform_) {
    return ValidateUniformRangeArguments(ws, num_samples);
  } else {
    return ValidateNonUniformRangeArguments(ws, num_samples);
  }
  // TODO: validate that bin ranges grow monoticaly
  // currently its checked later on by OpenCV
}

void HistogramCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  int nsamples = input.shape().num_samples();
  auto &thread_pool = ws.GetThreadPool();
  std::vector<int> all_channels;

  for (int j = 0; j < hist_dim_; ++j) {
    all_channels.push_back(j);
  }

  TYPE_SWITCH(input.type(), type2id, Type, (uint8_t, uint16_t, float), (
  {
    auto in_view = view<const Type>(input);

    if (is_identity_) {
      auto out_view_id = view<Type>(output);
      run_identity<Type>(thread_pool, in_view, out_view_id);
       return;
    }

    auto out_view = view<float>(output);
    TensorListView<StorageCPU, const Type, DynamicDimensions> transposed_in_view;

    bool needs_transpose = NeedsTranspose();
    if (needs_transpose) {
      transpose_mem_.template Reserve<mm::memory_kind::host>(
        in_view.num_elements() * sizeof(Type));
      auto scratch = transpose_mem_.GetScratchpad();

      transposed_in_view =
      transpose_view(thread_pool, scratch, in_view, axes_order_);
    }

    auto splited_in_views = reinterpret<Type>(
      needs_transpose ? transposed_in_view : in_view, splited_input_shapes_);
    auto splited_out_views = reinterpret<float>(out_view, splited_output_shapes_);

    assert(splited_in_views.num_samples() == splited_out_views.num_samples());
    assert(split_mapping_.size() == size_t(splited_in_views.num_samples()));

    for (int i = 0; i < splited_in_views.num_samples(); ++i) {
      thread_pool.AddWork([&, i](int thread_id) {
        SmallVector<int, 2> in_sizes;

        auto in_shape_span = splited_in_views.tensor_shape_span(i);
        in_sizes.push_back(in_shape_span[0]);  // FIXME, volume?

        int in_type = CVMatType<Type>::get(hist_dim_);

        std::vector<cv::Mat> images = {
          cv::Mat(1, in_sizes.data(), in_type, splited_in_views[i].data)};

        cv::InputArray input_mat(images);

        std::vector<int> bins;
        auto out_shape_span = splited_out_views.tensor_shape_span(i);

        for (int j = 0; j < hist_dim_; ++j) {
          bins.push_back(out_shape_span[j]);
        }

        std::size_t sample_range = split_mapping_[i];
        const std::vector<float> &ranges = batch_ranges_[sample_range];

        cv::Mat mask;
        cv::Mat output_mat;

        cv::calcHist(input_mat, all_channels, mask, output_mat, bins, ranges, uniform_);

        assert(output_mat.isContinuous() && output_mat.type() == CV_32FC1);
        assert(output_mat.total() == size_t(splited_out_views[i].num_elements()));

        // OpenCV always allocates output array and we need to copy it.
        float *hist_data = output_mat.ptr<float>();
          TensorView<StorageCPU, float> hist_view(hist_data, splited_out_views[i].shape);
          kernels::copy(splited_out_views[i], hist_view);
      });
    }
    thread_pool.RunAll(true);

  }), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}

DALI_SCHEMA(HistogramBase)
    .AddOptionalArg<std::vector<int>>("axes",
                                      R"code(Axis or axes along which reduction is performed.

Not providing any axis results in reduction of all elements.)code",
                                      nullptr)
    .AddOptionalArg<TensorLayout>(
        "axis_names", R"code(Name(s) of the axis or axes along which the reduction is performed.

The input layout is used to translate the axis names to axis indices, for example ``axis_names="HW"`` with input
layout `"FHWC"` is equivalent to specifying ``axes=[1,2]``. This argument cannot be used together with ``axes``.

Please note that any reduction axis specified by `axes` or `axis_names` can not be a specified as
 ``channel_axis``, or ``channel_axis_name`` at same time.
)code",
        nullptr)
    .AddOptionalArg<int>("channel_axis", R"code(Specifies channel axis for multidimensional histogram
)code",
                         nullptr)
    .AddOptionalArg<std::string>("channel_axis_name",
                                 "Specifies channel axis name for mulitidimensional histogram",
                                 nullptr);

DALI_SCHEMA(HistogramOpName)
    .DocStr(R"code("Calculates 1D or ND histogram of the input tensor with non-uniform histogram bin ranges.

This version of histogram function in contrast to ``UniformHistogram`` requires specifing tensor(s) of histogram bin ranges.
Ranges of histogram must be specified as tensor number of bins plus one float32 elements.
The numbers specify bin boundries such that, first of them is bin inclusive lower bound and the second is exclusive upper bound and in same time lower bound of next bin.

For example range tensor with sequence:

``A, B, C, D``

would describe 3 bins ``[A, B) [B, C), [C, D)``

When calulating multidimensional (see below) ranges should be specified as an additional tensor argument for each dimension.

The histogram of the input tensor when ``channel_axis`` or ``channel_axis_name`` is specified is calculated as multidimensional
histogram ie. as if histogram would be calculated for each seperate channel of this axis.
If channel axis is not specified 1D histogram is calculated.
Current implentation supports up to 32 channels for histogram calculation.

Histogram calculation supports specifing arbitrary axes of reduction.

For example for tensor with layout "HWC" one could calculate different single and multidimentional histograms.

One could specify ``axes_names = "W"`` as an reduction axis tensor of "HC" histograms in dimension "W" would be calculated.
By not specifing reduction axis all axes are taken of consideration and single dimesional histogram of all axes is calculated.
    )code")
    .NumInput(2, 1 + HISTOGRAM_MAX_CH)
    .NumOutput(1)
    .AddParent("HistogramBase");

DALI_SCHEMA(UniformHistogramOpName)
    .DocStr(R"code(Calculates 1D or ND histogram of the input tensor with uniform histogram bin ranges.

Calculates histogram of of uniform bin ranges, second input tensor specifies lower bound of range of values and third argument specfies
upper range of values in each histogram dimension.

For example lower range (2nd tensor argument) ``[0]`` and upper range (3rd tensor argument) ``[255]`` and ``num_bins=[16]`` will calculate histogram
with 16 bins of uniformly subdivided in range ``[0, 255]``

For higher dimensional histogram each ``num_bins``, lower and upper range input tensor element describe such subdivision in corresponding dimension.

The histogram of the input tensor when ``channel_axis`` or ``channel_axis_name`` is specified is calculated as multidimensional
histogram ie. as if histogram would be calculated for each seperate channel of this axis.
If channel axis is not specified 1D histogram is calculated.
Current implentation supports up to 32 channels for histogram calculation.

Histogram calculation supports specifing arbitrary axes of reduction.

For example for tensor with layout "HWC" one could calculate different single and multidimentional histograms.

One could specify ``axes_names = "W"`` as an reduction axis tensor of "HC" histograms in dimension "W" would be calculated.
By not specifing reduction axis all axes are taken of consideration and single dimesional histogram of all axes is calculated.
)code")
    .NumInput(3)
    .NumOutput(1)
    .AddOptionalArg("num_bins", "An integer tensor of histogram bins for each histogram dimension", std::vector<int>(),
                    true)
    .AddParent("HistogramBase");

DALI_REGISTER_OPERATOR(HistogramOpName, HistogramCPU, CPU);
DALI_REGISTER_OPERATOR(UniformHistogramOpName, HistogramCPU, CPU);
