// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dali/operators/image/convolution/filter.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(experimental__Filter)
    .DocStr(R"code(Convolves the image with the provided filter.

.. note::
  In fact, the operator computes a correlation, not a convolution,
  i.e. the order of filter elements is not flipped when computing the product of
  the filter and the image.

)code")
    .NumInput(2, 3)
    .NumOutput(1)
    .AllowSequences()
    .InputDox(0, "data", "TensorList", R"code(Batch of input samples.

Sample can be an image, a video or volumetric (3D) data.
Samples can contain channels: channel-first and channel-last layouts are supported.
In case of video/sequences, the frame extent must preced the channels extent, i.e.,
for example, a video with ``"FCHW"`` layout is supported, but ``"CFHW"`` samples are not.

Samples with the following types are supported:
int8, int16, uint8, uint16, float16, float32.

Please note that the intermediate type used for the computation is always float32.

.. note::
  The CPU variant does not support volumetric (3D) data, nor inputs of types: int8 and float16.
)code")
    .InputDox(1, "filter", "TensorList", R"code(Batch of filters.

For inputs with two spatial dimensions (images or video), each filter must be a 2D array
(or a sequence of 2D arrays to be applied
:func:`per-frame<nvidia.dali.fn.per_frame>` to a video input).
For volumetric inputs, the filter must be a 3D array.
The filter values must have float32 type.)code")
    .InputDox(2, "fill_value", "TensorList", R"code(Batch of scalars used for padding.

If ``"border"`` is set to ``"constant"``, the input samples will be padded with
the corresponding scalars when convolved with the filter.
The scalars must be of the same type as the input samples.
For video/sequence input, an array of scalars can be specified to be applied
:func:`per-frame<nvidia.dali.fn.per_frame>`.)code")
    .InputDevice(1, 3, InputDevice::MatchBackendOrCPU)
    .AddOptionalArg("anchor",
                    R"code(Specifies the position of the filter over the input.

If the filter size is ``(r, s)`` and the anchor is ``(a, b)``, the output
at position ``(x, y)`` is a product of the filter and the input rectangle spanned between the
corners: top-left ``(x - a, y - b)`` and bottom-right ``(x - a + r - 1, x - b + s - 1)``.

If the -1 (the default) is specifed, the middle (rounded down to integer)
of the filter extents is used, which, for odd sized filters, results in the filter
centered over the input.

The anchor must be, depending on the input dimensionality, a 2D or 3D point whose each extent lies
within filter boundaries (``[0, ..., filter_extent - 1]``). The ordering of anchor's extents
corresponds to the order of filter's extents.

The parameter is ignored in ``"valid"`` mode.
.)code",
                    std::vector<int>{-1}, true, true)
    .AddOptionalArg("border",
                    R"code(Controls how to handle out-of-bound filter positions over the sample.

Supported values are: ``"reflect_101"``, ``"reflect_1001"``, ``"wrap"``,
``"clamp"``, ``"constant"``.

- ``"reflect_101"`` (default), reflects the input but does not repeat the outermost
  values (``dcb|abcdefghi|hgf``).
- ``"reflect_1001"``: reflects the input including outermost values (``cba|abcdefghi|ihg``)
- ``"wrap"``: wraps the input (``ghi|abcdefghi|abc``).
- ``"clamp"``: the input is padded with outermost values (``aaa|abcdefghi|iii``).
- ``"constant"``: the input is padded with the user-provided scalar (zeros by default).
  within the sample.
)code",
                    "reflect_101")
    .AddOptionalArg("mode",
                    R"code(Supported values are: ``"same"`` and ``"valid"``.

- ``"same"`` (default): The input and output sizes are the same and `border` is used
  to handle out-of-bound filter positions.
- ``"valid"``: the output sample is cropped (by ``filter_extent - 1``) so that all
  filter positions lie fully within the input sample.
)code",
                    "same")
    .AddOptionalTypeArg("dtype", R"code(Output data type.
The output type can either be float or must be same as input type.
If not set, the input type is used.

.. note::
  The intermediate type used for actual computation is float32. If the output is of integral type,
  the values will be clamped to the output type range.
)code");

namespace filter {

namespace ocv {
using namespace boundary;  // NOLINT(build/namespaces)
struct BorderSimple {
  BorderSimple(int border_type) : border_type_{border_type} {}  // NOLINT(runtime/explicit)

  void operator()(const cv::Mat& in_img, cv::Mat& out_img, int d_depth, const cv::Mat& filter_mat,
                  int anchor_x, int anchor_y) {
    cv::filter2D(in_img, out_img, d_depth, filter_mat, {anchor_x, anchor_y}, 0, border_type_);
  }

 protected:
  int border_type_;
};


template <typename In>
struct BorderConstant {
  BorderConstant(In fill_val) : fill_val_{fill_val} {}  // NOLINT(runtime/explicit)

  void operator()(const cv::Mat& in_img, cv::Mat& out_img, int d_depth, const cv::Mat& filter_mat,
                  int anchor_x, int anchor_y) {
    if (fill_val_ == 0) {
      cv::filter2D(in_img, out_img, d_depth, filter_mat, {anchor_x, anchor_y}, 0,
                   cv::BORDER_CONSTANT);
    } else {
      cv::Mat in_padded;
      int top = anchor_y, bottom = filter_mat.rows - anchor_y - 1;
      int left = anchor_x, right = filter_mat.cols - anchor_x - 1;
      cv::copyMakeBorder(in_img, in_padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                         cv::Scalar(fill_val_, fill_val_, fill_val_));
      auto roi = cv::Rect(left, top, in_img.cols, in_img.rows);
      auto in_roi = in_padded(roi);
      cv::filter2D(in_roi, out_img, d_depth, filter_mat, {anchor_x, anchor_y}, 0);
    }
  }

 protected:
  In fill_val_;
};


struct BorderWrap {
  void operator()(const cv::Mat& in_img, cv::Mat& out_img, int d_depth, const cv::Mat& filter_mat,
                  int anchor_x, int anchor_y) {
    cv::Mat in_padded;
    int top = anchor_y, bottom = filter_mat.rows - anchor_y - 1;
    int left = anchor_x, right = filter_mat.cols - anchor_x - 1;
    cv::copyMakeBorder(in_img, in_padded, top, bottom, left, right, cv::BORDER_WRAP);
    auto roi = cv::Rect(left, top, in_img.cols, in_img.rows);
    auto in_roi = in_padded(roi);
    cv::filter2D(in_roi, out_img, d_depth, filter_mat, {anchor_x, anchor_y}, 0);
  }
};

struct BorderValidOnly {
  void operator()(const cv::Mat& in_img, cv::Mat& out_img, int d_depth, const cv::Mat& filter_mat,
                  int anchor_x, int anchor_y) {
    auto roi = cv::Rect(anchor_x, anchor_y, in_img.cols - filter_mat.cols + 1,
                        in_img.rows - filter_mat.rows + 1);
    auto in_roi = in_img(roi);
    cv::filter2D(in_roi, out_img, d_depth, filter_mat, {anchor_x, anchor_y}, 0);
  }
};

template <typename In, typename Cb>
void with_border_handler(bool is_valid_mode, BoundaryType border_type, int sample_idx,
                         TensorListView<StorageCPU, In, 0>& fill_values, Cb&& cb) {
  if (is_valid_mode) {
    cb(BorderValidOnly{});
  } else if (border_type == BoundaryType::CONSTANT) {
    BorderConstant<In> handler =
        fill_values.num_samples() == 0 ? 0 : fill_values[sample_idx].data[0];
    cb(handler);
  } else if (border_type == BoundaryType::WRAP) {
    cb(BorderWrap{});
  } else {
    cb(BorderSimple{OCVBorderForDALIBoundary(border_type)});
  }
}

}  // namespace ocv

template <typename Out, typename In, int num_seq_dims>
class FilterOpCpuImpl : public OpImplBase<CPUBackend> {
  static constexpr int axes = 2;

 public:
  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit FilterOpCpuImpl(const OpSpec* spec, InputDesc input_desc)
      : spec_{*spec},
        input_desc_{input_desc},
        anchor_arg_{"anchor", spec_},
        border_type_{parse_filter_border_type(spec_.GetArgument<std::string>("border"))} {}

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    const auto& input = ws.template Input<CPUBackend>(0);
    int num_samples = input.num_samples();
    anchor_arg_.Acquire(spec_, ws, num_samples, TensorShape<1>{axes});
    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape = infer_output_shape(input.shape(), ws.GetInputShape(1), input_desc_);
    return true;
  }

  void RunImpl(Workspace& ws) override {
    const auto& input = ws.template Input<CPUBackend>(0);
    const auto& filter = ws.template Input<CPUBackend>(1);
    auto& output = ws.template Output<CPUBackend>(0);
    output.SetLayout(input.GetLayout());
    auto in_view = view<const In>(input);
    auto filter_view = view<const float, axes>(filter);
    auto anchor_view = anchor_arg_.get();
    auto out_view = view<Out>(output);
    int num_samples = input.num_samples();
    auto& tp = ws.GetThreadPool();
    TensorListView<StorageCPU, const In, 0> fill_values;
    if (ws.NumInput() >= 3) {
      fill_values = view<const In, 0>(ws.template Input<CPUBackend>(2));
    }
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto planes_range = sequence_utils::unfolded_views_range<num_seq_dims>(out_view[sample_idx],
                                                                             in_view[sample_idx]);
      const auto& in_range = planes_range.template get<1>();
      auto slice_shape = in_range.SliceShape();
      // by `input_desc.num_seq_dims + input_desc.axes + input_desc.has_channels == ndim`
      // in ShouldExpand, the `DALI_ENFORCE(num_axes == 2)` in get_filter_cpu_op_impl
      // and `unfolded_views_range<num_seq_dims>` above
      int sample_dim = slice_shape.sample_dim();
      assert(sample_dim == 2 || sample_dim == 3);
      int num_channels = sample_dim == 2 ? 1 : slice_shape[2];
      DALI_ENFORCE(
          1 <= num_channels && num_channels <= CV_CN_MAX,
          make_string("The CPU filter operator supports images with number of channels in [1, ",
                      CV_CN_MAX, "] channels. However, the sample at index ", sample_idx, " has ",
                      num_channels, " channels."));
      DALI_ENFORCE(slice_shape[0] <= std::numeric_limits<int>::max() &&
                       slice_shape[1] <= std::numeric_limits<int>::max(),
                   make_string("The image height and width must not exceed the ",
                               std::numeric_limits<int>::max(), ". However, the sample at index ",
                               sample_idx, " has shape ", in_view[sample_idx].shape, "."));
      auto sample_filter = filter_view[sample_idx];
      auto sample_anchor = anchor_view[sample_idx];
      for (int dim = 0; dim < axes; dim++) {
        DALI_ENFORCE(
            -1 <= sample_anchor.data[dim] && sample_anchor.data[dim] < sample_filter.shape[dim],
            make_string("Anchor must lie within the filter. Got anchor ",
                        vec2{sample_anchor.data[0], sample_anchor.data[1]},
                        " with a filter of shape ", sample_filter.shape, " for sample of idx ",
                        sample_idx, "."));
      }
      ocv::with_border_handler(
          input_desc_.is_valid_mode, border_type_, sample_idx, fill_values, [&](auto ocv_handler) {
            for (auto&& views : planes_range) {
              tp.AddWork(
                  [this, views, sample_filter, sample_anchor, ocv_handler](int) {
                    auto& [sample_out, sample_in] = views;
                    RunSample(sample_out, sample_filter, sample_in, sample_anchor, ocv_handler);
                  },
                  in_range.SliceSize());
            }
          });
    }
    tp.RunAll();
  }

 protected:
  template <typename OcvHandler, int ndim>
  void RunSample(TensorView<StorageCPU, Out, ndim> out,
                 TensorView<StorageCPU, const float, axes> filter,
                 TensorView<StorageCPU, const In, ndim> in,
                 TensorView<StorageCPU, const int, 1> anchor, OcvHandler handler) {
    auto& filter_shape = filter.shape;
    int sample_dim = in.shape.sample_dim();
    int num_channels = sample_dim == 2 ? 1 : in.shape[2];
    assert((out.shape.sample_dim() == 2 ? 1 : out.shape[2]) == num_channels);
    int in_flag = type2ocv<In>::value(num_channels);
    int out_flag = type2ocv<Out>::value(num_channels);
    const cv::Mat in_img = CreateMatFromPtr(in.shape[0], in.shape[1], in_flag, in.data);
    const cv::Mat filter_mat =
        CreateMatFromPtr(filter_shape[0], filter_shape[1], CV_32F, filter.data);
    cv::Mat out_img = CreateMatFromPtr(out.shape[0], out.shape[1], out_flag, out.data);
    static_assert(std::is_same_v<In, Out> || std::is_same_v<Out, float>);
    int d_depth = std::is_same_v<In, Out> ? -1 : CV_32F;
    int anchor_y = anchor.data[0] < 0 ? filter_shape[0] / 2 : anchor.data[0];
    int anchor_x = anchor.data[1] < 0 ? filter_shape[1] / 2 : anchor.data[1];
    handler(in_img, out_img, d_depth, filter_mat, anchor_x, anchor_y);
  }

  const OpSpec& spec_;
  InputDesc input_desc_;
  ArgValue<int, 1> anchor_arg_;
  BoundaryType border_type_;
  std::vector<ivec2> anchors_;
};

template <typename Out, typename In>
std::unique_ptr<OpImplBase<CPUBackend>> get_filter_cpu_op_impl(const OpSpec& spec_,
                                                               const InputDesc& input_desc) {
  int num_axes = input_desc.axes;
  std::string hint;
  if (num_axes == 3) {
    hint =
        "If you are processing images with channels, please make sure to mark the "
        "layout accordingly (`HWC` or `CHW`).";
  }
  DALI_ENFORCE(num_axes == 2,
               make_string("Unsupported input data dimensionality. ", "Got input with ", num_axes,
                           " spatial dimensions. CPU filter operator supports only 2-dimensional ",
                           "convolutions. ", hint));
  VALUE_SWITCH(input_desc.num_seq_dims, NumSeqDims, (0, 1, 2), (
      return std::make_unique<FilterOpCpuImpl<Out, In, NumSeqDims>>(&spec_, input_desc);
    ), (  // NOLINT
      DALI_FAIL(make_string("Unsupported number of outermost channel/frame dimensions: `",
        input_desc.num_seq_dims,
        "`. The input layout can start with at most two such extents "
        "(`F`, `C` or `FC`)."));
    ));  // NOLINT
}

}  // namespace filter

template <>
template <typename Out, typename In, typename W>
std::unique_ptr<OpImplBase<CPUBackend>> Filter<CPUBackend>::GetFilterImpl(
    const OpSpec& spec, const filter::InputDesc& input_desc) {
  static_assert(std::is_same_v<W, float>, "The CPU filter supports only float windows");
  return filter::get_filter_cpu_op_impl<Out, In>(spec, input_desc);
}

DALI_REGISTER_OPERATOR(experimental__Filter, Filter<CPUBackend>, CPU);

}  // namespace dali
