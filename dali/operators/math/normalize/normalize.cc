// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/math/normalize/normalize.h"
#include "dali/kernels/reduce/reduce.h"
#include "dali/kernels/normalize/normalize_cpu.h"
#include "dali/core/math_util.h"
#include "dali/operators/math/normalize/normalize_utils.h"

namespace dali {

DALI_SCHEMA(Normalize)
  .DocStr(R"(Normalizes the input by removing mean and dividing by standard deviation.

The mean and standard deviation can be calculated internally for specified subset of axes or
can be externally provided as `mean` and `stddev` arguments.

The normalization is done following the formula::

    out = scale * (in - mean) / stddev + shift

The expression assumes that *out*, *in* are equally shaped tensors whereas *mean* and *stddev*
may be either tensors of same shape or scalars or a mix of these. The expression follows
*numpy* broadcasting rules.

Sizes of (non-scalar) `mean` and `stddev` must have either extent of 1, if given axis is reduced,
or match the corresponding extent of the input. A dimension is considered reduced if it's listed
in `axes` or `axis_names`. If neither `axes` nor `axis_names` argument is
present, the set of reduced axes is inferred by comparing input shape to the shape of
mean/stddev arguments, but it is enforced that the set of reduced axes is the same for all tensors
in the batch.

Examples of valid argument combinations:

1. Per-sample normalization of dimensions 0 and 2::

    axes = 0,2                                        # optional
    input.shape = [ [480, 640, 3], [1080, 1920, 4] ]
    batch = False
    mean.shape =  [ [1, 640, 1], [1, 1920, 1] ]
    stddev = (not supplied)

With these shapes, batch normalization is not possible, because the non-reduced dimension
has different extent across samples.

2. Batch normalization of dimensions 0 and 1::

    axes = 0,1                                        # optional
    input.shape = [ [480, 640, 3], [1080, 1920, 3] ]
    batch = True
    mean = (scalar)
    stddev.shape =  [ [1, 1, 3] ] ]

For color images, this normalizes the 3 color channels separately, but across all
samples in the batch.
)")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .AddOptionalArg("batch", "If True, the mean and standard deviation are calculated across tensors "
    "in the batch. This also requires that the input sample shapes in the non-averaged axes match.",
    false)
  .AddOptionalArg<float>("mean", "Mean value to subtract from the data. It can be either a scalar "
    "or a batch of tensors with same dimensionality as the input and the extent in each dimension "
    "must either match that of the input or be equal to 1 (in which case the value will be "
    "broadcast in this dimension). If not specified, the mean is calculated from the input. "
    "Non-scalar mean cannot be used when `batch` argument is True.",
    0.0f, true)
  .AddOptionalArg<float>("stddev", "Standard deviation value to scale the data. For shape "
    "constraints, see *mean* argument. If not specified, the standard deviation is calculated "
    "from the input. Non-scalar mean cannot be used when `batch` argument is True.",
    0.0f, true)
  .AddOptionalArg("axes", "Indices of dimensions along which the input is normalized. By default, "
    "all axes are used. Axes can also be specified by name, see *axes_names*.",
    std::vector<int>{}, false)
  .AddOptionalArg<TensorLayout>("axis_names", "Names of the axes in the input - axis indices "
    "are taken from the input layout. This argument cannot be used together with *axes*.", "")
  .AddOptionalArg("shift", "The value to which the mean will map in the output. Useful for "
    "unsigned output types.", 0.0f, false)
  .AddOptionalArg("scale", "The scaling factor applied to the output. Useful for integral "
    "output types", 1.0f, false)
  .AddOptionalArg("dtype", "Output type. When using integral types, use *shift* and *scale* to "
    "improve usage of the output type's dynamic range. If dtype is an integral type, out of range "
    "values are clamped, and non-integer values are rounded to nearest integer.", DALI_FLOAT);

DALI_REGISTER_OPERATOR(Normalize, Normalize<CPUBackend>, CPU);

using namespace normalize;  // NOLINT

template <>
template <typename OutputType, typename InputType>
void Normalize<CPUBackend>::SetupTyped(const HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int nsamples = input.ntensor();
  int nthreads = ws.GetThreadPool().size();

  using Kernel = kernels::NormalizeCPU<OutputType, InputType, float>;
  kmgr_.Resize<Kernel>(nthreads, nsamples);

  kernels::KernelContext ctx;
  for (int i = 0; i < nsamples; i++) {
    kmgr_.Setup<Kernel>(i, ctx, data_shape_[i], param_shape_[batch_norm_ ? 0 : i]);
  }
}

template <>
void Normalize<CPUBackend>::FoldMeans() {
  assert(mean_.ntensor() > 0);
  SumSamples(view<float>(mean_));
  // calculate the normalization factor in double, then cast to float
  float den = static_cast<float>(1.0 / ReducedVolume(data_shape_, make_span(axes_)));
  auto sample0 = make_tensor_cpu(mean_.mutable_tensor<float>(0), mean_.shape()[0]);
  auto elems = sample0.num_elements();
  for (int i = 0; i < elems; i++) {
    sample0.data[i] *= den;
  }
}

template <>
void Normalize<CPUBackend>::FoldStdDev() {
  assert(inv_stddev_.ntensor() > 0);
  SumSamples(view<float>(inv_stddev_));
  // calculate the normalization factor in double, then cast to float
  auto v = ReducedVolume(data_shape_, make_span(axes_));
  float rdiv = static_cast<float>(1.0 / v);
  auto sample0 = make_tensor_cpu(inv_stddev_.mutable_tensor<float>(0), inv_stddev_.shape()[0]);
  auto elems = sample0.num_elements();
  ScaleRSqrtKeepZero(sample0.data, elems, rdiv, scale_);
}

template <>
template <typename OutputType, typename InputType>
void Normalize<CPUBackend>::RunTyped(HostWorkspace &ws) {
  ThreadPool &tp = ws.GetThreadPool();

  auto &input = ws.InputRef<CPUBackend>(0);
  TensorListView<StorageCPU, const InputType> in_view = view<const InputType>(input);

  auto &output = ws.OutputRef<CPUBackend>(0);
  TensorListView<StorageCPU, OutputType> out_view = view<OutputType>(output);

  int nsamples = input.ntensor();
  int nthreads = ws.GetThreadPool().size();

  float scalar_mean = 0;
  float scalar_inv_stddev = 1;

  kernels::InListCPU<float, -1> mean_view, inv_stddev_view;

  // consume arguments

  auto mutable_mean = view<float>(mean_);
  auto mutable_stddev = view<float>(inv_stddev_);

  if (has_scalar_mean_) {
    scalar_mean = spec_.GetArgument<float>("mean");
    if (!IsFullReduction()) {
      UniformFill(mean_, scalar_mean);
      mean_view = view<const float>(mean_);
    } else {
      assert(param_shape_.num_elements() == param_shape_.num_samples());
      mean_view.shape = param_shape_;
      mean_view.data.resize(param_shape_.num_samples());
      for (auto &d : mean_view.data)
        d = &scalar_mean;
    }
  } else if (has_tensor_mean_) {
    mean_view = mean_input_;
  } else {
    assert(ShouldCalcMean());
    mean_view = mutable_mean;
  }

  if (has_scalar_stddev_) {
    scalar_inv_stddev = scale_ / spec_.GetArgument<float>("stddev");
    if (!IsFullReduction()) {
      UniformFill(inv_stddev_, scalar_inv_stddev);
      inv_stddev_view = view<const float>(inv_stddev_);
    } else {
      assert(param_shape_.num_elements() == param_shape_.num_samples());
      inv_stddev_view.shape = param_shape_;
      inv_stddev_view.data.resize(param_shape_.num_samples());
      for (auto &d : inv_stddev_view.data)
        d = &scalar_inv_stddev;
    }
  } else if (has_tensor_stddev_) {
    inv_stddev_.Resize(param_shape_);
    mutable_stddev = view<float>(inv_stddev_);
    CalcInvStdDev(mutable_stddev, stddev_input_, scale_);
    inv_stddev_view = mutable_stddev;
  } else {
    assert(ShouldCalcStdDev());
    inv_stddev_view = mutable_stddev;
  }

  // When using batch normalization, we need to reduce the per-sample results
  // between calculating mean and standard deviation and between standard deviation
  // and rescaling the input.

  if (batch_norm_) {
    if (ShouldCalcMean()) {
      for (int i = 0; i < nsamples; i++) {
        tp.DoWorkWithID([&, i](int thread_idx) {
          kernels::Mean<float, InputType> mean;
          mean.Setup(mutable_mean[i], in_view[i], make_span(axes_));
          // Reset per-sample values, but don't postprocess
          mean.Run(true, false);
        });
      }
      tp.WaitForWork();
      // Aggregate and postprocess now
      FoldMeans();
    }

    if (ShouldCalcStdDev()) {
      auto sample_mean = mean_view[0];
      for (int i = 0; i < nsamples; i++) {
        tp.DoWorkWithID([&, i](int thread_idx) {
          kernels::Variance<float, InputType> stddev;
          stddev.Setup(mutable_stddev[i], in_view[i], make_span(axes_), sample_mean);
          // Reset per-sample values, but don't postprocess
          stddev.Run(true, false);
        });
      }
      tp.WaitForWork();
      // Aggregate and postprocess now - use inverse square root.
      FoldStdDev();
    }
  }

  assert(static_cast<int>(mean_view.data.size()) == mean_view.num_samples());
  assert(static_cast<int>(inv_stddev_view.data.size()) == inv_stddev_view.num_samples());


  for (int i = 0; i < nsamples; i++) {
    tp.DoWorkWithID([&, i](int thread_idx) {
      auto sample_mean = mean_view.num_samples() == 1 || batch_norm_
                              ? mean_view[0]
                              : mean_view[i];

      auto sample_inv_stddev = inv_stddev_view.num_samples() == 1 || batch_norm_
                              ? inv_stddev_view[0]
                              : inv_stddev_view[i];

      if (!batch_norm_) {
        if (ShouldCalcMean()) {
          kernels::Mean<float, InputType> mean;
          mean.Setup(mutable_mean[i], in_view[i], make_span(axes_));
          // Reset per-sample values and preprocess
          mean.Run(true, true);
          assert(sample_mean.data == mutable_mean[i].data);
        }

        if (ShouldCalcStdDev()) {
          kernels::Variance<float, InputType> stddev;
          stddev.Setup(mutable_stddev[i], in_view[i], make_span(axes_), sample_mean);
          // Reset per-sample values, but don't postprocess
          stddev.Run(true, false);
          // Fused postprocessing with inverse square root.
          SumSquare2InvStdDev(mutable_stddev[i], data_shape_[i], scale_);
          assert(sample_inv_stddev.data == mutable_stddev[i].data);
        }
      }
      kernels::KernelContext ctx;
      using Kernel = kernels::NormalizeCPU<OutputType, InputType, float>;
      kmgr_.Run<Kernel>(thread_idx, i, ctx,
          out_view[i], in_view[i], sample_mean, sample_inv_stddev, shift_);
    });
  }

  tp.WaitForWork();
}


}  // namespace dali
