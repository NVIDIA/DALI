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
#include "dali/core/math_util.h"
#include "dali/core/tensor_layout.h"
#include "dali/kernels/normalize/normalize_cpu.h"
#include "dali/kernels/reduce/reduce_cpu.h"
#include "dali/operators/math/normalize/normalize_utils.h"

namespace dali {

DALI_SCHEMA(Normalize)
  .DocStr(R"code(Normalizes the input by removing the mean and dividing by the standard deviation.

The mean and standard deviation can be calculated internally for the specified subset
of axes or can be externally provided as the ``mean`` and ``stddev`` arguments.

The normalization is done following the formula::

  out = scale * (in - mean) / stddev + shift

The formula assumes that *out* and *in* are equally shaped tensors, but *mean* and *stddev* might
be either tensors of same shape, scalars, or a mix of these.

.. note::
  The expression follows the *numpy* broadcasting rules.

Sizes of the non-scalar ``mean`` and ``stddev`` must have an extent of 1, if given axis
is reduced, or match the corresponding extent of the input. A dimension is considered reduced
if it is listed in ``axes`` or ``axis_names``. If neither the ``axes`` nor the ``axis_names``
argument is present, the set of reduced axes is inferred by comparing the input
shape to the shape of the mean/stddev arguments, but the set of reduced axes must
be the same for all tensors in the batch.

Here are some examples of valid argument combinations:

1. Per-sample normalization of dimensions 0 and 2::

    axes = 0,2                                        # optional
    input.shape = [ [480, 640, 3], [1080, 1920, 4] ]
    batch = False
    mean.shape =  [ [1, 640, 1], [1, 1920, 1] ]
    stddev = (not supplied)

  With these shapes, batch normalization is not possible, because the non-reduced dimension
  has a different extent across samples.

2. Batch normalization of dimensions 0 and 1::

    axes = 0,1                                        # optional
    input.shape = [ [480, 640, 3], [1080, 1920, 3] ]
    batch = True
    mean = (scalar)
    stddev.shape =  [ [1, 1, 3] ] ]

For color images, this example normalizes the 3 color channels separately, but across all
samples in the batch.)code")
  .NumInput(1)
  .NumOutput(1)
  .SupportVolumetric()
  .AllowSequences()
  .AddOptionalArg("batch", R"code(If set to True, the mean and standard deviation are calculated
across tensors in the batch.

This argument also requires that the input sample shapes in the non-reduced axes match.)code",
    false)
  .AddOptionalArg<float>("mean", R"code(Mean value to be subtracted from the data.

The value can be a scalar or a batch of tensors with the same dimensionality as the input.
The extent in each dimension must match the value of the input or be equal to 1. If the
extent is 1, the value will be broadcast in this dimension. If the value is not specified, the
mean is calculated from the input. A non-scalar mean cannot be used when batch argument
is set to True.)code",
    nullptr, true)
  .AddOptionalArg<float>("stddev", R"code(Standard deviation value to scale the data.

See ``mean`` argument for more information about shape constraints. If a value is not specified,
the standard deviation is calculated from the input. A non-scalar ``stddev`` cannot be used when
``batch`` argument is set to True.)code",
    nullptr, true)
  .AddOptionalArg("axes", R"code(Indices of dimensions along which the input is normalized.

By default, all axes are used, and the axes can also be specified by name.
See ``axis_names`` for more informaton.)code", std::vector<int>{}, false)
  .AddOptionalArg<TensorLayout>("axis_names", R"code(Names of the axes in the input.

Axis indices are taken from the input layout, and this argument cannot be used with ``axes``.)code",
    "")
  .AddOptionalArg("shift", R"code(The value to which the mean will map in the output.

This argument is useful for unsigned output types.)code", 0.0f, false)
  .AddOptionalArg("scale", R"code(The scaling factor applied to the output.

This argument is useful for integral output types.)code", 1.0f, false)
  .AddOptionalArg("epsilon", R"code(A value that is added to the variance to avoid division by
small numbers.)code", 0.0f, false)
  .AddOptionalArg("ddof", R"code(Delta Degrees of Freedom for Bessel's correction.

The variance is estimated by using the following formula::

  sum(Xi - mean)**2 / (N - ddof).

This argument is ignored when an externally supplied standard deviation is used.)code", 0, false)
  .AddOptionalArg("dtype", R"code(Output data type.

When using integral types, use ``shift`` and ``scale`` to improve the usage of the output
type's dynamic range. If ``dtype`` is an integral type, out of range values are clamped,
and non-integer values are rounded to nearest integer.)code", DALI_FLOAT);

template <>
class Normalize<CPUBackend> : public NormalizeBase<CPUBackend> {
 public:
  explicit Normalize(const OpSpec &spec) : NormalizeBase<CPUBackend>(spec) {}

 private:
  friend class NormalizeBase<CPUBackend>;

  template <typename OutputType, typename InputType>
  void SetupTyped(const HostWorkspace &ws);

  template <typename OutputType, typename InputType>
  void RunTyped(HostWorkspace &ws);

  void AllocTempStorage();
  void FoldMeans();
  void FoldStdDev();

  kernels::KernelManager kmgr_;
};

DALI_REGISTER_OPERATOR(Normalize, Normalize<CPUBackend>, CPU);

using namespace normalize;  // NOLINT

template <typename OutputType, typename InputType>
void Normalize<CPUBackend>::SetupTyped(const HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int nsamples = input.ntensor();
  int nthreads = ws.GetThreadPool().NumThreads();

  using Kernel = kernels::NormalizeCPU<OutputType, InputType, float>;
  kmgr_.Resize<Kernel>(nthreads, nsamples);

  kernels::KernelContext ctx;
  for (int i = 0; i < nsamples; i++) {
    kmgr_.Setup<Kernel>(i, ctx, data_shape_[i], param_shape_[batch_norm_ ? 0 : i]);
  }
  AllocTempStorage();
}

void Normalize<CPUBackend>::AllocTempStorage() {
  const TypeInfo &float_type = TypeTable::GetTypeInfo(DALI_FLOAT);
  int n = data_shape_.num_samples();
  const TensorListShape<> &tmp_shape = batch_norm_
    ? uniform_list_shape(n, param_shape_[0])  // extend to all samples, to enable parallelism
    : param_shape_;

  if (ShouldCalcMean()) {
    mean_.Resize(tmp_shape);
  } else if (has_tensor_mean_) {
    assert(!batch_norm_);
    // use mean as-is
    assert(param_shape_ == mean_input_.shape);
  } else if (has_scalar_mean_) {
    // need to broadcast mean to match required shape
    if (is_uniform(param_shape_)) {
      // if param_shape_ is uniform, we need only one tensor
      mean_.Resize(TensorListShape<>({ param_shape_[0] }));
    } else {
      mean_.Resize(param_shape_);
    }
  }
  if (ShouldCalcStdDev()) {
    inv_stddev_.Resize(tmp_shape);
  } else if (has_tensor_stddev_) {
    assert(!batch_norm_);
    // we need space to calculate inverse stddev
    inv_stddev_.Resize(stddev_input_.shape);
  } else {
    assert(has_scalar_stddev_);
    if (!IsFullReduction()) {
      // need to broadcast stddev to match required shape
      if (is_uniform(param_shape_)) {
        // if param_shape_ is uniform, we need only one tensor
        inv_stddev_.Resize(TensorListShape<>({ param_shape_[0] }));
      } else {
        inv_stddev_.Resize(param_shape_);
      }
    }
  }
  mean_.set_type(float_type);
  inv_stddev_.set_type(float_type);
}


void Normalize<CPUBackend>::FoldMeans() {
  assert(mean_.ntensor() > 0);
  SumSamples(view<float>(mean_));
  // calculate the normalization factor in double, then cast to float
  auto v = ReducedVolume(data_shape_, make_span(axes_));
  if (v == 0) {
    return;
  }
  float den = static_cast<float>(1.0 / v);
  auto sample0 = make_tensor_cpu(mean_.mutable_tensor<float>(0), mean_.shape()[0]);
  auto elems = sample0.num_elements();
  for (int i = 0; i < elems; i++) {
    sample0.data[i] *= den;
  }
}

void Normalize<CPUBackend>::FoldStdDev() {
  assert(inv_stddev_.ntensor() > 0);
  SumSamples(view<float>(inv_stddev_));
  // calculate the normalization factor in double, then cast to float
  auto v = ReducedVolume(data_shape_, make_span(axes_));
  if (v == 0) {
    return;
  }
  float rdiv = 0;
  float scale = scale_;
  if (v > degrees_of_freedom_) {
    rdiv = static_cast<float>(1.0 / (v - degrees_of_freedom_));
  } else {
    if (epsilon_ == 0) {
      rdiv = 1;
      scale = 0;
    }
  }
  auto sample0 = make_tensor_cpu(inv_stddev_.mutable_tensor<float>(0), inv_stddev_.shape()[0]);
  auto elems = sample0.num_elements();
  ScaleRSqrtKeepZero(sample0.data, elems, epsilon_, rdiv, scale);
}

template <typename OutputType, typename InputType>
void Normalize<CPUBackend>::RunTyped(HostWorkspace &ws) {
  ThreadPool &tp = ws.GetThreadPool();

  auto &input = ws.InputRef<CPUBackend>(0);
  TensorListView<StorageCPU, const InputType> in_view = view<const InputType>(input);
  auto in_shape = input.shape();

  auto &output = ws.OutputRef<CPUBackend>(0);
  TensorListView<StorageCPU, OutputType> out_view = view<OutputType>(output);
  output.SetLayout(input.GetLayout());

  int nsamples = input.ntensor();
  int nthreads = ws.GetThreadPool().NumThreads();

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
    float scalar_stddev = spec_.GetArgument<float>("stddev");
    if (epsilon_)  // add epsilon to variance
      scalar_inv_stddev = scale_ * rsqrt(scalar_stddev*scalar_stddev + epsilon_);
    else
      scalar_inv_stddev = scale_ / scalar_stddev;
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
    CalcInvStdDev(mutable_stddev, stddev_input_, epsilon_, scale_);
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
        tp.AddWork([&, i](int thread_idx) {
          kernels::MeanCPU<float, InputType> mean;
          mean.Setup(mutable_mean[i], in_view[i], make_span(axes_));
          // Reset per-sample values, but don't postprocess
          mean.Run(true, false);
        }, in_shape.tensor_size(i));
      }
      tp.RunAll();
      // Aggregate and postprocess now
      FoldMeans();
    }

    if (ShouldCalcStdDev()) {
      auto sample_mean = mean_view[0];
      for (int i = 0; i < nsamples; i++) {
        tp.AddWork([&, i](int thread_idx) {
          kernels::VarianceCPU<float, InputType> stddev;
          stddev.Setup(mutable_stddev[i], in_view[i], make_span(axes_), sample_mean);
          // Reset per-sample values, but don't postprocess
          stddev.Run(true, false);
        }, in_shape.tensor_size(i));
      }
      tp.RunAll();
      // Aggregate and postprocess now - use inverse square root.
      FoldStdDev();
    }
  }

  assert(static_cast<int>(mean_view.data.size()) == mean_view.num_samples());
  assert(static_cast<int>(inv_stddev_view.data.size()) == inv_stddev_view.num_samples());


  for (int i = 0; i < nsamples; i++) {
    tp.AddWork([&, i](int thread_idx) {
      auto sample_mean = mean_view.num_samples() == 1 || batch_norm_
                              ? mean_view[0]
                              : mean_view[i];

      auto sample_inv_stddev = inv_stddev_view.num_samples() == 1 || batch_norm_
                              ? inv_stddev_view[0]
                              : inv_stddev_view[i];

      if (!batch_norm_) {
        if (ShouldCalcMean()) {
          kernels::MeanCPU<float, InputType> mean;
          mean.Setup(mutable_mean[i], in_view[i], make_span(axes_));
          // Reset per-sample values and preprocess
          mean.Run(true, true);
          assert(sample_mean.data == mutable_mean[i].data);
        }

        if (ShouldCalcStdDev()) {
          kernels::VarianceCPU<float, InputType> stddev;
          stddev.Setup(mutable_stddev[i], in_view[i], make_span(axes_), sample_mean);
          // Reset per-sample values, but don't postprocess
          stddev.Run(true, false);
          // Fused postprocessing with inverse square root.
          SumSquare2InvStdDev(mutable_stddev[i], data_shape_[i],
                              degrees_of_freedom_, epsilon_, scale_);
          assert(sample_inv_stddev.data == mutable_stddev[i].data);
        }
      }
      kernels::KernelContext ctx;
      using Kernel = kernels::NormalizeCPU<OutputType, InputType, float>;
      kmgr_.Run<Kernel>(thread_idx, i, ctx,
          out_view[i], in_view[i], sample_mean, sample_inv_stddev, shift_);
    }, in_shape.tensor_size(i));
  }

  tp.RunAll();
}


}  // namespace dali
