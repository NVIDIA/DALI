// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_
#define DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_

#include <cassert>
#include <memory>
#include <vector>
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/core/static_switch.h"


// #define TENSOR_RESIZE_SUPPORTED_NDIM (2, 3)
// #define TENSOR_RESIZE_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
//                                        uint64_t, int64_t, float)

#define TENSOR_RESIZE_SUPPORTED_NDIM (2, 3)
#define TENSOR_RESIZE_SUPPORTED_TYPES (uint8_t, float)


namespace dali {
namespace tensor_resize {

template <typename Backend>
class TensorResizeImplBase {
 public:
  virtual ~TensorResizeImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const Workspace &ws,
                         const TensorListShape<> &sizes) = 0;
  virtual void RunImpl(Workspace &ws) = 0;
};

template <typename Backend, template <typename Out, typename In, int static_ndim> class Impl>
class TensorResize : public Operator<Backend> {
 public:
  explicit TensorResize(const OpSpec &spec) : Operator<Backend>(spec) {
    if (spec.HasArgument("dtype")) {
      dtype_arg_ = spec.GetArgument<DALIDataType>("dtype");
    }
  }

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    auto in_shape = input.shape();
    int ndim = input.sample_dim();
    int nsamples = in_shape.num_samples();

    int nargs = ndim;  // or naxes
    sizes_.Acquire(spec_, ws, nsamples, TensorShape<1>{nargs});
    TensorListShape<> sizes = in_shape;
    for (int s = 0; s < nsamples; s++) {
      auto sample_sizes = sizes.tensor_shape_span(s);
      const auto *sample_sizes_arg = sizes_[s].data;
      for (int d = 0; d < ndim; d++) {
        sample_sizes[d] = sample_sizes_arg[d];
      }
    }

    int spatial_ndim = NeedExtraChDim(ws) ? ndim : ndim - 1;
    if (impl_ == nullptr || spatial_ndim_ != spatial_ndim ||
        input.type() != in_dtype_ || output.type() != out_dtype_) {
      TYPE_SWITCH(input.type(), type2id, T, TENSOR_RESIZE_SUPPORTED_TYPES, (
      if (dtype_arg_ == DALI_NO_TYPE) {
        VALUE_SWITCH(spatial_ndim, SpatialDims, TENSOR_RESIZE_SUPPORTED_NDIM, (
          impl_ = std::make_unique<Impl<T, T, SpatialDims>>(&spec_);
        ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
      } else {
        TYPE_SWITCH(dtype_arg_, type2id, Out, TENSOR_RESIZE_SUPPORTED_TYPES, (
          VALUE_SWITCH(spatial_ndim, SpatialDims, TENSOR_RESIZE_SUPPORTED_NDIM, (
            impl_ = std::make_unique<Impl<Out, T, SpatialDims>>(&spec_);
          ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
        ), DALI_FAIL(make_string("Unsupported data type: ", dtype_arg_)));  // NOLINT
      }
    ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
      spatial_ndim_ = spatial_ndim;
      in_dtype_ = input.type();
      out_dtype_ = output.type();
    }

    assert(impl_ != nullptr);
    return impl_->SetupImpl(output_desc, ws, sizes);
  }

  void RunImpl(Workspace &ws) override {
    assert(impl_ != nullptr);
    impl_->RunImpl(ws);
  }

  bool NeedExtraChDim(const Workspace &ws) {
    const auto &input = ws.Input<CPUBackend>(0);
    auto in_shape = input.shape();
    int ndim = input.sample_dim();
    int nsamples = input.num_samples();
    int nsizes = sizes_[0].num_elements();
    auto get_uniform_last_size = [&]() {
      int last_size = -1;
      bool last_size_uniform = true;
      for (int s = 0; last_size_uniform && s < nsamples; s++) {
        if (last_size == -1)
          last_size = sizes_[s].data[nsizes - 1];
        else if (last_size != sizes_[s].data[nsizes - 1])
          last_size_uniform = false;
      }
      return last_size_uniform ? last_size : -1;
    };

    // If last dim is uniform, we can interpret it as channels
    auto get_uniform_last_extent = [&]() {
      int last_dim = ndim - 1;
      int nsamples = in_shape.num_samples();
      if (nsamples == 0)
        return 0;
      int nchannels = in_shape.tensor_shape_span(0)[last_dim];
      for (int s = 1; s < nsamples; s++) {
        if (nchannels != in_shape.tensor_shape_span(s)[last_dim])
          return -1;  // last dim is not uniform, can't be interpreted as nchannels
      }
      return nchannels;
    };

    int last_extent = get_uniform_last_extent();
    int last_size = get_uniform_last_size();
    return last_extent != last_size || last_extent == -1;
  }

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  std::unique_ptr<TensorResizeImplBase<Backend>> impl_;

  int spatial_ndim_ = -1;
  DALIDataType in_dtype_ = DALI_NO_TYPE;
  DALIDataType out_dtype_ = DALI_NO_TYPE;
  DALIDataType dtype_arg_ = DALI_NO_TYPE;

  ArgValue<int, 1> sizes_{"sizes", spec_};
  ArgValue<float, 1> scales_{"scales", spec_};
};

}  // namespace tensor_resize
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_H_
