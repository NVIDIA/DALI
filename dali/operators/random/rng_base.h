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

#ifndef DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
#define DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_

#include <random>
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

DALI_SCHEMA(RNGAttr)
    .DocStr(R"code()code")
    .AddOptionalArg<std::vector<int>>("shape",
      R"code(Shape of the data.)code", nullptr, true)
    .AddOptionalArg<DALIDataType>("dtype",
      R"code(Data type.)code", nullptr);

template<typename Backend, typename Impl>
class RNGBase : public Operator<Backend> {
 public:
  ~RNGBase() override = default;

 protected:
  explicit RNGBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        rng_(spec.GetArgument<int64_t>("seed"), spec.GetArgument<DALIDataType>("batch_size")) {
  }

  Impl &This() noexcept { return static_cast<Impl&>(*this); }
  const Impl &This() const noexcept { return static_cast<const Impl&>(*this); }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    if (!spec_.TryGetArgument(dtype_, "dtype"))
      dtype_ = This().DefaultDataType();

    bool has_shape = spec_.ArgumentDefined("shape");
    bool has_shape_like = spec_.NumRegularInput() == 1;
    DALI_ENFORCE(!(has_shape && has_shape_like),
      "Providing argument \"shape\" is incompatible with providing a shape-like input");

    if (has_shape_like) {
      shape_ = ws.template InputRef<Backend>(0).shape();
    } else if (has_shape) {
      GetShapeArgument(shape_, spec_, "shape", ws);
    } else {
      shape_ = uniform_list_shape(spec_.template GetArgument<DALIDataType>("batch_size"), {1});
    }
    batch_size_ = shape_.size();

    This().AcquireArgs(spec_, ws, batch_size_);

    output_desc.resize(1);
    output_desc[0].shape = shape_;
    output_desc[0].type = TypeTable::GetTypeInfo(dtype_);
    return true;
  }

  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws) {
    auto &output = ws.OutputRef<CPUBackend>(0);
    auto out_view = view<T>(output);
    const auto &out_shape = out_view.shape;
    auto &tp = ws.GetThreadPool();
    for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
      auto sample_sz = out_shape.tensor_size(sample_id);
      tp.AddWork(
          [&, sample_id, sample_sz](int thread_id) {
            span<T> out_span{out_view[sample_id].data, sample_sz};
            This().template Generate<T>(out_span, sample_id, rng_[sample_id]);
          }, sample_sz);
    }
    tp.RunAll();
  }

  using Operator<Backend>::spec_;
  using Operator<Backend>::batch_size_;
  DALIDataType dtype_;
  BatchRNG<std::mt19937_64> rng_;
  TensorListShape<> shape_;
};




}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
