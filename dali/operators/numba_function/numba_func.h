// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_
#define DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

// The structure of an array descriptor can be viewed here:
// https://github.com/numba/numba/blob/b1be2f12c83c01f57fe34fab9a9d77334f9baa1d/numba/cuda/dispatcher.py#L325
// https://github.com/numba/numba/blob/3b9dde799bc499188f9d7728ad590776899624e1/numba/_arraystruct.h#L9C1
struct NumbaDevArray {
  void *meminfo;
  void *parent;
  int64_t nitems;
  int64_t itemsize;
  void *data;
  span<const int64_t> shape;
  span<int64_t> strides;

  NumbaDevArray(span<const int64_t> shape, span<int64_t> strides,
                DALIDataType type):
    meminfo(nullptr),
    parent(nullptr),
    data(nullptr),
    shape(shape),
    strides(strides) {
    nitems = volume(shape);
    itemsize = TypeTable::GetTypeInfo(type).size();
  }

  /// @brief Push array descriptor to a vector of func args as void pointers.
  void PushArgs(vector<void*> &args) {
    args.push_back(&meminfo);
    args.push_back(&parent);
    args.push_back(&nitems);
    args.push_back(&itemsize);
    args.push_back(&data);
    for (int64_t i = 0; i < shape.size(); ++i) {
      const void *ptr = &shape[i];
      args.push_back(const_cast<void*>(ptr));
    }
    for (int64_t i = 0; i < strides.size(); ++i) {
      args.push_back((&strides[i]));
    }
  }
};

template <typename Backend>
class NumbaFuncImpl : public Operator<Backend> {
 public:
  using Base = Operator<Backend>;

  explicit NumbaFuncImpl(const OpSpec &spec_);

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  /**
   * @brief Setup output descriptors calling the setup_fn to determine the output shapes
   */
  void OutputsSetupFn(std::vector<OutputDesc> &output_desc, int noutputs,
                      int ninputs, int nsamples);

  /**
   * @brief Setup output descriptors copying shapes from inputs
   */
  void OutputsSetupNoFn(std::vector<OutputDesc> &output_desc, int noutputs,
                        int ninputs, int nsamples);

  void RunImpl(Workspace &ws) override;

 private:
  using NumbaPtr = uint64_t;

  vector<vector<ssize_t>> in_sizes_;
  vector<vector<ssize_t>> out_sizes_;
  vector<vector<void*>> in_memory_ptrs_;
  vector<vector<void*>> out_memory_ptrs_;
  SmallVector<int, 3> blocks_;
  SmallVector<int, 3> threads_per_block_;
  NumbaPtr run_fn_;
  NumbaPtr setup_fn_;
  bool batch_processing_;
  SmallVector<DALIDataType, 6> out_types_;
  SmallVector<DALIDataType, 6> in_types_;
  SmallVector<int, 6> outs_ndim_;
  SmallVector<int, 6> ins_ndim_;
  std::vector<uintptr_t> output_shape_ptrs_;
  std::vector<uintptr_t> input_shape_ptrs_;
  vector<TensorListShape<-1>> in_shapes_;
  vector<TensorListShape<-1>> out_shapes_;
  vector<TensorShape<-1>> in_strides_;
  vector<TensorShape<-1>> out_strides_;
  vector<NumbaDevArray> in_arrays_;
  vector<NumbaDevArray> out_arrays_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_NUMBA_FUNCTION_NUMBA_FUNC_H_
