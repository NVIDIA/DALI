// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_FACTORY_H_
#define DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/pipeline/operators/arithmetic/expression_tree.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
class ExprImplParam {
 protected:
  // We differentiate between TensorVector and TensorList, that still have inconsistent
  // access patterns
  static constexpr bool is_cpu = std::is_same<Backend, CPUBackend>::value;

  template <bool IsTensor, typename Type>
  std::enable_if_t<IsTensor && is_cpu, const Type *> ObtainInput(const ExprFunc &expr,
                                                                 workspace_t<Backend> &ws,
                                                                 const OpSpec &spec,
                                                                 TileDesc tile,
                                                                 int subexpr_id) {
    int input_id = dynamic_cast<const ExprTensor&>(expr[subexpr_id]).GetInputIndex();
    auto *tensor = ws.template InputRef<Backend>(input_id)[tile.sample_idx].template data<Type>();
    return tensor + tile.extent_idx * tile.tile_size;
  }

  template <bool IsTensor, typename Type>
  std::enable_if_t<IsTensor && !is_cpu, const Type *> ObtainInput(const ExprFunc &expr,
                                                                  workspace_t<Backend> &ws,
                                                                  const OpSpec &spec,
                                                                  TileDesc tile,
                                                                  int subexpr_id) {
    int input_id = dynamic_cast<const ExprTensor&>(expr[subexpr_id]).GetInputIndex();
    auto *tensor = ws.template InputRef<Backend>(input_id).template tensor<Type>(tile.sample_idx);
    return tensor + tile.extent_idx * tile.tile_size;
  }

  template <bool IsTensor, typename Type>
  std::enable_if_t<!IsTensor, Type> ObtainInput(const ExprFunc &expr,
                                                workspace_t<Backend> &ws, const OpSpec &spec,
                                                TileDesc tile, int subexpr_id) {
    int scalar_id = dynamic_cast<const ExprConstant&>(expr[subexpr_id]).GetConstIndex();
    if (IsIntegral(expr.GetTypeId())) {
      return static_cast<Type>(spec.GetArgument<std::vector<int>>("integer_scalars")[scalar_id]);
    }
    return static_cast<Type>(spec.GetArgument<std::vector<float>>("float_scalars")[scalar_id]);
  }

  template <typename Result>
  std::enable_if_t<is_cpu, Result *> ObtainOutput(const ExprFunc &expr,
                                                  workspace_t<Backend> &ws, const OpSpec &spec,
                                                  TileDesc tile) {
    auto *tensor =
        ws.template OutputRef<Backend>(0)[tile.sample_idx].template mutable_data<Result>();
    return tensor + tile.extent_idx * tile.tile_size;
  }

  template <typename Result>
  std::enable_if_t<!is_cpu, Result *> ObtainOutput(const ExprFunc &expr,
                                                   workspace_t<GPUBackend> &ws, const OpSpec &spec,
                                                   TileDesc tile) {
    auto *tensor =
        ws.template OutputRef<Backend>(0).template mutable_tensor<Result>(tile.sample_idx);
    return tensor + tile.extent_idx * tile.tile_size;
  }
};

std::unique_ptr<ExprImplBase> ExprImplFactory(const HostWorkspace &ws,
                                              const ExprNode &expr);

std::unique_ptr<ExprImplBase> ExprImplFactory(const DeviceWorkspace &ws,
                                              const ExprNode &expr);

struct ExprImplCache {
  template <typename Backend>
  ExprImplBase *GetExprImpl(const ExprNode &expr) {
    auto node_desc = expr.GetNodeDesc();
    auto it = cache_.find(node_desc);
    if (it != cache_.end()) {
      return it->second.get();
    }
    auto new_impl = ExprImplFactory(workspace_t<Backend>{}, expr);
    auto ptr = std::shared_ptr<ExprImplBase>(std::move(new_impl));
    cache_[node_desc] = ptr;
    return ptr.get();
  }

 private:
  std::map<std::string, std::shared_ptr<ExprImplBase>> cache_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_ARITHMETIC_EXPRESSION_IMPL_FACTORY_H_
