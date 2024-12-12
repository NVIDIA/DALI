// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/graph/cse.h"
#include <functional>
#include <map>
#include <string>

namespace dali {
namespace graph {

struct OpSpecCompare {
  int operator()(const OpSpec &a, const OpSpec &b) const {
    auto na = std::make_tuple(a.Arguments().size(), a.NumInput(), a.NumOutput());
    auto nb = std::make_tuple(b.Arguments().size(), b.NumInput(), b.NumOutput());
    if (na < nb)
      return -1;
    if (na > nb)
      return 1;
    for (int i = 0; i < a.NumInput(); i++) {
      auto ia = a.Input(i);
      auto ib = a.Input(i);
      int cmp = ia.compare(ib);
      if (cmp)
        return cmp;
    }
    auto names_a = a.ListArgumentNames();
    auto names_b = b.ListArgumentNames();
    int cmp = compare(names_a, names_b);
    if (cmp)
      return cmp;

    for (auto name_v : names_a) {
      string name(name_v);  // TODO(michalz): use string_view in OpSpec
      int i = a.GetArgumentIdx(name).value();
      int j = b.GetArgumentIdx(name).value();
      int cmp = compare(*a.Arguments()[i], *b.Arguments()[j]);
      if (cmp)
        return cmp;
    }
    // we deliberately don't compare output names here!
    return 0;
  }
};

struct OpSpecLess {
  bool operator()(const OpSpec &a, const OpSpec &b) const {
    return OpSpecCompare()(a, b) < 0;
  }
};


class CSE {
 public:
  explicit CSE(OpGraph &graph) : graph_(graph) {}

  void Run() {
  }

  void Run(OpNode *node) {
  }

  std::map<OpSpec, OpNode *, OpSpecCompare> normalized_nodes_;
  std::map<string, string, std::less<>> renamed_;
  OpGraph &graph_;
};

void EliminateCommonSubgraphs(OpGraph &graph) {
}

}  // namespace graph
}  // namespace dali
