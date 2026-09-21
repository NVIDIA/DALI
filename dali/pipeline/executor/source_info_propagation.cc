// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/source_info_propagation.h"

#include <cassert>
#include <functional>
#include <string>

#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

namespace {

template <typename Backend>
bool SourceInfoDefined(const TensorList<Backend> &tl) {
  for (int s = 0; s < tl.num_samples(); s++)
    if (!tl.GetMeta(s).GetSourceInfo().empty())
      return true;
  return false;
}

inline bool OutputSourceInfoDefined(Workspace &ws) {
  for (int o = 0; o < ws.NumOutput(); o++) {
    if (ws.WithOutput(o, [](auto &output) { return SourceInfoDefined(output); }))
      return true;
  }
  return false;
}

template <typename Backend>
void ClearSourceInfo(TensorList<Backend> &tl) {
  for (int s = 0; s < tl.num_samples(); s++)
    tl.SetSourceInfo(s, "");
}

}  // namespace

void ClearOutputSourceInfo(Workspace &ws) {
  for (int o = 0; o < ws.NumOutput(); o++) {
    ws.WithOutput(o, [](auto &output) { ClearSourceInfo(output); });
  }
}

bool PropagateSourceInfo(Workspace &ws) {
  int num_inputs = ws.NumInput();
  int num_outputs = ws.NumOutput();

  if (num_inputs == 0 || num_outputs == 0)
    return false;  // there's nothing to propagate

  if (OutputSourceInfoDefined(ws))
    return false;  // the operator defined the source info, no need to set it

  SmallVector<std::function<const std::string &(int)>, 8> get_src_info;
  get_src_info.resize(num_inputs);
  int batch_size = 0;
  for (int i = 0; i < num_inputs; i++) {
    auto process_input = [&](auto &input) {
      // check the batch size
      if (i == 0)
        batch_size = input.num_samples();
      else if (input.num_samples() != batch_size)
        return false;  // mismatched input batch size - this is a special operator, bailing out

      get_src_info[i] = [&input](int sample)->const std::string & {
        return input.GetMeta(sample).GetSourceInfo();
      };
      return true;
    };

    if (!ws.WithInput(i, process_input))
      return false;
  }

  for (int o = 0; o < num_outputs; o++) {
    if (ws.GetOutputBatchSize(o) != batch_size)
      return false;  // this operator changes the batch size - bailing out
  }

  BatchVector<const std::string*> source_infos;
  source_infos.resize(batch_size);
  for (int s = 0; s < batch_size; s++) {
    const std::string *sinfo = nullptr;
    for (int i = 0; i < num_inputs; i++) {
      auto &si = get_src_info[i](s);
      if (si.empty())
        continue;
      if (sinfo && *sinfo != si)
        return false;  // inconsistent source info - bailing out
      sinfo = &si;
    }
    source_infos[s] = sinfo;
  }

  auto set_source_infos = [&](auto &out) {
    assert(out.num_samples() == batch_size);
    for (int s = 0; s < batch_size; s++) {
      if (auto *si = source_infos[s])
        out.SetSourceInfo(s, *si);
    }
  };

  for (int o = 0; o < num_outputs; o++) {
    ws.WithOutput(o, set_source_infos);
  }
  return true;
}

}  // namespace dali
