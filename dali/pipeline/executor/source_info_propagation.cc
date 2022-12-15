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

#include "dali/pipeline/executor/source_info_propagation.h"

#include <cassert>
#include <functional>
#include <string>

#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

template <typename Backend>
bool SourceInfoDefined(const TensorList<Backend> &tl) {
  for (int s = 0; s < tl.num_samples(); s++)
    if (!tl.GetMeta(s).GetSourceInfo().empty())
      return true;
  return false;
}

inline bool OutputSourceInfoDefined(Workspace &ws) {
  for (int o = 0; o < ws.NumOutput(); o++) {
    if (ws.OutputIsType<GPUBackend>(o)) {
      if (SourceInfoDefined(ws.Output<GPUBackend>(o)))
        return true;
    } else {
      assert(ws.OutputIsType<CPUBackend>(o));
      if (SourceInfoDefined(ws.Output<CPUBackend>(o)))
        return true;
    }
  }
  return false;
}

inline int GetOutputBatchSize(Workspace &ws, int output_idx) {
  if (ws.OutputIsType<CPUBackend>(output_idx)) {
    return ws.Output<CPUBackend>(output_idx).num_samples();
  } else {
    assert(ws.OutputIsType<GPUBackend>(output_idx));
    return ws.Output<GPUBackend>(output_idx).num_samples();
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

      get_src_info[i] = [&input, i](int sample)->const std::string & {
        return input.GetMeta(sample).GetSourceInfo();
      };
      return true;
    };

    if (ws.InputIsType<CPUBackend>(i)) {
      if (!process_input(ws.Input<CPUBackend>(i)))
        return false;
    } else {
      assert(ws.InputIsType<GPUBackend>(i));
      if (!process_input(ws.Input<GPUBackend>(i)))
        return false;
    }
  }

  for (int o = 0; o < num_outputs; o++) {
    if (GetOutputBatchSize(ws, o) != batch_size)
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
    if (ws.OutputIsType<CPUBackend>(o)) {
      set_source_infos(ws.Output<CPUBackend>(o));
    } else {
      assert(ws.OutputIsType<GPUBackend>(o));
      set_source_infos(ws.Output<GPUBackend>(o));
    }
  }
  return true;
}

}  // namespace dali
