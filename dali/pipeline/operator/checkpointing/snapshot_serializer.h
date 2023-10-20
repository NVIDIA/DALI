// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_CHECKPOINTING_SNAPSHOT_SERIALIZER_H_
#define DALI_PIPELINE_OPERATOR_CHECKPOINTING_SNAPSHOT_SERIALIZER_H_

#include <random>
#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"

namespace dali {

/**
 * @brief An object that provides implementation of serialization/deserialization
 *        for checkpointing.
*/
class DLL_PUBLIC SnapshotSerializer {
 public:
  DLL_PUBLIC std::string Serialize(const std::vector<std::mt19937> &snapshot);

  DLL_PUBLIC std::string Serialize(const std::vector<std::mt19937_64> &snapshot);

  DLL_PUBLIC std::string Serialize(const LoaderStateSnapshot &snapshot);

  /**
   * @brief Deserializes string into an object.
   *
   * The template should be specialized for a type iff it is serialized by this object.
  */
  template<typename T> T Deserialize(const std::string &data);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_SNAPSHOT_SERIALIZER_H_
