// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_GCS_CLIENT_MANAGER_H_
#define DALI_UTIL_GCS_CLIENT_MANAGER_H_

#include <google/cloud/storage/client.h>
#include "dali/core/api_helper.h"

namespace dali {

/**
 * @brief Owns the process-wide configuration of the GCS client.
 *
 * Unlike `Aws::S3::S3Client`, `google::cloud::storage::Client` is not documented as safe for
 * concurrent use of a *single* instance ("Two threads operating on the same instance of this
 * class is not guaranteed to work"). Copies, on the other hand, share the underlying connection
 * pool and are explicitly safe to use from different threads, and copying is about as expensive
 * as copying a few shared pointers. Therefore `client()` hands out a copy and each caller
 * (file stream, file discovery) keeps its own.
 *
 * There is no global init/shutdown to perform - the library initializes libcurl lazily.
 *
 * Defined out-of-line in gcs_client_manager.cc, compiled only into libdali.so, and exported
 * (DLL_PUBLIC) so that libdali_operators.so - which links against libdali.so - resolves
 * `Instance()` to that one copy instead of getting a private one of its own: a header-only
 * singleton would take its own `getenv()` snapshot of DALI_GCS_* separately in each shared
 * library, and the two could disagree if the environment changed between the two libraries'
 * first `gs://` access.
 */
class DLL_PUBLIC GCSClientManager {
 public:
  static GCSClientManager& Instance();

  /**
   * @brief Returns a client sharing the connection pool with all other clients handed out here.
   */
  google::cloud::storage::Client client() const {
    return client_;
  }

 private:
  GCSClientManager();

  google::cloud::storage::Client client_;
};

}  // namespace dali

#endif  // DALI_UTIL_GCS_CLIENT_MANAGER_H_
