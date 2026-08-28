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

#include <google/cloud/credentials.h>
#include <google/cloud/options.h>
#include <google/cloud/storage/client.h>
#include <google/cloud/storage/options.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

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
 */
class GCSClientManager {
 public:
  static GCSClientManager& Instance() {
    static GCSClientManager s_manager_;
    return s_manager_;
  }

  /**
   * @brief Returns a client sharing the connection pool with all other clients handed out here.
   */
  google::cloud::storage::Client client() const {
    return client_;
  }

 private:
  static bool EnvFlag(const char* name, bool default_value) {
    auto* value = std::getenv(name);
    if (!value)
      return default_value;
    return std::atoi(value) != 0;
  }

  static google::cloud::Options MakeOptions() {
    namespace gcs = google::cloud::storage;
    google::cloud::Options options;

    // The library also honors CLOUD_STORAGE_EMULATOR_ENDPOINT on its own; this is the DALI-side
    // counterpart of AWS_ENDPOINT_URL.
    if (auto* endpoint_url = std::getenv("DALI_GCS_ENDPOINT_URL")) {
      options.set<gcs::RestEndpointOption>(endpoint_url);
    }

    // By default the client uses Application Default Credentials. Reading a public bucket (or a
    // local emulator) requires opting out of authentication explicitly.
    if (EnvFlag("DALI_GCS_ANONYMOUS", false)) {
      options.set<google::cloud::UnifiedCredentialsOption>(
          google::cloud::MakeInsecureCredentials());
    }

    // DALI only ever issues ranged reads, and GCS reports checksums for whole objects only, so a
    // per-read CRC32C over the payload cannot be validated end-to-end - it would just burn CPU in
    // the data loading path. It can be turned back on for debugging.
    if (!EnvFlag("DALI_GCS_VERIFY_CHECKSUMS", false)) {
      options.set<gcs::DownloadChecksumValidationOption>(  // NOLINT(build/include_what_you_use)
          gcs::ChecksumAlgorithm::kNone);
    }

    return options;
  }

  GCSClientManager() : client_(MakeOptions()) {}

  google::cloud::storage::Client client_;
};

}  // namespace dali

#endif  // DALI_UTIL_GCS_CLIENT_MANAGER_H_
