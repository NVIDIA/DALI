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

#include "dali/util/gcs_client_manager.h"

#include <google/cloud/credentials.h>
#include <google/cloud/options.h>
#include <google/cloud/storage/options.h>
#include <google/cloud/storage/retry_policy.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include "dali/core/error_handling.h"

namespace dali {

namespace {

bool EnvFlag(const char* name, bool default_value) {
  auto* value = std::getenv(name);
  if (!value)
    return default_value;
  return std::atoi(value) != 0;
}

/**
 * @brief Reads a non-negative integer from the environment.
 *
 * Unlike EnvFlag, a malformed value is reported and ignored rather than silently taken as 0 -
 * the variable read this way bounds how long a stalled request may take, and a typo turning
 * into 0 would remove the bound instead of shortening it.
 */
int EnvInt(const char* name, int default_value) {
  auto* value = std::getenv(name);
  if (!value || *value == '\0')
    return default_value;
  char* end = nullptr;
  errno = 0;
  std::int64_t parsed = std::strtoll(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    DALI_WARN(name, " is set to \"", value, "\", which is not a non-negative integer; using ",
              default_value, " instead.");
    return default_value;
  }
  return static_cast<int>(parsed);
}

google::cloud::Options MakeOptions() {
  namespace gcs = google::cloud::storage;
  google::cloud::Options options;

  // The library also honors CLOUD_STORAGE_EMULATOR_ENDPOINT on its own; DALI_GCS_ENDPOINT_URL
  // additionally lets a caller override the endpoint without touching the environment the
  // library itself reads, e.g. to point at a local/mock GCS server for testing.
  if (auto* endpoint_url = std::getenv("DALI_GCS_ENDPOINT_URL")) {
    options.set<gcs::RestEndpointOption>(endpoint_url);
  }

  // By default the client uses Application Default Credentials. Reading a public bucket (or a
  // local emulator) requires opting out of authentication explicitly.
  if (EnvFlag("DALI_GCS_ANONYMOUS", false)) {
    options.set<google::cloud::UnifiedCredentialsOption>(google::cloud::MakeInsecureCredentials());
  }

  // DALI only ever issues ranged reads, and GCS reports checksums for whole objects only, so a
  // per-read CRC32C over the payload cannot be validated end-to-end - it would just burn CPU in
  // the data loading path. It can be turned back on for debugging.
  if (!EnvFlag("DALI_GCS_VERIFY_CHECKSUMS", false)) {
    options.set<gcs::DownloadChecksumValidationOption>(  // NOLINT(build/include_what_you_use)
        gcs::ChecksumAlgorithm::kNone);
  }

  // google-cloud-cpp bounds neither the retry loop nor a stalled transfer tightly enough for a
  // data loading library: the defaults are a 120 s stall timeout and a 15 minute retry window,
  // and a refused or timed out connection is retryable (kUnavailable/kDeadlineExceeded), so an
  // endpoint that is dead or misrouted makes Pipeline.build() hang for a quarter of an hour
  // with no output. Bound a single stalled transfer and the whole retry loop by the same
  // budget, which caps a failing request at roughly twice that budget while still retrying the
  // transient 5xx/429 that GCS asks clients to retry - with the library's default backoff
  // (1 s, doubling) 60 s leaves room for about six attempts. The stall timeout doubles as the
  // connect timeout: the library falls back to it for CURLOPT_CONNECTTIMEOUT_MS when no
  // explicit connect timeout is configured.
  int timeout_s = EnvInt("DALI_GCS_REQUEST_TIMEOUT_SEC", 60);
  if (timeout_s > 0) {
    std::chrono::seconds timeout(timeout_s);
    options.set<gcs::TransferStallTimeoutOption>(timeout)  // NOLINT(build/include_what_you_use)
        .set<gcs::DownloadStallTimeoutOption>(timeout)  // NOLINT(build/include_what_you_use)
        .set<gcs::RetryPolicyOption>(  // NOLINT(build/include_what_you_use)
            gcs::LimitedTimeRetryPolicy(timeout).clone());
  }

  return options;
}

}  // namespace

GCSClientManager& GCSClientManager::Instance() {
  static GCSClientManager s_manager_;
  return s_manager_;
}

GCSClientManager::GCSClientManager() : client_(MakeOptions()) {}

}  // namespace dali
