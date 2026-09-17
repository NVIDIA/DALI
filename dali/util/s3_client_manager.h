// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_S3_CLIENT_MANAGER_H_
#define DALI_UTIL_S3_CLIENT_MANAGER_H_

#include <aws/s3/S3Client.h>
#include <memory>
#include "dali/core/api_helper.h"

namespace dali {

/**
 * @brief Owns the process-wide AWS S3 client, and the InitAPI/ShutdownAPI pair around it.
 *
 * Defined out-of-line in s3_client_manager.cc, compiled only into libdali.so, and exported
 * (DLL_PUBLIC) so that libdali_operators.so - which links against libdali.so - resolves
 * `Instance()` to that one copy instead of getting a private one of its own: a header-only
 * singleton would take its own `getenv()` snapshot of AWS_ENDPOINT_URL/DALI_S3_NO_VERIFY_SSL
 * separately in each shared library, and, more importantly, would run `Aws::InitAPI`/
 * `Aws::ShutdownAPI` twice - the SDK does not support that.
 */
struct DLL_PUBLIC S3ClientManager {
 public:
  static S3ClientManager& Instance();

  Aws::S3::S3Client* client() {
    return client_.get();
  }

 private:
  S3ClientManager();
  ~S3ClientManager();

  std::unique_ptr<Aws::S3::S3Client> client_;
};

}  // namespace dali

#endif  // DALI_UTIL_S3_CLIENT_MANAGER_H_
