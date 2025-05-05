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

#ifndef DALI_UTIL_S3_CLIENT_MANAGER_H_
#define DALI_UTIL_S3_CLIENT_MANAGER_H_

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "dali/core/common.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

struct S3ClientManager {
 public:
  static S3ClientManager& Instance() {
    static int init = [&]() {
      RunInitOrShutdown([&](int) { Aws::InitAPI(Aws::SDKOptions{}); });
      return 0;
    }();
    // We want RunInitOrShutdown s_thread_pool_ to outlive s_manager_
    static S3ClientManager s_manager_;
    return s_manager_;
  }

  Aws::S3::S3Client* client() {
    return client_.get();
  }

 private:
  // Documentation says:
  // Please call this from the same thread from which InitAPI() has been called (use a dedicated
  // thread if necessary). This avoids problems in initializing the dependent Common RunTime C
  // libraries.
  static void RunInitOrShutdown(std::function<void(int)> work) {
    static ThreadPool s_thread_pool_(1, CPU_ONLY_DEVICE_ID, false, "S3ClientManager");
    s_thread_pool_.AddWork(std::move(work));
    s_thread_pool_.RunAll();
  }

  S3ClientManager() {
    Aws::S3::S3ClientConfiguration config;
    auto endpoint_url_ptr = std::getenv("AWS_ENDPOINT_URL");
    if (endpoint_url_ptr) {
      config.endpointOverride = std::string(endpoint_url_ptr);
    }
    auto no_verify_ptr = std::getenv("DALI_S3_NO_VERIFY_SSL");
    if (no_verify_ptr) {
      config.verifySSL = std::atoi(no_verify_ptr) == 0;
    }
    client_ = std::make_unique<Aws::S3::S3Client>(std::move(config));
  }

  ~S3ClientManager() {
    client_.reset();
    RunInitOrShutdown([&](int) { Aws::ShutdownAPI(Aws::SDKOptions{}); });
  }

  std::unique_ptr<Aws::S3::S3Client> client_;
};

}  // namespace dali

#endif  // DALI_UTIL_S3_CLIENT_MANAGER_H_
