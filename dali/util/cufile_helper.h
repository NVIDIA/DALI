// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_CUFILE_HELPER_H_
#define DALI_UTIL_CUFILE_HELPER_H_

// general stuff
#include <cstdio>
#include <string>

#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
#include <linux/sysctl.h>
#include <sys/syscall.h>
#endif
#include <unistd.h>

// dali device guard
#include "dali/core/dynlink_cufile.h"
#include "dali/core/device_guard.h"

// we need this class to make sure that the driver
// is only opened once per thread. It is not thread safe, coordination outside
namespace cufile {

class CUFileDriverHandle{
 public:
  explicit CUFileDriverHandle(const int& device = 0) {
    dali::DeviceGuard g(device);
    CUDA_CALL(cuFileDriverOpen());
  }

  ~CUFileDriverHandle() {
    CUDA_CALL(cuFileDriverClose());
  }
};

// wrapper struct to conveniently store the fd's as well
class CUFileHandle{
 public:
  CUFileHandle() {
    fd = -1;
    fdd = -1;
  }

  ~CUFileHandle() {
    Close();
  }

  void Close() {
    if ((fd != -1) && (fdd != -1))
      cuFileHandleDeregister(cufh);
    if (fd != -1) close(fd);
    if (fdd != -1) close(fdd);
    fd = -1;
    fdd = -1;
  }

  CUfileHandle_t cufh;
  int fd;  // descriptor for buffered IO
  int fdd;  // descriptor for direct IO
};

}  // namespace cufile

#endif  // DALI_UTIL_CUFILE_HELPER_H_
