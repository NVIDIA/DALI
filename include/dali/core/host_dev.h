// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_HOST_DEV_H_
#define DALI_CORE_HOST_DEV_H_

#if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))
#define DALI_NO_EXEC_CHECK #pragma nv_exec_check_disable
#else
#define DALI_NO_EXEC_CHECK
#endif

#if defined(__CUDACC__)
#define DALI_HOST_DEV __host__ __device__
#else
#define DALI_HOST_DEV
#endif

#if defined(__CUDACC__)
#define DALI_HOST __host__
#else
#define DALI_HOST
#endif

#if defined(__CUDACC__)
#define DALI_DEVICE __device__
#else
#define DALI_DEVICE
#endif


#endif  // DALI_CORE_HOST_DEV_H_
