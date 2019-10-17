// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_FORCE_INLINE_H_
#define DALI_CORE_FORCE_INLINE_H_

#ifndef DALI_FORCEINLINE
#if defined(_MSC_VER)
#define DALI_FORCEINLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ >= 4
#define DALI_FORCEINLINE __attribute__((always_inline)) inline
#else
#define DALI_FORCEINLINE inline
#endif
#endif

#endif  // DALI_CORE_FORCE_INLINE_H_
