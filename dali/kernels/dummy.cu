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

// Empty file for Clang-only compilation that is still using NVCC and thus
// forcing CMake to link using NVCC, to alow for device linking (required for separable
// compilation). Older CMakes use Clang for linking even when the CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS
// is ON.
