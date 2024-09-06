/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format off
#ifndef NVIMGCODEC_VERSION_H__
#define NVIMGCODEC_VERSION_H__

#define NVIMGCODEC_VER_MAJOR 0
#define NVIMGCODEC_VER_MINOR 4
#define NVIMGCODEC_VER_PATCH 0
#define NVIMGCODEC_VER_BUILD 

#define MAKE_SEMANTIC_VERSION(major, minor, patch) ((major * 1000) + (minor * 100) + patch)
#define NVIMGCODEC_MAJOR_FROM_SEMVER(ver) (ver / 1000)
#define NVIMGCODEC_MINOR_FROM_SEMVER(ver) ((ver % 1000) / 100)
#define NVIMGCODEC_PATCH_FROM_SEMVER(ver) ((ver % 1000) % 100)
#define NVIMGCODEC_STREAM_VER(ver) \
    NVIMGCODEC_MAJOR_FROM_SEMVER(ver) << "." << NVIMGCODEC_MINOR_FROM_SEMVER(ver) << "." << NVIMGCODEC_PATCH_FROM_SEMVER(ver)
    
#define NVIMGCODEC_VER MAKE_SEMANTIC_VERSION(NVIMGCODEC_VER_MAJOR, NVIMGCODEC_VER_MINOR, NVIMGCODEC_VER_PATCH)

#define NVIMGCODEC_EXT_API_VER_MAJOR 0
#define NVIMGCODEC_EXT_API_VER_MINOR 2
#define NVIMGCODEC_EXT_API_VER_PATCH 0

#define NVIMGCODEC_EXT_API_VER MAKE_SEMANTIC_VERSION(NVIMGCODEC_EXT_API_VER_MAJOR, NVIMGCODEC_EXT_API_VER_MINOR, NVIMGCODEC_EXT_API_VER_PATCH)

#endif // NVIMGCODEC_VERSION 
