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

#ifndef DALI_OPERATORS_READER_LOADER_UTILS_H_
#define DALI_OPERATORS_READER_LOADER_UTILS_H_

#include <vector>
#include <string>
#include "dali/core/api_helper.h"

namespace dali {

/*
 * When adding new vector here,
 * make sure to add it also in `bool HasKnownExtension(const std::string &filepath);`
 */
static const std::vector<std::string> kKnownImageExtensions = {".jpg", ".jpeg", ".png", ".bmp",
                                                               ".tif", ".tiff", ".pnm", ".ppm",
                                                               ".pgm", ".pbm", ".jp2", ".webp"};

static const std::vector<std::string> kKnownAudioExtensions = {".flac", ".ogg", ".wav"};

/**
 * Checks, if the name of the file provided in the argument ends with image extension, that is known.
 */
DLL_PUBLIC bool HasExtension(std::string filepath, const std::vector<std::string> &extensions);

/**
 * Convenient overload to verify against all known extensions
 */
DLL_PUBLIC bool HasKnownExtension(const std::string &filepath);

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_UTILS_H_
