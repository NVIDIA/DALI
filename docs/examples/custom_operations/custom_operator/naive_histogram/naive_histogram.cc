// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "naive_histogram.h"

namespace naive_histogram {

using namespace ::dali;

DALI_SCHEMA(NaiveHistogram)
                .DocStr("Creates a histogram.")
                .AddOptionalArg("n_bins", "Number of bins in the histogram", 24)
                .NumInput(1)
                .NumOutput(1);

DALI_REGISTER_OPERATOR(NaiveHistogram, NaiveHistogram<GPUBackend>, GPU);

}  // namespace naive_histogram
