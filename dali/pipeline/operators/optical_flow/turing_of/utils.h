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

#ifndef DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_UTILS_H_
#define DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_UTILS_H_

#define TURING_OF_API_CALL(nvOFAPI)                                                  \
    do                                                                               \
    {                                                                                \
        NV_OF_STATUS errorCode = nvOFAPI;                                            \
        if (errorCode != NV_OF_SUCCESS) {                                            \
            std::ostringstream errorLog;                                             \
            errorLog << #nvOFAPI << " returned error: " << errorCode << std::endl;   \
            DALI_FAIL(errorLog.str());                                               \
        }                                                                            \
    } while (0)

#endif  // DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_UTILS_H_
