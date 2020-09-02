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


#include "dali/operators.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/fuzzing/dali_harness.h"


int main(int argc, char *argv[]) {
  // Parse and validate command line arg
  // This is assumed to be run through the fuzzer, so we don't check arguments validity
  std::string path(argv[1]);

  // Init DALI
  dali::InitOperatorsLib();
  dali::DALIInit(
    dali::OpSpec("CPUAllocator"),
    dali::OpSpec("PinnedCPUAllocator"),
    dali::OpSpec("GPUAllocator"));

  // Run test
  dali::DecoderHarness harness{path};
  harness.Run();

  return 0;
}
