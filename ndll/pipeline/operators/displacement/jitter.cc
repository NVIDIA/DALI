// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/jitter.h"
#include "ndll/pipeline/operators/util/randomizer_impl_cpu.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace ndll {

// TODO(ptredak): re-enable it once RNG is changed on the CPU to be deterministic
// NDLL_REGISTER_OPERATOR(Jitter, Jitter<CPUBackend>, CPU);

NDLL_SCHEMA(Jitter)
    .DocStr(R"code(Perform a random Jitter augmentation.
    The output image is produced by moving each pixel by a
    random amount bounded by half of `nDegree` parameter
    (in both x and y dimensions).)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("nDegree",
        R"code(`int`
        Each pixel is moved by a random amount in
        range `[-nDegree/2, nDegree/2]`.)code", 2)
    .AddParent("DisplacementFilter");

}  // namespace ndll
