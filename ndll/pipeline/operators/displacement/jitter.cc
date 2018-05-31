// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/jitter.h"
#include "ndll/pipeline/operators/util/randomizer_impl_cpu.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace ndll {

// TODO(ptredak): re-enable it once RNG is changed on the CPU to be deterministic
// NDLL_REGISTER_OPERATOR(Jitter, Jitter<CPUBackend>, CPU);

NDLL_SCHEMA(Jitter)
    .DocStr("Perform a Jitter-style augmentation")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("nDegree", "Foo", 2)
    DISPLACEMENT_SCHEMA_ARGS;

}  // namespace ndll
