// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/common.h"
#include "dali/pipeline/operators/displacement/water.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Water, Water<CPUBackend>, CPU);

DALI_SCHEMA(Water)
    .DocStr("Perform a water augmentation (make image appear to be underwater).")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("ampl_x",
        R"code(`float`
        Amplitude of the wave in x direction.)code", 10.f)
    .AddOptionalArg("ampl_y",
        R"code(`float`
        Amplitude of the wave in y direction.)code", 10.f)
    .AddOptionalArg<float>("freq_x",
        R"code(`float`
        Frequency of the wave in x direction.)code", 2.0 * M_PI / 128)
    .AddOptionalArg<float>("freq_y",
        R"code(`float`
        Frequence of the wave in y direction.)code", 2.0 * M_PI / 128)
    .AddOptionalArg("phase_x",
        R"code(`float`
        Phase of the wave in x direction.)code", 0.f)
    .AddOptionalArg("phase_y",
        R"code(`float`
        Phase of the wave in y direction.)code", 0.f)
    .AddParent("DisplacementFilter");

}  // namespace dali

