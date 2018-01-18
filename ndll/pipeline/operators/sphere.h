// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_SPHERE_H_
#define NDLL_PIPELINE_OPERATORS_SPHERE_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"

namespace ndll {

#define SPHERE_PREAMBLE(H, W, C)                \
    const int mid_x = W / 2;                    \
    const int mid_y = H / 2;                    \
    const int d = mid_x > mid_y ? mid_x : mid_y;\
    int newX, newY;                             \
    const int nYOffset = (W * C + 1) / 2 * 2;

#define SPHERE_CORE(H, W, C)                                        \
    const int trueY = h - mid_y;                                    \
    const int trueX = w - mid_x;                                    \
    const double rad = sqrtf(trueX * trueX + trueY * trueY) / d;    \
    const int from = (newX = mid_x + rad * trueX) > 0 && newX < W &&\
                     (newY = mid_y + rad * trueY) > 0 && newY < H ? \
                      newX * C + newY * nYOffset : 0;

NDLLError_t BatchedSphere(const uint8 *in_batch, int N, const Dims &dims,
                          uint8 *out_batch, cudaStream_t stream);

template <typename Backend>
class Sphere : public Operator<Backend> {
 public:
    inline explicit Sphere(const OpSpec &spec) : Operator<Backend>(spec) {}

    virtual ~Sphere() = default;

 protected:
    void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        const auto &output = ws->Output<CPUBackend>(idx);

        NDLL_ENFORCE(input.ndim() == 3);
        NDLL_ENFORCE(IsType<uint8>(input.type()),
                     "Expects input data in uint8.");

        const vector<Index> &shape = input.shape();
        const int H = shape[0];
        const int W = shape[1];
        const int C = shape[2];
        NDLL_ENFORCE(C == 1 || C == 3,
                     "Sphere supports hwc rgb & grayscale inputs.");

        output->Resize(shape);
        AUGMENT_TRANSFORM_CPU(H, W, C, input.template data<uint8>(),
                              static_cast<uint8*>(output->raw_mutable_data()), SPHERE);
    }

    void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
        DataDependentSetup(ws, idx);

        const auto &input = ws->Input<GPUBackend>(idx);
        const auto output = ws->Output<GPUBackend>(idx);
        const auto &shape = input.shape();

        NDLL_CALL(BatchedSphere(
                input.template data<uint8>(),
                batch_size_, shape[0],
                static_cast<uint8*>(output->raw_mutable_data()),
                ws->stream()));
    }

    inline void DataDependentSetup(DeviceWorkspace *ws, const int idx) {
        const auto &input = ws->Input<GPUBackend>(idx);
        const auto output = ws->Output<GPUBackend>(idx);
        NDLL_ENFORCE(IsType<uint8>(input.type()),
                     "Expected input data stored in uint8.");

        vector<Dims> output_shape(batch_size_);
        for (int i = 0; i < batch_size_; ++i) {
            // Verify the inputs
            const auto &input_shape = input.tensor_shape(i);
            NDLL_ENFORCE(input_shape.size() == 3,
                         "Expects 3-dimensional image input.");

            NDLL_ENFORCE(input_shape[2] == 1 || input_shape[2] == 3,
                       "Not valid color type argument (1 or 3)");

            // Collect the output shapes
            output_shape[i] = input_shape;
        }

        // Resize the output
        output->Resize(output_shape);
    }

 private:
    USE_OPERATOR_MEMBERS();
};
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_SPHERE_H_
