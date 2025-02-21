// Copyright (c) 2019, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <vector>

#include "dali/core/dev_buffer.h"
#include "dali/core/format.h"
#include "dali/core/math_util.h"
#include "dali/core/tensor_shape_print.h"

#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/test/resampling_test/resampling_test_params.h"

#include "dali/test/cv_mat_utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/dynamic_scratchpad.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {
namespace resample_test {

static constexpr int kMaxChannels = 16;

struct Bubble {
  vec3 centre;
  float color[kMaxChannels];
  float frequency;
  float decay;
};

template <typename T>
__global__ void DrawBubblesKernel(T *data, ivec3 size, int nch,
                                  const Bubble *bubbles, int nbubbles) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  T *pixel = &data[nch * (x + size.x * (y + size.y * z))];

  vec3 pos(x + 0.5f, y + 0.5f, z + 0.5f);

  float color[kMaxChannels] = { 0 };
  for (int i = 0; i < nbubbles; i++) {
    float dsq = (bubbles[i].centre - pos).length_square();
    float d = dsq*rsqrt(dsq);
    float magnitude = expf(bubbles[i].decay * dsq);
    float phase = bubbles[i].frequency * d;
    for (int c = 0; c < nch; c++)
      color[c] += bubbles[i].color[c] * (1 + cos(phase)) * magnitude * 0.5f;
  }
  for (int c = 0; c < nch; c++)
    pixel[c] = ConvertSatNorm<T>(color[c]);
}

template <typename T>
struct TestDataGenerator {
  DeviceBuffer<Bubble> gpu_bubbles;

  template <int ndim>
  void DrawBubbles(const TensorView<StorageGPU, T, ndim> &tensor, span<const Bubble> bubbles,
                   cudaStream_t stream) {
    static_assert(ndim == 4 || ndim == DynamicDimensions, "Tensor must be 4D or dynamic (and 4D)");
    assert(tensor.dim() == 4 && "Tensor must be 4D");
    gpu_bubbles.from_host(bubbles.data(), bubbles.size(), stream);
    ivec3 size(tensor.shape[2], tensor.shape[1], tensor.shape[0]);
    int nch = tensor.shape[3];
    assert(tensor.shape[3] <= kMaxChannels);
    dim3 block(32, 32, 1);
    dim3 grid(div_ceil(size.x, 32), div_ceil(size.y, 32), size.z);
    DrawBubblesKernel<<<grid, block, 0, stream>>>(tensor.data, size, nch,
                                                  gpu_bubbles, bubbles.size());
  }

  template <int ndim>
  void GenerateTestData(const TensorView<StorageGPU, T, ndim> &tensor, int num_bubbles = 5,
                        cudaStream_t stream = 0) {
    static_assert(ndim == 4 || ndim == DynamicDimensions, "Tensor must be 4D or dynamic (and 4D)");
    assert(tensor.dim() == 4 && "Tensor must be 4D");
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<float> dist(0, 1);
    std::uniform_real_distribution<float> freq_dist(M_PI/10, M_PI/3);
    std::uniform_real_distribution<float> sigma_dist(10, 100);

    auto shape = tensor.shape;
    int nch = shape[3];
    assert(nch <= kMaxChannels);

    std::vector<Bubble> bubbles(num_bubbles);
    for (int i = 0; i < num_bubbles; i++) {
      bubbles[i].centre = { shape[2] * dist(rng), shape[1] * dist(rng), shape[0] * dist(rng) };
      for (int c = 0; c < nch; c++)
        bubbles[i].color[c] = dist(rng);
      bubbles[i].frequency = freq_dist(rng);
      bubbles[i].decay = -1/(M_SQRT2 * sigma_dist(rng));
    }
    DrawBubbles(tensor, make_span(bubbles), stream);
  }
};

// Slices - duplicate params and shapes for depth slices as if they were additional samples

template <int ndim>
TensorListShape<ndim == DynamicDimensions ? DynamicDimensions : ndim-1>
GetSliceShapes(const TensorListShape<ndim> &tls) {
  TensorListShape<ndim == DynamicDimensions ? DynamicDimensions : ndim-1> slice_tls;
  int N = tls.num_samples();
  int total_slices = 0;
  for (int i = 0; i < N; i++) {
    total_slices += tls.tensor_shape_span(i)[0];
  }

  int D = tls.sample_dim() - 1;
  slice_tls.resize(total_slices, D);

  for (int i = 0, slice = 0; i < N; i++) {
    auto ts = tls.tensor_shape_span(i);
    for (int z = 0; z < ts[0]; z++, slice++) {
      auto slice_ts = slice_tls.tensor_shape_span(slice);
      for (int d = 0; d < D; d++) {
        slice_ts[d] = ts[d+1];
      }
    }
  }
  return slice_tls;
}

template <typename Storage, typename T, int ndim>
auto GetSliceImages(const TensorListView<Storage, T, ndim> &volumes) {
  return reshape(volumes, GetSliceShapes(volumes.shape), true);
}

template <int ndim>
void GetSliceParams(vector<ResamplingParams2D> &slice_params,
                    span<const ResamplingParams3D> params,
                    const TensorListShape<ndim> &in_shape) {
  slice_params.clear();
  int N = in_shape.num_samples();
  assert(static_cast<int>(params.size()) == N);
  for (int i = 0; i < N; i++) {
    int depth = in_shape.tensor_shape_span(i)[0];
    ResamplingParams2D p;
    p[0] = params[i][1];
    p[1] = params[i][2];
    for (int z = 0; z < depth; z++) {
      slice_params.push_back(p);
    }
  }
}

// ZShapes, ZImages - resize Z dim, fuse XY and keep old size

template <int ndim>
auto GetZShapes(const TensorListShape<ndim> &tls) {
  return collapse_dim(tls, 1);
}

template <typename Storage, typename T, int ndim>
auto GetZImages(const TensorListView<Storage, T, ndim> &volumes) {
  return reshape(volumes, GetZShapes(volumes.shape), true);
}

/**
 * @param z_params - parameters for resizing along Z axis, keeping fused XY intact
 * @param params   - original parameters
 * @param in_shape - input shape for _this stage_ (if Z is resized after XY, it is tmp_shape)
 *
 * @remarks This function cannot work with ROI in X/Y axes - it must be run as the second stage
 *          (after resizing all the slices).
 */
template <int ndim>
void GetZParams(vector<ResamplingParams2D> &z_params,
                span<const ResamplingParams3D> params,
                const TensorListShape<ndim> &in_shape) {
  z_params.clear();
  int N = in_shape.num_samples();
  assert(static_cast<int>(params.size()) == N);
  for (int i = 0; i < N; i++) {
    auto sample_shape = in_shape.tensor_shape_span(i);
    int depth = sample_shape[0];
    ResamplingParams2D p = {};
    p[0] = params[i][0];
    p[1].output_size = sample_shape[1] * sample_shape[2];
    p[1].roi.start = 0;
    p[1].roi.end = p[1].output_size;
    z_params.push_back(p);
  }
}


/**
 * @brief Use 2x 2D resampling to achieve 3D
 *
 * The first step decomposes the resampling into slices and resamples XY dimensions, fusing depth
 * and batch dim.
 * The second step fuses XY dimensions into generalized rows - which is OK, since we don't resize
 * that dimension and ROI is already applied. The Z dimension becomes the new Y.
 *
 * The result may differ slightly between this and true 3D resampling, because the order of
 * operations is not optimized and may be different.
 */
template <typename Out, typename In>
void Resample3Dvia2D(TestTensorList<Out> &out,
                     TestTensorList<In> &in,
                     span<const ResamplingParams3D> params,
                     cudaStream_t stream) {
  TestTensorList<float> tmp;

  auto in_view = in.template gpu<4>(stream);
  const auto &in_shape = in_view.shape;
  assert(in_shape.sample_dim() == 4);

  TensorListShape<4> tmp_shape, out_shape;
  int N = in_shape.num_samples();
  tmp_shape.resize(N);
  out_shape.resize(N);
  for (int i = 0; i < N; i++) {
    auto in_sample_shape  = in_shape.tensor_shape_span(i);
    auto tmp_sample_shape = tmp_shape.tensor_shape_span(i);
    auto out_sample_shape = out_shape.tensor_shape_span(i);
    for (int d = 0; d < 3; d++) {
      out_sample_shape[d] = params[i][d].output_size;
      if (out_sample_shape[d] == KeepOriginalSize)
        out_sample_shape[d] = in_sample_shape[d];
      tmp_sample_shape[d] = d == 0 ? in_sample_shape[d] : out_sample_shape[d];
    }
    tmp_sample_shape[3] = out_sample_shape[3] = in_sample_shape[3];  // number of channels
  }

  tmp.reshape(tmp_shape);
  out.reshape(out_shape);
  auto tmp_view = tmp.gpu<4>(stream);
  auto out_view = out.template gpu<4>(stream);

  vector<ResamplingParams2D> params_xy;
  vector<ResamplingParams2D> params_z;

  GetSliceParams(params_xy, params, in_shape);
  auto in_slices = GetSliceImages(in_view);
  auto tmp_slices = GetSliceImages(tmp_view);
  assert(in_slices.num_samples() == tmp_slices.num_samples());

  {
    ResampleGPU<float, In, 2> res_xy;

    KernelContext ctx;
    ctx.gpu.stream = stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    auto req = res_xy.Setup(ctx, in_slices, make_span(params_xy));

    assert(req.output_shapes[0] == tmp_slices.shape);
    res_xy.Run(ctx, tmp_slices, in_slices, make_span(params_xy));
  }

  GetZParams(params_z, params, tmp_shape);
  auto tmp_z = GetZImages(tmp_view);
  auto out_z = GetZImages(out_view);

  {
    ResampleGPU<Out, float, 2> res_z;

    KernelContext ctx;
    ctx.gpu.stream = stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    auto req = res_z.Setup(ctx, tmp_z, make_span(params_z));
    assert(req.output_shapes[0] == out_z.shape);
    res_z.Run(ctx, out_z, tmp_z, make_span(params_z));
  }
}

template <typename TestParams>
class Resample3DTest;

template <typename Out, typename In, ResamplingFilterType interp>
struct ResamplingTestParams {
  using OutputType = Out;
  using InputType = In;
  static constexpr ResamplingFilterType interp_type() { return interp; }
};

template <typename Out, typename In, ResamplingFilterType interp>
class Resample3DTest<ResamplingTestParams<Out, In, interp>>
: public ::testing::Test {
 public:
  Resample3DTest() {
    InitShapes();
  }

 protected:
  void InitShapes() {
    in_shapes.resize(3);
    out_shapes.resize(3);

    // NOTE: The shapes are chosen as to avoid source pixel centers exactly halfway
    // between original pixels, because it can lead to rounding discrepancies between
    // cpu and gpu variants (and we're using two-pass GPU as a reference here).

    // 3 channels
    in_shapes[0] = {{
      { 40, 60, 50, 3 },
      { 32, 80, 120, 3 },
    }};

    out_shapes[0] = {{
      { 51, 40, 70, 3 },
      { 73, 87, 29, 3 },
    }};

    // variable number of channels
    in_shapes[1] = {{
      { 10, 200, 120, 1 },
      { 100, 10, 10, 3 },
      { 70, 80, 90, 6 },
    }};

    out_shapes[1] = {{
      { 31, 200, 120, 1 },
      { 51, 27, 33, 3 },
      { 73, 181, 43, 6 },
    }};

    // many channels
    in_shapes[2] = {{
      { 40, 40, 40, 11 },
    }};

    out_shapes[2] = {{
      { 51, 51, 51, 11 },
    }};
  }

  vector<ResamplingParams3D> GenerateParams(const TensorListShape<4> &out_shape,
                                            const TensorListShape<4> &in_shape) {
    vector<ResamplingParams3D> params;
    params.resize(in_shape.num_samples());
    std::bernoulli_distribution dist;
    std::uniform_real_distribution<float> start_dist(0.05, 0.3);
    std::uniform_real_distribution<float> end_dist(0.7, 0.95);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      auto in_sample_shape = in_shape.tensor_shape_span(i);
      for (int d = 0; d < 3; d++) {
        params[i][d].min_filter = interp;
        params[i][d].mag_filter = interp;
        params[i][d].output_size = out_shape.tensor_shape_span(i)[d];
        if (d == 2) {
          do {
            params[i][d].roi.use_roi = true;
            params[i][d].roi.start = start_dist(rng) * in_sample_shape[d];
            params[i][d].roi.end = end_dist(rng) * in_sample_shape[d];
            if (dist(rng))
              std::swap(params[i][d].roi.start, params[i][d].roi.end);
          } while (interp == ResamplingFilterType::Nearest &&
              !CheckNN(params[i][d].output_size, params[i][d].roi.start, params[i][d].roi.end));
        }
      }
    }
    return params;
  }

  // Checks for possible rounding problems leading to selecting different source pixel
  // when running NN resampling.
  static bool CheckNN(int size, float start, float end) {
    float step = (end - start) / size;
    float x = start + step * 0.5f;
    for (int i = 0; i < size; i++, x += step) {
      if (std::abs(x - std::floor(x)) < 0.01f)
        return false;
    }
    return true;
  }

  void RunGPU() {
    cudaStream_t stream = 0;

    ResampleGPU<Out, In, 3> kernel;
    KernelContext ctx;
    ctx.gpu.stream = stream;

    TestDataGenerator<In> tdg;
    TestTensorList<In> in;
    TestTensorList<Out> out, ref;

    int niter = NumIter();
    for (int iter = 0; iter < niter; iter++) {
      const TensorListShape<4> &in_shape = in_shapes[iter];
      int N = in_shape.num_samples();
      in.reshape(in_shape);
      for (int i = 0; i < N; i++)
        tdg.GenerateTestData(in.gpu(stream)[i], 30, stream);

      const TensorListShape<4> &out_shape = out_shapes[iter];

      vector<ResamplingParams3D> params = GenerateParams(out_shape, in_shape);

      Resample3Dvia2D(ref, in, make_span(params), stream);
      auto ref_cpu = ref.cpu(stream);
      assert(ref_cpu.shape == out_shape);

      auto req = kernel.Setup(ctx, in.template gpu<4>(stream), make_span(params));
      ASSERT_EQ(req.output_shapes.size(), 1u) << "Expected only 1 output";
      ASSERT_EQ(req.output_shapes[0], out_shape) << "Unexpected output shape";
      out.reshape(out_shape);
      CUDA_CALL(
        cudaMemsetAsync(out.gpu(stream).data[0], 0, sizeof(Out)*out_shape.num_elements(), stream));

      auto out_gpu = out.template gpu<4>(stream);

      DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
      ctx.scratchpad = &dyn_scratchpad;

      kernel.Run(ctx, out_gpu, in.template gpu<4>(stream), make_span(params));

      auto out_cpu = out.cpu(stream);
      if (interp == ResamplingFilterType::Nearest) {
        Check(out_cpu, ref_cpu);
      } else {
        // Epsilons are quite big because, processing order in the reference is forced to be XYZ
        // or YXZ, whereas the tested implementation can use any order.
        double eps = std::is_integral<Out>::value ? 1 : 1e-3;
        Check(out_cpu, ref_cpu, EqualEpsRel(eps, 1e-4));
      }
    }
  }

  void RunCPU() {
    cudaStream_t stream = 0;

    ResampleCPU<Out, In, 3> kernel;
    KernelContext ctx;

    TestDataGenerator<In> tdg;
    TestTensorList<In> in;
    TestTensorList<Out> out, ref;

    int niter = NumIter();
    for (int iter = 0; iter < niter; iter++) {
      const TensorListShape<4> &in_shape = in_shapes[iter];
      int N = in_shape.num_samples();
      in.reshape(in_shape);
      for (int i = 0; i < N; i++)
        tdg.GenerateTestData(in.gpu(stream)[i], 10, stream);

      const TensorListShape<4> &out_shape = out_shapes[iter];
      out.reshape(out_shape);
      memset(out.cpu(stream).data[0], 0, sizeof(Out)*out_shape.num_elements());

      vector<ResamplingParams3D> params = GenerateParams(out_shape, in_shape);

      if (iter != 1)
        continue;

      Resample3Dvia2D(ref, in, make_span(params), stream);
      auto ref_cpu = ref.cpu(stream);
      ref.invalidate_gpu();

      assert(ref_cpu.shape == out_shape);
      auto in_cpu = in.template cpu<4>(stream);
      auto out_cpu = out.template cpu<4>(stream);

      for (int i = 0; i < N; i++) {
        auto req = kernel.Setup(ctx, in_cpu[i], params[i]);
        ASSERT_EQ(req.output_shapes.size(), 1u) << "Expected only 1 output";
        ASSERT_EQ(req.output_shapes[0][0], out_shape[i]) << "Unexpected output shape";

        DynamicScratchpad dyn_scratchpad(AccessOrder::host());
        ctx.scratchpad = &dyn_scratchpad;

        kernel.Run(ctx, out_cpu[i], in_cpu[i], params[i]);

        if (interp == ResamplingFilterType::Nearest) {
          Check(out_cpu[i], ref_cpu[i]);
        } else {
          // Epsilons are quite big because:
          // - GPU uses fma
          // - GPU uses different rounding
          // - processing order in the reference is forced to be XYZ or YXZ, whereas
          //   the tested implementation can use any order.
          double eps = std::is_integral<Out>::value ? 1 :
                       std::is_integral<In>::value ? max_value<In>()*1e-6 : 1e-5;
          Check(out_cpu[i], ref_cpu[i], EqualEpsRel(eps, 2e-3));
        }
      }
    }
  }

  vector<TensorListShape<4>> in_shapes, out_shapes;

  int NumIter() const { return in_shapes.size(); }
  std::mt19937_64 rng{1234};
};

using Resample3DTestTypes = ::testing::Types<
  ResamplingTestParams<uint8_t, uint8_t, ResamplingFilterType::Nearest>,
  ResamplingTestParams<float, uint8_t, ResamplingFilterType::Linear>,
  ResamplingTestParams<int16_t, int16_t, ResamplingFilterType::Cubic>,
  ResamplingTestParams<float, uint16_t, ResamplingFilterType::Lanczos3>
>;

TYPED_TEST_SUITE(Resample3DTest, Resample3DTestTypes);

TYPED_TEST(Resample3DTest, TestGPU) {
  this->RunGPU();
}

TYPED_TEST(Resample3DTest, TestCPU) {
  this->RunCPU();
}

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
