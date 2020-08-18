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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dali/core/dev_buffer.h"
#include "dali/core/math_util.h"
#include "dali/core/tensor_shape_print.h"

#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/test/resampling_test/resampling_test_params.h"

#include "dali/test/cv_mat_utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {
namespace resample_test {

struct Bubble {
  vec3 centre;
  vec3 color;
  float frequency;
  float decay;
};

template <typename T>
__global__ void DrawBubblesKernel(T *data, ivec3 size, const Bubble *bubbles, int nbubbles) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  T *pixel = &data[3 * (x + size.x * (y + size.y * z))];

  vec3 pos(x + 0.5f, y + 0.5f, z + 0.5f);

  vec3 color(0, 0, 0);
  for (int i = 0; i < nbubbles; i++) {
    float dsq = (bubbles[i].centre - pos).length_square();
    float d = dsq*rsqrt(dsq);
    float magnitude = expf(bubbles[i].decay * dsq);
    float phase = bubbles[i].frequency * d;
    color += bubbles[i].color * (1 + cos(phase)) * magnitude * 0.5f;
  }
  pixel[0] = ConvertSatNorm<T>(color[0]);
  pixel[1] = ConvertSatNorm<T>(color[1]);
  pixel[2] = ConvertSatNorm<T>(color[2]);
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
    assert(tensor.shape[3] == 3);
    dim3 block(32, 32, 1);
    dim3 grid(div_ceil(size.x, 32), div_ceil(size.y, 32), size.z);
    DrawBubblesKernel<<<grid, block, 0, stream>>>(tensor.data, size,
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
    assert(shape[3] == 3);  // only RGB

    std::vector<Bubble> bubbles(num_bubbles);
    for (int i = 0; i < num_bubbles; i++) {
      bubbles[i].centre = { shape[2] * dist(rng), shape[1] * dist(rng), shape[0] * dist(rng) };
      bubbles[i].color = { dist(rng), dist(rng), dist(rng) };
      bubbles[i].frequency = freq_dist(rng);
      bubbles[i].decay = -1/(M_SQRT2 * sigma_dist(rng));
    }
    DrawBubbles(tensor, make_span(bubbles), stream);
  }
};

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
                    const TensorListShape<ndim> &tls) {
  slice_params.clear();
  int N = tls.num_samples();
  assert(static_cast<int>(params.size()) == N);
  for (int i = 0; i < N; i++) {
    int depth = tls.tensor_shape_span(i)[0];
    ResamplingParams2D p;
    p[0] = params[i][1];
    p[1] = params[i][2];
    for (int z = 0; z < depth; z++) {
      slice_params.push_back(p);
    }
  }
}

template <int ndim>
auto GetZShapes(const TensorListShape<ndim> &tls) {
  return collapse_dim(tls, 0);
}

template <typename Storage, typename T, int ndim>
auto GetZImages(const TensorListView<Storage, T, ndim> &volumes) {
  return reshape(volumes, GetZShapes(volumes.shape), true);
}

template <int ndim>
void GetZParams(vector<ResamplingParams2D> &z_params,
                span<const ResamplingParams3D> params,
                const TensorListShape<ndim> &tls) {
  z_params.clear();
  int N = tls.num_samples();
  assert(static_cast<int>(params.size()) == N);
  for (int i = 0; i < N; i++) {
    auto sample_shape = tls.tensor_shape_span(i);
    int depth = sample_shape[0];
    ResamplingParams2D p;
    p[0] = params[i][0];
    p[1] = {};
    p[1].output_size = sample_shape[1] * sample_shape[2];
    p[1].roi.start = 0;
    p[1].roi.end = p[1].output_size;
    z_params.push_back(p);
  }
}


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

  ScratchpadAllocator sa;

  {
    ResampleGPU<float, In, 2> res_xy;
    KernelContext ctx;
    ctx.gpu.stream = stream;

    auto req = res_xy.Setup(ctx, in_slices, make_span(params_xy));
    sa.Reserve(req.scratch_sizes);
    auto scratch = sa.GetScratchpad();
    ctx.scratchpad = &scratch;
    assert(req.output_shapes[0] == tmp_shape);
    res_xy.Run(ctx, tmp_slices, in_slices, make_span(params_xy));
  }

  GetZParams(params_z, params, in_shape);
  auto tmp_z = GetZImages(tmp_view);
  auto out_z = GetZImages(out_view);

  {
    ResampleGPU<Out, float, 2> res_z;

    KernelContext ctx;
    ctx.gpu.stream = stream;
    auto req = res_z.Setup(ctx, tmp_z, make_span(params_z));
    sa.Reserve(req.scratch_sizes);
    auto scratch = sa.GetScratchpad();
    ctx.scratchpad = &scratch;
    assert(req.output_shapes[0] == out_shape);
    res_z.Run(ctx, out_z, tmp_z, make_span(params_z));
  }

}

TEST(Resample3D, DataGeneratorTest) {
  using T = uint8_t;
  TestDataGenerator<T> tdg;
  TestTensorList<T> tl;
  TensorListShape<4> shape = {{
    { 40, 60, 50, 3 },
    { 32, 80, 120, 3 },
  }};
  tl.reshape(shape);
  tdg.GenerateTestData(tl.gpu()[0], 10);
  tdg.GenerateTestData(tl.gpu()[1], 20);
  char fname[256];
  auto tl_cpu = tl.cpu();
  for (int idx = 0; idx < shape.num_samples(); idx++) {
    auto volume = tl_cpu[idx];
    for (int slice = 0; slice < shape[idx][0]; slice++) {
      snprintf(fname, sizeof(fname), "img_%i_slice_%i.png", idx, slice);
      testing::SaveImage(fname, subtensor(volume, slice));
    }
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
  void RunGPU() {
    cudaStream_t stream = 0;
    TestDataGenerator<In> tdg;
    TestTensorList<In> in;
    TensorListShape<4> shape = {{
      { 40, 60, 50, 3 },
      { 32, 80, 120, 3 },
    }};
    in.reshape(shape);
    tdg.GenerateTestData(in.gpu(stream)[0], 10, stream);
    tdg.GenerateTestData(in.gpu(stream)[1], 20, stream);

    TensorListShape<4> out_shape = {{
      { 20, 40, 60, 3 },
      { 72, 88, 128, 3 },
    }};

    vector<ResamplingParams3D> params;
    params.resize(shape.num_samples());
    for (int i = 0; i < shape.num_samples(); i++) {
      for (int d = 0; d < 3; d++) {
        params[i][d].min_filter = params[i][d].mag_filter = interp;
        params[i][d].output_size = out_shape.tensor_shape_span(i)[d];
      }
    }

    TestTensorList<Out> out, ref;

    Resample3Dvia2D(ref, in, make_span(params), stream);
    auto ref_cpu = ref.cpu(stream);
    ref.invalidate_gpu();  // free memory
    assert(ref_cpu.shape == out_shape);

    ResampleGPU<Out, In, 3> kernel;
    KernelContext ctx;
    ctx.gpu.stream = stream;
    auto req = kernel.Setup(ctx, in.template gpu<4>(stream), make_span(params));
    ASSERT_EQ(req.output_shapes.size(), 1u) << "Expected only 1 output";
    ASSERT_EQ(req.output_shapes[0], out_shape) << "Unexpected output shape";
    out.reshape(out_shape);
    cudaMemsetAsync(out.gpu(stream).data[0], 0, out_shape.num_elements(), stream);

    ScratchpadAllocator sa;
    sa.Reserve(req.scratch_sizes);
    auto scratchpad = sa.GetScratchpad();
    ctx.scratchpad = &scratchpad;
    kernel.Run(ctx, out.template gpu<4>(stream), in.template gpu<4>(stream), make_span(params));

    auto out_cpu = out.cpu(stream);
    double eps = std::is_integral<Out>::value ? 1 : 1e-5;
    Check(out_cpu, ref_cpu, EqualEpsRel(eps, 1e-5));
  }
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

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
