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

#include <gtest/gtest.h>
#include <random>
#include "dali/kernels/reduce/reduce_axes_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/kernels/alloc.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {

template <typename T>
void copyD2D(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
void copyD2H(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void copyH2D(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
void copyH2H(T *dst, const T *src, size_t n, cudaStream_t stream = 0) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToHost, stream));
}

/**
 * @brief Represents a strongly typed device-side buffer with size and capacity.
 *
 * This class behaves somewhat like vector in terms of storage growth.
 * It doesn't support copy construction/assignment nor indexing (not possible on host-side).
 * It does support common query functions, exponential resize and shrinking.
 * It also provides copy utilities from both host- and device-side sources, with ability
 * to specify CUDA stream.
 */
template <typename T>
struct DeviceBuffer {
  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer &&other) {
    *this = other;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) {
    data_ = std::move(other.data_);
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  size_t size() const { return size_; }
  size_t size_bytes() const { return sizeof(T) * size_; }
  size_t capacity() const { return capacity_; }

  operator T *() { return data_.get(); }
  operator const T *() const { return data_.get(); }

  T *data() { return data_.get(); }
  const T *data() const { return data_.get(); }

  bool empty() const { return size_ == 0; }

  void clear() { size_ = 0; }
  void free() { size_ = 0; capacity_ = 0; data_.reset(); }

  void shrink_to_fit(cudaStream_t stream = 0) {
    reallocate(size_, stream);
  }

  void from_host(const T *source, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    copyH2D(data_.get(), source, size(), stream);
  }

  void from_device(const T *source, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    copyD2D(data_.get(), source, size(), stream);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_host(const ArrayLike &source, cudaStream_t stream = 0) {
    from_host(&source[0], dali::size(source), stream);
  }

  template <typename ArrayLike>
  if_array_like<ArrayLike> from_device(const ArrayLike &source, cudaStream_t stream = 0) {
    from_device(&source[0], dali::size(source), stream);
  }

  void copy(const DeviceBuffer &src, cudaStream_t stream = 0) {
    clear();
    resize(src.size());
    copyH2D(data_.get(), src.data(), size(), stream);
  }

  void resize(size_t new_size, cudaStream_t stream = 0) {
    if (new_size > capacity_) {
      size_t new_cap = max(2 * capacity_, new_size);
      reallocate(new_cap, stream);
    }
    size_ = new_size;
  }

private:
  void reallocate(size_t new_cap, cudaStream_t stream) {
    if (size_ == 0) {
      data_.reset();
      capacity_ = size_ = 0;
      data_ = kernels::memory::alloc_unique<T>(AllocType::GPU, new_cap);
      capacity_ = new_cap;
    } else {
      auto new_data = kernels::memory::alloc_unique<T>(AllocType::GPU, new_cap);
      copyD2D(new_data.get(), data_.get(), size(), stream);
    }
  }

  kernels::memory::KernelUniquePtr<T> data_;
  size_t capacity_ = 0;
  size_t size_ = 0;
};

template <typename Out, typename Reduction, typename T>
void RefReduceInner(Out *out, const T *in, int64_t n_outer, int64_t n_inner, const Reduction &R) {
  for (int64_t outer = 0, offset = 0; outer < n_outer; outer++, offset += n_inner) {
    out[outer] = RefReduce<Out>(make_span(in + offset, n_inner), R);
  }
}

using int_dist = std::uniform_int_distribution<int>;

template <typename Reduction>
class ReduceInnerGPUTest : public ::testing::Test {
 public:
  using SampleDesc = ReduceInnerSampleDesc<float, float>;

  void PrepareData(int N, int_dist outer_shape_dist, int_dist inner_shape_dist) {
    std::uniform_real_distribution<float> dist(0, 1);
    std::mt19937_64 rng;

    TensorListShape<2> tls;
    TensorListShape<2> out_tls;

    this->N = N;
    tls.resize(N);
    out_tls.resize(N);
    for (int i = 0; i < N; i++) {
      int outer = outer_shape_dist(rng);
      int inner = inner_shape_dist(rng);
      tls.set_tensor_shape(i, { outer, inner });
    }
    in.reshape(tls);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, 0, 1);

    for (int i = 0; i < N; i++) {
      auto ts = tls[i];
      SampleDesc desc;
      desc.n_outer = ts[0];
      desc.n_inner = ts[1];
      desc.inner_macroblocks = 1;
      desc.inner_macroblock_size = desc.n_inner;
      while (desc.inner_macroblock_size > 0x8000) {
        desc.inner_macroblocks <<= 1;
        desc.inner_macroblock_size = div_ceil(desc.n_inner, desc.inner_macroblocks);
      }
      out_tls.set_tensor_shape(i, { desc.n_outer, desc.inner_macroblocks });
      cpu_descs.push_back(desc);
    }
    out.reshape(out_tls);

    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();
    for (int i = 0; i < N; i++) {
      SampleDesc &desc = cpu_descs[i];
      desc.in = gpu_in.data[i];
      desc.out = gpu_out.data[i];
    }
  }

  void Run() {
    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();
    gpu_descs.from_host(cpu_descs);

    dim3 grid(32, N);
    dim3 block(32, 32);
    auto start = CUDAEvent::CreateWithFlags(0);
    auto end =   CUDAEvent::CreateWithFlags(0);
    cudaEventRecord(start);
    ReduceInnerKernel<<<grid, block>>>(gpu_descs.data(), reduction);
    cudaEventRecord(end);
    CUDA_CALL(cudaDeviceSynchronize());
    float t = 0;
    cudaEventElapsedTime(&t, start, end);
    t /= 1000;  // convert to seconds
    int64_t read = gpu_in.num_elements() * sizeof(float);
    int64_t written = gpu_out.num_elements() * sizeof(float);
    std::cerr << (read + written) / t * 1e-9 << " GB/s" << endl;
    CheckResult();
  }

  void CheckResult() {
    auto cpu_out = out.cpu();
    auto cpu_in = in.cpu();
    auto in_shape = cpu_in.shape;
    auto out_shape = cpu_out.shape;

    vector<float> ref_out;
    vector<float> full_out;  // when out is not a full reduction, we calculate the second stage here
    for (int i = 0; i < N; i++) {
      auto ts = in_shape[i];
      int64_t outer = ts[0];
      int64_t inner = ts[1];
      ref_out.resize(outer);
      RefReduceInner(ref_out.data(), cpu_in.data[i], outer, inner, reduction);
      auto out = cpu_out[i];
      if (out.shape[1] > 1) {
        full_out.resize(outer);
        RefReduceInner(full_out.data(), out.data, outer, out.shape[1], reduction);
        out = make_tensor_cpu<2>(full_out.data(), { outer, 1 });
      }
      auto ref = make_tensor_cpu<2>(ref_out.data(), { outer, 1 });
      Check(out, ref, EqualEpsRel(1e-6, 1e-6));
    }
  }

  Reduction reduction;
  int N;
  TestTensorList<float, 2> in, out;
  std::vector<SampleDesc> cpu_descs;
  DeviceBuffer<SampleDesc> gpu_descs;

};

using ReductionTestTypes = ::testing::Types<reductions::sum, reductions::min, reductions::max>;

TYPED_TEST_SUITE(ReduceInnerGPUTest, ReductionTestTypes);

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_2_63) {
  this->PrepareData(10, int_dist(10000, 200000), int_dist(2, 63));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_32_256) {
  this->PrepareData(10, int_dist(10000, 20000), int_dist(32, 256));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_64_1024) {
  this->PrepareData(10, int_dist(10000, 20000), int_dist(64, 1024));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_1k_32k) {
  this->PrepareData(10, int_dist(1, 100), int_dist(1024, 32*1024));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_16k_1M) {
  this->PrepareData(10, int_dist(1, 10), int_dist(16*1024, 1024*1024));
  this->Run();
}

}  // namespace kernels
}  // namespace dali
