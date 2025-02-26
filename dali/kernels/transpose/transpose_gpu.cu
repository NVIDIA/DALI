// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/transpose/transpose_gpu.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "dali/core/util.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/transpose/transpose_gpu_def.h"
#include "dali/kernels/transpose/transpose_gpu_impl.cuh"
#include "dali/kernels/transpose/transpose_gpu_setup.cuh"

namespace dali {
namespace kernels {
namespace transpose_impl {

struct TransposeInfo {
  int                 element_size;
  TransposeMethod     method;
  TensorShape<>       shape;
  SmallVector<int, 6> perm;
};

constexpr int kMaxInterleaveSize = 32;
constexpr int kMaxDeinterleaveSize = kMaxInterleaveSize;

inline bool UseTiledTranspose(const int64_t *shape, const int *perm, int ndim, int element_size) {
  if (perm[ndim-1] == ndim - 1) {
    assert(ndim >= 3);
    if (shape[ndim-1] * element_size >= kTiledTransposeMaxVectorSize)
      return false;

    ndim--;  // ignore last dimension - it will be treated as vector lanes
  }

  int xdim = ndim-1;
  int ydim = 0;
  for (; ydim < xdim; ydim++) {
    if (perm[ydim] == xdim)
      break;
  }
  double tile_coverage = shape[xdim] * shape[ydim];
  tile_coverage /= align_up(shape[xdim], kTileSize) * align_up(shape[ydim], kTileSize);
  return tile_coverage > 0.4;  // for now, it's an educated guess
}

TransposeMethod GetTransposeMethod(const int64_t *shape,
                                   const int *perm,
                                   int ndim,
                                   int element_size) {
  if (ndim <= 1)
    return TransposeMethod::Copy;

  if (UseTiledTranspose(shape, perm, ndim, element_size))
    return TransposeMethod::Tiled;

  if (perm[ndim-1] != ndim - 1) {
    if (shape[ndim-1] * element_size <= kMaxDeinterleaveSize)
      return TransposeMethod::Deinterleave;
    else if (shape[perm[ndim-1]] * element_size <= kMaxInterleaveSize)
      return TransposeMethod::Interleave;
  }

  return TransposeMethod::Generic;
}

void GetTransposeInfo(TransposeInfo &info, int element_size,
                      span<const int64_t> in_shape, span<const int> perm) {
  SimplifyPermute(info.shape, info.perm, in_shape.data(), perm.data(), in_shape.size());
  info.element_size = element_size;
  int ndim = info.shape.size();
  if (ndim > kMaxNDim)
    throw std::range_error("Transposition too complex");
  info.method = GetTransposeMethod(info.shape.data(), info.perm.data(), ndim, element_size);
}

void GetTransposeInfo(TransposeInfo *infos, int element_size,
                      const TensorListShape<> &tls, span<const int> perm) {
  int N = tls.num_samples();
  for (int i = 0; i < N; i++) {
    GetTransposeInfo(infos[i], element_size, tls.tensor_shape_span(i), perm);
  }
}

}  // namespace transpose_impl

using namespace transpose_impl;  // NOLINT

class TransposeGPU::Impl {
 public:
  template <typename T>
  KernelRequirements SetupTyped(
      const TensorListShape<> &in_shape,
      span<const int> permutation) {
    int N = in_shape.num_samples();
    int ndim = in_shape_.sample_dim();
    element_size_ = sizeof(T);
    in_shape_ = in_shape;
    permute_dims(out_shape_, in_shape_, permutation);

    infos_.resize(N);
    GetTransposeInfo(infos_.data(), sizeof(T), in_shape, permutation);

    tiled_descs_.clear();
    deinterleave_descs_.clear();
    generic_descs_.clear();
    idx_generic_.clear();
    idx_tiled_.clear();
    idx_deinterleave_.clear();

    tiled_descs_.reserve(infos_.size());
    deinterleave_descs_.reserve(infos_.size());
    generic_descs_.reserve(infos_.size());

    for (int i = 0; i < N; i++) {
      auto &shape = infos_[i].shape;
      auto perm = make_span(infos_[i].perm);
      switch (infos_[i].method) {
        case TransposeMethod::Tiled:
          {
            TiledTransposeDesc<T> desc;
            InitTiledTranspose(desc, shape, perm);
            AddDesc(desc);
            idx_tiled_.push_back(i);
          }
          break;
        case TransposeMethod::Deinterleave:
          {
            DeinterleaveDesc<T> desc;
            InitDeinterleave(desc, shape, perm);
            AddDesc(desc);
            idx_deinterleave_.push_back(i);
          }
          break;
        case TransposeMethod::Interleave:  // no specialized implementation yet
        case TransposeMethod::Copy:  // generic kernel does a good job at just copying
        default:
          {
            GenericTransposeDesc<T> desc;
            InitGenericTranspose(desc, shape, perm);
            AddDesc(desc);
            idx_generic_.push_back(i);
          }
          break;
      }
    }

    KernelRequirements req;
    req.output_shapes = { out_shape_ };
    return req;
  }

  KernelRequirements Setup(
      const TensorListShape<> &in_shape,
      span<const int> permutation,
      int element_size) {
    KernelRequirements req;
    VALUE_SWITCH(element_size, static_el_size, (1, 2, 4, 8, 16),
      (req = SetupTyped<type_of_size<static_el_size>>(in_shape, permutation)),
      (throw std::range_error("Transpose: Unexpected tensor element size."
                              "Must be one of (1,2,4,8,16)")));
    return req;
  }

  template <typename T>
  void RunTyped(KernelContext &ctx, T *const *out, const T *const *in) {
    RunTiled(ctx, out, in);
    RunDeinterleave(ctx, out, in);
    RunGeneric(ctx, out, in);
  }

  void Run(KernelContext &ctx, void *const *out, const void *const *in) {
    VALUE_SWITCH(element_size_, static_el_size, (1, 2, 4, 8, 16),
      (
        using T = type_of_size<static_el_size>;
        RunTyped(ctx, reinterpret_cast<T*const*>(out), reinterpret_cast<const T*const*>(in))
      ), (  // NOLINT
        throw std::range_error("Transpose: Unexpected tensor element size."
                               "Must be one of (1,2,4,8,16)")
      )  // NOLINT
    );   // NOLINT
  }


  template <typename T>
  void AddDesc(const GenericTransposeDesc<T> &desc) {
    generic_descs_.push_back(reinterpret_cast<const GenericTransposeDesc<void> &>(desc));
  }
  template <typename T>
  void AddDesc(const DeinterleaveDesc<T> &desc) {
    deinterleave_descs_.push_back(reinterpret_cast<const DeinterleaveDesc<void> &>(desc));
  }
  template <typename T>
  void AddDesc(const TiledTransposeDesc<T> &desc) {
    tiled_descs_.push_back(reinterpret_cast<const TiledTransposeDesc<void> &>(desc));
  }

  template <typename T>
  void RunGeneric(KernelContext &ctx, T *const *out, const T *const *in) {
    if (!generic_descs_.empty()) {
      uint64_t max_size = 0;
      int block_size = 256;
      for (size_t i = 0; i < generic_descs_.size(); i++) {
        generic_descs_[i].out = out[idx_generic_[i]];
        generic_descs_[i].in =  in[idx_generic_[i]];
        if (generic_descs_[i].size > max_size)
          max_size = generic_descs_[i].size;
      }
      auto *gpu_descs = reinterpret_cast<GenericTransposeDesc<T>*>(
        std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, generic_descs_)));

      dim3 grid(div_ceil(max_size, block_size * 8), generic_descs_.size());

      TransposeGenericBatch<<<grid, block_size, 0, ctx.gpu.stream>>>(gpu_descs);
    }
  }

  template <typename T>
  void RunTiled(KernelContext &ctx, T *const *out, const T *const *in) {
    if (!tiled_descs_.empty()) {
      int64_t max_tiles = 0;
      for (size_t i = 0; i < tiled_descs_.size(); i++) {
        if (tiled_descs_[i].total_tiles > max_tiles)
          max_tiles = tiled_descs_[i].total_tiles;
      }
      int grid_x = max_tiles;
      int threshold = 64 / tiled_descs_.size();
      if (grid_x > threshold) {
        grid_x = threshold + (grid_x - threshold) / 4;
      }
      for (size_t i = 0; i < tiled_descs_.size(); i++) {
        UpdateTiledTranspose(tiled_descs_[i], out[idx_tiled_[i]], in[idx_tiled_[i]], grid_x);
      }
      auto *gpu_descs = reinterpret_cast<TiledTransposeDesc<T>*>(
        ctx.scratchpad->ToGPU(ctx.gpu.stream, tiled_descs_));

      int max_threads = MaxThreadsPerBlockStatic(TransposeTiledBatch<T>);
      assert(max_threads >= kTileSize);

      int block_y = 16;  // start with 32x16 block and try smaller until found
      while (kTileSize * block_y > max_threads)
        block_y >>= 1;

      dim3 block(kTileSize, block_y);
      dim3 grid(grid_x, tiled_descs_.size());

      const int shm_size = kTiledTransposeMaxSharedMem;
      TransposeTiledBatch<<<grid, block, shm_size, ctx.gpu.stream>>>(gpu_descs);
    }
  }

  template <typename T>
  void RunDeinterleave(KernelContext &ctx, T *const *out, const T *const *in) {
    if (!deinterleave_descs_.empty()) {
      int64_t max_size = 0;
      int block_size = 256;

      for (size_t i = 0; i < deinterleave_descs_.size(); i++) {
        auto &desc = deinterleave_descs_[i];
        desc.out = out[idx_deinterleave_[i]];
        desc.in =  in[idx_deinterleave_[i]];
        int64_t outer_size = desc.size / desc.in_strides[desc.ndim-2];
        if (outer_size > max_size)
          max_size = outer_size;
      }

      auto *gpu_descs = reinterpret_cast<DeinterleaveDesc<T>*>(
        std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, deinterleave_descs_)));

      dim3 grid(div_ceil(max_size, 4*block_size), deinterleave_descs_.size());
      TransposeDeinterleaveBatch<<<grid, block_size, 0, ctx.gpu.stream>>>(gpu_descs);
    }
  }

  int element_size_ = 0;
  TensorListShape<> in_shape_, out_shape_;
  std::vector<TransposeInfo> infos_;
  std::vector<GenericTransposeDesc<void>> generic_descs_;
  std::vector<TiledTransposeDesc<void>>   tiled_descs_;
  std::vector<DeinterleaveDesc<void>>     deinterleave_descs_;
  std::vector<int> idx_generic_, idx_tiled_, idx_deinterleave_;  // sample indices
};

TransposeGPU::TransposeGPU() {
  impl_ = std::make_unique<Impl>();
}

TransposeGPU::~TransposeGPU() = default;

void TransposeGPU::CheckShapes(const TensorListShape<> &in_shape,
                               const TensorListShape<> &out_shape,
                               int element_size) {
  assert(impl_ != nullptr);
  DALI_ENFORCE(impl_->in_shape_ == in_shape, "Input shape different than used in Setup");
  DALI_ENFORCE(impl_->out_shape_ == out_shape,
    "Output shape does not match the one produced in Setup");
  DALI_ENFORCE(impl_->element_size_ == element_size,
                "Different element size than used in Setup");
}

KernelRequirements TransposeGPU::Setup(
    KernelContext &ctx,
    const TensorListShape<> &in_shape,
    span<const int> permutation,
    int element_size) {
  assert(impl_ != nullptr);
  return impl_->Setup(in_shape, permutation, element_size);
}

void TransposeGPU::Run(KernelContext &ctx, void *const *out, const void *const *in) {
  assert(impl_ != nullptr);
  impl_->Run(ctx, out, in);
}

}  // namespace kernels
}  // namespace dali
