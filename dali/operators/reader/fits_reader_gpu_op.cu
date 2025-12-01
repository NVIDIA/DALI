// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/operators/reader/fits_reader_gpu_op.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

template <typename T>
__device__ unsigned int read_lastpix(unsigned char *compressed_tile) {
  unsigned int lastpix = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    unsigned char bytevalue = compressed_tile[i];
    lastpix = lastpix | (bytevalue << ((sizeof(T) - i - 1) << 3));
  }
  return lastpix;
}

template <typename T>
__global__ void rice_decompress(unsigned char *compressed_data, T *uncompressed_data,
                                const int64_t *tile_offset, const int64_t *tile_size, int blocksize,
                                int64_t tiles, int64_t maxtilelen, double bscale, double bzero) {
  const int fsbits[3] = {3, 4, 5};
  const int fsmax[3] = {6, 14, 25};
  const int nonzero_count[256] = {
      0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int64_t tile = index; tile < tiles; tile += stride) {
    unsigned char *compressed_tile = compressed_data + tile_offset[tile];
    int i, imax, k;
    int nbits, nzero, fs;
    unsigned int b, diff, lastpix;
    int bbits, bytelog2;
    int64_t beg;

    bytelog2 = 0;
    for (int i = sizeof(T); i > 1; i >>= 1) {
      ++bytelog2;
    }
    bbits = 1 << fsbits[bytelog2];

    lastpix = read_lastpix<T>(compressed_tile);
    compressed_tile += sizeof(T);
    b = *compressed_tile++;

    beg = tile * maxtilelen;
    nbits = 8;
    for (i = 0; i < tile_size[tile];) {
      nbits -= fsbits[bytelog2];
      while (nbits < 0) {
        b = (b << 8) | (*compressed_tile++);
        nbits += 8;
      }
      fs = (b >> nbits) - 1;

      b &= (1 << nbits) - 1;
      imax = i + blocksize;
      if (imax > tile_size[tile]) {
        imax = tile_size[tile];
      }
      if (fs < 0) {
        for (; i < imax; i++) {
          uncompressed_data[beg + i] = lastpix;
        }
      } else if (fs == fsmax[bytelog2]) {
        for (; i < imax; i++) {
          k = bbits - nbits;
          diff = b << k;
          for (k -= 8; k >= 0; k -= 8) {
            b = *compressed_tile++;
            diff |= b << k;
          }
          if (nbits > 0) {
            b = *compressed_tile++;
            diff |= b >> (-k);
            b &= (1 << nbits) - 1;
          } else {
            b = 0;
          }

          if ((diff & 1) == 0) {
            diff = diff >> 1;
          } else {
            diff = ~(diff >> 1);
          }
          uncompressed_data[beg + i] = diff + lastpix;
          lastpix = uncompressed_data[beg + i];
        }
      } else {
        for (; i < imax; i++) {
          while (b == 0) {
            nbits += 8;
            b = *compressed_tile++;
          }
          nzero = nbits - nonzero_count[b];
          nbits -= nzero + 1;
          b ^= 1 << nbits;
          nbits -= fs;
          while (nbits < 0) {
            b = (b << 8) | (*compressed_tile++);
            nbits += 8;
          }
          diff = (nzero << fs) | (b >> nbits);
          b &= (1 << nbits) - 1;

          if ((diff & 1) == 0) {
            diff = diff >> 1;
          } else {
            diff = ~(diff >> 1);
          }
          uncompressed_data[beg + i] = diff + lastpix;
          lastpix = uncompressed_data[beg + i];
        }
      }
    }

    if (bscale != 1. || bzero != 0.) {
      for (int j = 0; j < tile_size[tile]; ++j) {
        uncompressed_data[beg + j] = static_cast<std::make_signed_t<T>>(
            static_cast<std::make_signed_t<T>>(uncompressed_data[beg + j]) * bscale + bzero);
      }
    }
  }
}

void FitsReaderGPU::RunImpl(Workspace &ws) {
  int num_outputs = ws.NumOutput();
  int batch_size = GetCurrBatchSize();
  TensorList<CPUBackend> sample_list_cpu;

  kernels::DynamicScratchpad s(ws.stream());

  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    auto &output = ws.Output<GPUBackend>(output_idx);
    auto &sample_0 = GetSample(0);
    bool compressed = sample_0.header[output_idx].compressed;

    sample_list_cpu.Reset();
    sample_list_cpu.set_sample_dim(sample_0.data[output_idx].ndim());
    sample_list_cpu.set_type(sample_0.data[output_idx].type());
    sample_list_cpu.set_device_id(sample_0.data[output_idx].device_id());
    sample_list_cpu.SetSize(batch_size);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      sample_list_cpu.SetSample(sample_id, sample.data[output_idx]);
    }

    if (compressed) {
      /* Temporary buffer must be used for doing H2D copy.
      Otherwise, if we copy  straight from a buffer that is pinned and
      uses host order the data might  be cobbled during H2D copy.
      */
      TensorList<GPUBackend> sample_list_gpu;
      TensorList<CPUBackend> samples_tmp;
      samples_tmp.SetContiguity(BatchContiguity::Contiguous);
      samples_tmp.set_order(ws.stream());
      sample_list_gpu.set_order(ws.stream());

      samples_tmp.Copy(sample_list_cpu);
      sample_list_gpu.Copy(samples_tmp, ws.stream());

      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        int64_t *tile_offset_cuda, *tile_size_cuda;
        auto &sample = GetSample(sample_id);
        auto header = sample.header[output_idx];
        int blockSize = 256, numBlocks = static_cast<int>(header.rows + blockSize - 1) / blockSize;

        std::tie(tile_offset_cuda, tile_size_cuda) = s.ToContiguousGPU(
            ws.stream(), sample.tile_offset[output_idx], sample.tile_size[output_idx]);

        if (sample.header[output_idx].bytepix == 1) {
          rice_decompress<uint8_t><<<numBlocks, blockSize, 0, ws.stream()>>>(
              static_cast<uint8_t *>(sample_list_gpu.raw_mutable_tensor(sample_id)),
              static_cast<uint8_t *>(output.raw_mutable_tensor(sample_id)), tile_offset_cuda,
              tile_size_cuda, sample.header[output_idx].blocksize, header.rows, header.maxtilelen,
              header.bscale, header.bzero);
          CUDA_CALL(cudaGetLastError());
        } else if (sample.header[output_idx].bytepix == 2) {
          rice_decompress<uint16_t><<<numBlocks, blockSize, 0, ws.stream()>>>(
              static_cast<uint8_t *>(sample_list_gpu.raw_mutable_tensor(sample_id)),
              static_cast<uint16_t *>(output.raw_mutable_tensor(sample_id)), tile_offset_cuda,
              tile_size_cuda, sample.header[output_idx].blocksize, header.rows, header.maxtilelen,
              header.bscale, header.bzero);
          CUDA_CALL(cudaGetLastError());
        } else {
          rice_decompress<uint32_t><<<numBlocks, blockSize, 0, ws.stream()>>>(
              static_cast<uint8_t *>(sample_list_gpu.raw_mutable_tensor(sample_id)),
              static_cast<uint32_t *>(output.raw_mutable_tensor(sample_id)), tile_offset_cuda,
              tile_size_cuda, sample.header[output_idx].blocksize, header.rows, header.maxtilelen,
              header.bscale, header.bzero);
          CUDA_CALL(cudaGetLastError());
        }
      }
    } else {
      output.Copy(sample_list_cpu, ws.stream());
    }
  }
}

DALI_REGISTER_OPERATOR(experimental__readers__Fits, FitsReaderGPU, GPU);

}  // namespace dali
