// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/fits_reader_gpu_op.h"

#include <string>
#include <vector>

namespace dali {

__global__ void rice_decompress(unsigned char *compressed_data, void *uncompressed_data,
                                const long *tile_offset, const long *tile_size, int bytepix,
                                int blocksize, long tiles, long maxtilelen) {
  int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(blockDim.x * gridDim.x);

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

  if (bytepix == 1) {
    for (long tile = index; tile < tiles; tile += stride) {
      unsigned char *compressed_tile = compressed_data + tile_offset[tile];
      int i, imax;
      int k;
      int nbits, nzero, fs;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      long beg;

      fsbits = 3;
      fsmax = 6;
      bbits = 1 << fsbits;

      lastpix = compressed_tile[0];
      compressed_tile += 1;
      b = *compressed_tile++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_tile++);
          nbits += 8;
        }
        fs = (int)(b >> nbits) - 1;

        b &= (1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile]) {
          imax = (int)tile_size[tile];
        }
        if (fs < 0) {
          for (; i < imax; i++) {
            ((unsigned char *)uncompressed_data)[beg + i] = lastpix;
          }
        } else if (fs == fsmax) {
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
            ((unsigned char *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned char *)uncompressed_data)[beg + i];
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
            ((unsigned char *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned char *)uncompressed_data)[beg + i];
          }
        }
      }
    }
  } else if (bytepix == 2) {
    for (long tile = index; tile < tiles; tile += stride) {
      unsigned char *compressed_tile = compressed_data + tile_offset[tile];
      int i, imax, k;
      int nbits, nzero, fs;
      unsigned char bytevalue;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      long beg;

      fsbits = 4;
      fsmax = 14;
      bbits = 1 << fsbits;

      lastpix = 0;
      bytevalue = compressed_tile[0];
      lastpix = lastpix | (bytevalue << 8);
      bytevalue = compressed_tile[1];
      lastpix = lastpix | bytevalue;

      compressed_tile += 2;
      b = *compressed_tile++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_tile++);
          nbits += 8;
        }
        fs = (int)(b >> nbits) - 1;

        b &= (int)(1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile]) {
          imax = (int)tile_size[tile];
        }
        if (fs < 0) {
          for (; i < imax; i++)
            ((unsigned short *)uncompressed_data)[beg + i] = lastpix;
        } else if (fs == fsmax) {
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
            ((unsigned short *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned short *)uncompressed_data)[beg + i];
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
              b = (b << 8) | *compressed_tile++;
              nbits += 8;
            }
            diff = (nzero << fs) | (b >> nbits);
            b &= (1 << nbits) - 1;

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned short *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned short *)uncompressed_data)[beg + i];
          }
        }
      }
    }
  } else {
    for (long tile = index; tile < tiles; tile += stride) {
      unsigned char *compressed_tile = compressed_data + tile_offset[tile];
      int i, imax, k;
      int nbits, nzero, fs;
      unsigned char bytevalue;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      long beg;

      fsbits = 5;
      fsmax = 25;
      bbits = 1 << fsbits;

      lastpix = 0;
      bytevalue = compressed_tile[0];
      lastpix = lastpix | (bytevalue << 24);
      bytevalue = compressed_tile[1];
      lastpix = lastpix | (bytevalue << 16);
      bytevalue = compressed_tile[2];
      lastpix = lastpix | (bytevalue << 8);
      bytevalue = compressed_tile[3];
      lastpix = lastpix | bytevalue;

      compressed_tile += 4;
      b = *compressed_tile++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_tile++);
          nbits += 8;
        }
        fs = (int)(b >> nbits) - 1;

        b &= (1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile]) {
          imax = (int)tile_size[tile];
        }
        if (fs < 0) {
          for (; i < imax; i++) {
            ((unsigned int *)uncompressed_data)[beg + i] = lastpix;
          }
        } else if (fs == fsmax) {
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
            ((unsigned int *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned int *)uncompressed_data)[beg + i];
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
            ((unsigned int *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned int *)uncompressed_data)[beg + i];
          }
        }
      }
    }
  }
}

void FitsReaderGPU::RunImpl(Workspace &ws) {
  int num_outputs = ws.NumOutput();
  int batch_size = GetCurrBatchSize();

  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    auto &output = ws.Output<GPUBackend>(output_idx);
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);

      if (sample.header[output_idx].compressed) {
        void *decoded_data_cuda;
        unsigned char *undecoded_data_cuda;
        long *tile_offset_cuda;
        long *tile_size_cuda;
        int64_t tiles = sample.header[output_idx].tiles,
                maxtilelen = sample.header[output_idx].maxtilelen,
                zbitpix = sample.header[output_idx].zbitpix;

        if (zbitpix == 8) {
          cudaMalloc(&decoded_data_cuda, tiles * maxtilelen * sizeof(char));
        } else if (zbitpix == 16) {
          cudaMalloc(&decoded_data_cuda, tiles * maxtilelen * sizeof(short));
        } else {
          cudaMalloc(&decoded_data_cuda, tiles * maxtilelen * sizeof(int));
        }

        cudaMalloc(&tile_offset_cuda, (tiles + 1) * sizeof(long));
        cudaMemcpy(tile_offset_cuda, sample.tile_offset[output_idx].data(),
                   (tiles + 1) * sizeof(long), cudaMemcpyHostToDevice);

        cudaMalloc(&tile_size_cuda, tiles * sizeof(long));
        cudaMemcpy(tile_size_cuda, sample.tile_size[output_idx].data(), tiles * sizeof(long),
                   cudaMemcpyHostToDevice);

        cudaMalloc(&undecoded_data_cuda,
                   sample.tile_offset[output_idx][tiles] * sizeof(unsigned char));
        cudaMemcpy(undecoded_data_cuda, sample.data[output_idx].raw_data(),
                   sample.tile_offset[output_idx][tiles] * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (int)(tiles + blockSize - 1) / blockSize;

        DALI_ENFORCE(false, "got here!");
        rice_decompress<<<numBlocks, blockSize>>>(
            undecoded_data_cuda, output.raw_mutable_tensor(sample_id), tile_offset_cuda,
            tile_size_cuda, sample.header[output_idx].bytepix, sample.header[output_idx].blocksize,
            tiles, maxtilelen);

        cudaFree(undecoded_data_cuda);
        cudaFree(tile_size_cuda);
        cudaFree(tile_offset_cuda);

      } else {
        cudaMemcpyAsync(output.raw_mutable_tensor(sample_id), sample.data[output_idx].raw_data(),
                        sample.data[output_idx].nbytes(), cudaMemcpyHostToDevice);
      }
    }
  }
}

DALI_REGISTER_OPERATOR(experimental__readers__Fits, FitsReaderGPU, GPU);

}  // namespace dali
