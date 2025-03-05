// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/video/legacy/reader/nvdecoder/imgproc.h"

#include <cuda_fp16.h>

namespace dali {

namespace {

// using math from https://msdn.microsoft.com/en-us/library/windows/desktop/dd206750(v=vs.85).aspx

template<typename T>
struct YCbCr {
  T y, cb, cr;
};

// https://docs.microsoft.com/en-gb/windows/desktop/medfound/recommended-8-bit-yuv-formats-for-video-rendering#converting-8-bit-yuv-to-rgb888
__constant__ float ycbcr2rgb_mat_norm[9] = {
  1.164383f,  0.0f,       1.596027f,
  1.164383f, -0.391762f, -0.812968f,
  1.164383f,  2.017232f,  0.0f
};

// not normalized need *255
__constant__ float ycbcr2rgb_mat[9] = {
  1.164383f * 255.0f,  0.0f,       1.596027f * 255.0f,
  1.164383f * 255.0f, -0.391762f * 255.0f, -0.812968f * 255.0f,
  1.164383f * 255.0f,  2.017232f * 255.0f,  0.0f
};


// https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV444_to_RGB888_conversion
__constant__ float ycbcr2rgb_mat_norm_full_range[9] = {
  1,  0.0f,          1.402f,
  1, -0.344136285f, -0.714136285f,
  1,  1.772f,        0.0f
};

// not normalized need *255
__constant__ float ycbcr2rgb_mat_full_range[9] = {
  1 * 255,  0.0f,                1.402f * 255,
  1 * 255, -0.344136285f * 255, -0.714136285f * 255,
  1 * 255,  1.772f * 255,        0.0f
};

__device__ float clip(float x, float max) {
  return fminf(fmaxf(x, 0.0f), max);
}

template<typename T>
__device__ T convert(const float x) {
  return static_cast<T>(x);
}

#if 0
template<>
__device__ half convert<half>(const float x) {
  return __float2half(x);
}

template<>
__device__ uint8_t convert<uint8_t>(const float x) {
  return static_cast<uint8_t>(roundf(x));
}
#endif

template<typename YCbCr_T, typename RGB_T, bool Normalized = false>
__device__ void ycbcr2rgb(const YCbCr<YCbCr_T>& ycbcr, RGB_T* rgb,
                        size_t stride) {
  auto y = (static_cast<float>(ycbcr.y) - 16.0f/255.0f);
  auto cb = (static_cast<float>(ycbcr.cb) - 128.0f/255.0f);
  auto cr = (static_cast<float>(ycbcr.cr) - 128.0f/255.0f);


  float r, g, b;
  if (Normalized) {
    auto& m = ycbcr2rgb_mat_norm;
    r = clip(y*m[0] + cb*m[1] + cr*m[2], 1.0f);
    g = clip(y*m[3] + cb*m[4] + cr*m[5], 1.0f);
    b = clip(y*m[6] + cb*m[7] + cr*m[8], 1.0f);
  } else {
    auto& m = ycbcr2rgb_mat;
    r = clip(y*m[0] + cb*m[1] + cr*m[2], 255.0f);
    g = clip(y*m[3] + cb*m[4] + cr*m[5], 255.0f);
    b = clip(y*m[6] + cb*m[7] + cr*m[8], 255.0f);
  }

  rgb[0] = convert<RGB_T>(r);
  rgb[stride] = convert<RGB_T>(g);
  rgb[stride*2] = convert<RGB_T>(b);
}


template<typename YCbCr_T, typename RGB_T, bool Normalized = false>
__device__ void ycbcr2rgb_full_range(const YCbCr<YCbCr_T>& ycbcr, RGB_T* rgb,
                        size_t stride) {
  auto y = (static_cast<float>(ycbcr.y));
  auto cb = (static_cast<float>(ycbcr.cb) - 128.0f/255.0f);
  auto cr = (static_cast<float>(ycbcr.cr) - 128.0f/255.0f);


  float r, g, b;
  if (Normalized) {
    auto& m = ycbcr2rgb_mat_norm_full_range;
    r = clip(y*m[0] + cb*m[1] + cr*m[2], 1.0f);
    g = clip(y*m[3] + cb*m[4] + cr*m[5], 1.0f);
    b = clip(y*m[6] + cb*m[7] + cr*m[8], 1.0f);
  } else {
    auto& m = ycbcr2rgb_mat_full_range;
    r = clip(y*m[0] + cb*m[1] + cr*m[2], 255.0f);
    g = clip(y*m[3] + cb*m[4] + cr*m[5], 255.0f);
    b = clip(y*m[6] + cb*m[7] + cr*m[8], 255.0f);
  }

  rgb[0] = convert<RGB_T>(r);
  rgb[stride] = convert<RGB_T>(g);
  rgb[stride*2] = convert<RGB_T>(b);
}

template<typename T, bool Normalized = false, bool RGB = true, bool FullRange = false>
__global__ void process_frame_kernel(
  cudaTextureObject_t luma, cudaTextureObject_t chroma,
  T* dst, int index,
  float fx, float fy,
  int dst_width, int dst_height, int c) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_width || dst_y >= dst_height)
      return;

  auto src_x = 0.0f;
  // TODO(spanev) something less hacky here, why 4:2:0 fails on this edge?
  float shift = (dst_x == dst_width - 1) ? 0 : 0.5f;
  src_x = static_cast<float>(dst_x) * fx + shift;
  auto src_y = static_cast<float>(dst_y) * fy + shift;

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tex2d-object
  YCbCr<float> ycbcr;
  ycbcr.y = tex2D<float>(luma, src_x, src_y);
  auto cbcr = tex2D<float2>(chroma, src_x * 0.5f, src_y * 0.5f);
  ycbcr.cb = cbcr.x;
  ycbcr.cr = cbcr.y;

  auto* out = &dst[(dst_x + dst_y * dst_width) * c];

  constexpr size_t stride = 1;
  if (RGB) {
    if (FullRange) {
      ycbcr2rgb_full_range<float, T, Normalized>(ycbcr, out, stride);
    } else {
      ycbcr2rgb<float, T, Normalized>(ycbcr, out, stride);
    }
  } else {
    constexpr float scaling = Normalized ? 1.0f : 255.0f;
    out[0] = convert<T>(ycbcr.y * scaling);
    out[stride] = convert<T>(ycbcr.cb * scaling);
    out[stride*2] = convert<T>(ycbcr.cr * scaling);
  }
}

inline constexpr int divUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

}  // namespace

template<typename T>
void process_frame(
  cudaTextureObject_t chroma, cudaTextureObject_t luma,
  SequenceWrapper& output, int index, cudaStream_t stream,
  uint16_t input_width, uint16_t input_height,
  bool rgb, bool normalized, bool full_range) {
  auto scale_width = input_width;
  auto scale_height = input_height;

  auto fx = static_cast<float>(input_width) / scale_width;
  auto fy = static_cast<float>(input_height) / scale_height;

  dim3 block(32, 8);
  dim3 grid(divUp(output.width, block.x), divUp(output.height, block.y));

  auto frame_stride =
          static_cast<ptrdiff_t>(index) * output.height * output.width * output.channels;
  LOG_LINE << "Processing frame " << index
            << " (frame_stride=" << frame_stride << ")" << std::endl;
  auto* tensor_out = output.sequence.mutable_data<T>() + frame_stride;

  if (normalized) {
    if (rgb) {
      if (full_range) {
        process_frame_kernel<T, true, true, true><<<grid, block, 0, stream>>>
            (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
      } else {
        process_frame_kernel<T, true, true, false><<<grid, block, 0, stream>>>
            (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
      }
    } else {
      process_frame_kernel<T, true, false><<<grid, block, 0, stream>>>
          (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
    }
  } else {
    if (rgb) {
      if (full_range) {
        process_frame_kernel<T, false, true, true><<<grid, block, 0, stream>>>
            (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
      } else {
         process_frame_kernel<T, false, true, false><<<grid, block, 0, stream>>>
          (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
      }
    } else {
      process_frame_kernel<T, false, false><<<grid, block, 0, stream>>>
          (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
    }
  }
}

template
void process_frame<float>(
  cudaTextureObject_t chroma, cudaTextureObject_t luma,
  SequenceWrapper& output, int index, cudaStream_t stream,
  uint16_t input_width, uint16_t input_height,
  bool rgb, bool normalized, bool full_range);

template
void process_frame<uint8_t>(
  cudaTextureObject_t chroma, cudaTextureObject_t luma,
  SequenceWrapper& output, int index, cudaStream_t stream,
  uint16_t input_width, uint16_t input_height,
  bool rgb, bool normalized, bool full_range);

}  // namespace dali
