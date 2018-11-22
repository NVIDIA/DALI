#include "dali/pipeline/operators/reader/nvdecoder/imgproc.h"

#include <cuda_fp16.h>

namespace dali {

namespace {

// using math from https://msdn.microsoft.com/en-us/library/windows/desktop/dd206750(v=vs.85).aspx

template<typename T>
struct yuv {
    T y, u, v;
};

// https://docs.microsoft.com/en-gb/windows/desktop/medfound/recommended-8-bit-yuv-formats-for-video-rendering#converting-8-bit-yuv-to-rgb888
__constant__ float yuv2rgb_mat_norm[9] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

// not normalized need *255
__constant__ float yuv2rgb_mat[9] = {
    1.164383f * 255.f,  0.0f,       1.596027f * 255.f,
    1.164383f * 255.f, -0.391762f * 255.f, -0.812968f * 255.f,
    1.164383f * 255.f,  2.017232f * 255.f,  0.0f
};

__device__ float clip(float x, float max) {
    return fminf(fmaxf(x, 0.0f), max);
}

template<typename T>
__device__ T convert(const float x) {
    return static_cast<T>(x);
}

template<>
__device__ half convert<half>(const float x) {
    return __float2half(x);
}

template<>
__device__ uint8_t convert<uint8_t>(const float x) {
    return static_cast<uint8_t>(roundf(x));
}

template<typename YUV_T, typename RGB_T, bool Normalized = false>
__device__ void yuv2rgb(const yuv<YUV_T>& yuv, RGB_T* rgb,
                        size_t stride) {
    auto y = (static_cast<float>(yuv.y) - 16.0f/255);
    auto u = (static_cast<float>(yuv.u) - 128.0f/255);
    auto v = (static_cast<float>(yuv.v) - 128.0f/255);


    float r, g, b;
    if (Normalized) {
        auto& m = yuv2rgb_mat_norm;
        r = clip(y*m[0] + u*m[1] + v*m[2], 1.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 1.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 1.0);
    } else {
        auto& m = yuv2rgb_mat;
        r = clip(y*m[0] + u*m[1] + v*m[2], 255.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 255.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 255.0);
    }

    rgb[0] = convert<RGB_T>(r);
    rgb[stride] = convert<RGB_T>(g);
    rgb[stride*2] = convert<RGB_T>(b);
}

template<typename T>
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
    src_x = static_cast<float>(dst_x) * fx;
    auto src_y = static_cast<float>(dst_y) * fy;


    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tex2d-object
    yuv<float> yuv;
    yuv.y = tex2D<float>(luma, src_x + 0.5, src_y + 0.5);
    auto uv = tex2D<float2>(chroma, (src_x / 2) + 0.5, (src_y / 2) + 0.5);
    yuv.u = uv.x;
    yuv.v = uv.y;

    auto* out = &dst[(dst_x + dst_y * dst_width) * c];

    size_t stride = 1;
    // TODO(spanev) Handle normalized version
    yuv2rgb<float, float, false>(yuv, out, stride);
}

inline constexpr int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

} //  namespace

template<typename T>
void process_frame(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    SequenceWrapper& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height) {
    auto scale_width = input_width;
    auto scale_height = input_height;

    auto fx = static_cast<float>(input_width) / scale_width;
    auto fy = static_cast<float>(input_height) / scale_height;

    dim3 block(32, 8);
    dim3 grid(divUp(output.width, block.x), divUp(output.height, block.y));

    int frame_stride = index * output.height * output.width * output.channels;
    LOG_LINE << "Processing frame " << index << " (frame_stride=" << frame_stride << ")" << std::endl;
    auto* tensor_out = output.sequence.mutable_data<T>() + frame_stride;

    process_frame_kernel<<<grid, block, 0, stream>>>
            (luma, chroma, tensor_out, index, fx, fy, output.width, output.height, output.channels);
}

template
void process_frame<float>(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    SequenceWrapper& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height);

}  // namespace dali