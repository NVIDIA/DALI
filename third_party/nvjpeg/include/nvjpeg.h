 /* Copyright 2009-2018 NVIDIA Corporation.  All rights reserved. 
 // 
 // NOTICE TO LICENSEE: 
 // 
 // The source code and/or documentation ("Licensed Deliverables") are 
 // subject to NVIDIA intellectual property rights under U.S. and 
 // international Copyright laws. 
 // 
 // The Licensed Deliverables contained herein are PROPRIETARY and 
 // CONFIDENTIAL to NVIDIA and are being provided under the terms and 
 // conditions of a form of NVIDIA software license agreement by and 
 // between NVIDIA and Licensee ("License Agreement") or electronically 
 // accepted by Licensee.  Notwithstanding any terms or conditions to 
 // the contrary in the License Agreement, reproduction or disclosure 
 // of the Licensed Deliverables to any third party without the express 
 // written consent of NVIDIA is prohibited. 
 // 
 // NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
 // LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
 // SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
 // PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
 // NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
 // DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
 // NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
 // NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
 // LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
 // SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
 // DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
 // WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
 // ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
 // OF THESE LICENSED DELIVERABLES. 
 // 
 // U.S. Government End Users.  These Licensed Deliverables are a 
 // "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
 // 1995), consisting of "commercial computer software" and "commercial 
 // computer software documentation" as such terms are used in 48 
 // C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
 // only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
 // 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
 // U.S. Government End Users acquire the Licensed Deliverables with 
 // only those rights set forth herein. 
 // 
 // Any use of the Licensed Deliverables in individual and commercial 
 // software must include, in the user documentation and internal 
 // comments to the code, the above Disclaimer and U.S. Government End 
 // Users Notice. 
  */ 
  
#ifndef NV_JPEG_HEADER
#define NV_JPEG_HEADER

#define NVJPEGAPI

#include "cuda_runtime.h"

#if defined(__cplusplus)
  extern "C" {
#endif

/* nvJPEG status enums, returned by nvJPEG API */
typedef enum
{
    NVJPEG_STATUS_SUCCESS            = 0,
    NVJPEG_STATUS_EXECUTION_FAILED   = 1,
    NVJPEG_STATUS_NOT_INITIALIZED    = 2,
    NVJPEG_STATUS_ALLOC_FAILED       = 3,
    NVJPEG_STATUS_INVALID_VALUE      = 4,
    NVJPEG_STATUS_ARCH_MISMATCH      = 5,
    NVJPEG_STATUS_INTERNAL_ERROR     = 6,
    NVJPEG_STATUS_NOT_SUPPORTED      = 7,
} nvjpegStatus_t;


// Enumeration returned by getImageInfo identifies image format stored inside JPEG input stream
// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
typedef enum
{
    // Initial release support: 4:4:4, 4:2:0, 4:2:2, so first priority
    NVJPEG_CSS_444,
    NVJPEG_CSS_422,
    NVJPEG_CSS_420,
    NVJPEG_CSS_440,
    NVJPEG_CSS_411,
    NVJPEG_CSS_410,
    NVJPEG_CSS_GRAY,
    NVJPEG_CSS_UNKNOWN
} nvjpegChromaSubsampling;

// Parameter of this type specifies what type of output user wants for image decoding
// Final support of this feature in initial release is under question:
// NVJPEG_OUTPUT_UNCHANGED and NVJPEG_OUTPUT_Y will be supported
// NVJPEG_OUTPUT_RGB and NVJPEG_OUTPUT_BGR are under investigation right now
typedef enum
{
    NVJPEG_OUTPUT_UNCHANGED,   // return decoded image as it is - planar luminance and chrominance (if exists)
    NVJPEG_OUTPUT_YUV,         // return planar luma and chroma (basically same as NVJPEG_OF_UNCHANGED)
    NVJPEG_OUTPUT_Y,           // return Y component only
    NVJPEG_OUTPUT_RGB,         // convert to planar RGB
    NVJPEG_OUTPUT_BGR,         // convert to planar BGR
    NVJPEG_OUTPUT_RGBI,         // convert to interleaved RGB
    NVJPEG_OUTPUT_BGRI,         // convert to interleaved BGR
} nvjpegOutputFormat;

// Implementation
// only hybrid backend (default) is supported right now
typedef enum 
{
    NVJPEG_BACKEND_DEFAULT = 0,
    NVJPEG_BACKEND_HYBRID,
    NVJPEG_BACKEND_GPU,
    NVJPEG_BACKEND_CPU,
} nvjpegBackend;

// Output buffers descriptor. Used for planar output only
// Data written to planes depends on output forman, i.e. RGB or YUV
typedef struct
{
    unsigned char *p1;
    int pitch1;
    unsigned char *p2;
    int pitch2;
    unsigned char *p3;
    int pitch3;
} nvjpegImageOutputPlanar;

// Output buffers descriptor. Used for interleaved output. 
// Applicable only for RGBi/BGRi output (NVJPEG_OUTPUT_RGBI/NVJPEG_OUTPUT_BGRI)
typedef struct
{
    unsigned char *ptr;
    int pitch;
} nvjpegImageOutputInterleaved;


// Prototype for device memory allocation. 
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);

// Memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for device memory allocations inside library
typedef struct 
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator;

// Opaque library handle identifier
struct nvjpegHandle;
typedef struct nvjpegHandle* nvjpegHandle_t;

// Initalization of nvjpeg handle. This handle is used for all consecutive 
// IN     backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN     allocator     : ptr to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree), but those allocations are persistent.
// OUT    handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreate(nvjpegBackend backend, nvjpegDevAllocator *allocator, nvjpegHandle_t *handle);

// Release the handle and resources.
// IN     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegDestroy(nvjpegHandle_t handle);

// 
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If the image is 1-channel, only widthY and heightY are valid. The other two groups
// are set to 0.
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN/OUT handle      : Library handle
// IN     data        : Pointer to the buffer containing the jpeg image to be decoded. 
// IN     length      : Length of the jpeg image buffer.
// OUT    nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// OUT    nWidthY     : Width of Y component.
// OUT    nHeightY    : Height to Y component.
// OUT    nWidthCb    : Width ofCbY component.
// OUT    nHeightCb   : Height to Cb component.
// OUT    nWidthCr    : Width ofCrY component.
// OUT    nHeightCr   : Height to Cr component.
// OUT    subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling
nvjpegStatus_t NVJPEGAPI nvjpegGetImageInfo(
          nvjpegHandle_t handle,
          const unsigned char *data, 
          unsigned int length,
          int *nComponents, 
          int *nWidthY,  
          int *nHeightY, 
          int *nWidthCb, 
          int *nHeightCb,  
          int *nWidthCr, 
          int *nHeightCr,
          nvjpegChromaSubsampling *subsampling);
                   

// Decodes single image, output in planar form. Output buffers should be large enough to be able to store 
// output of specified format. For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
// and minimum required memory buffer is nPlaneHeight*nPlanePitch
// 
// IN/OUT handle        : Library handle
// IN     data          : Pointer to the buffer containing the jpeg image to be decoded. 
// IN     length        : Length of the jpeg image buffer.
// IN     output_format : Output data format. See nvjpegOutputFormat for description
// IN/OUT destination   : Pointer to structure with information about output buffers, 
//                        see nvjpegImageOutputPlanar and nvjpegImageOutputInterleaved for description
// IN/OUT stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecode(
          nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length, 
          nvjpegOutputFormat output_format,
          void *destination,
          cudaStream_t stream);

// Same functionality and parameters as for nvjpegDecodePlanar, but separated in steps: 
// 1) CPU processing
// 2) Mixed processing that requires interaction of both GPU and CPU. Any previous call 
// to nvjpegDecodeGPU() with same handle should be finished before this call, i.e. cudaStreamSycnhronize() could be used
// 3) GPU processing 
// Actual amount of work done in each separate step depends on the selected backend. But in any way all 
// of those functions must be called in this specific order. If one of the steps returns error - decode should be done from the beginning.
nvjpegStatus_t NVJPEGAPI nvjpegDecodeCPU(
          nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length,
          nvjpegOutputFormat output_format,
          void *destination,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeMixed(
          nvjpegHandle_t handle,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeGPU(
          nvjpegHandle_t handle,
          cudaStream_t stream);

// Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT handle          : Library handle
// IN     batch_size      : Size of the batch
// IN     max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN     output_format   : Output data format. Will be the same for every image in batch
//
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedInitialize(
          nvjpegHandle_t handle,
          int batch_size,
          int max_cpu_threads,
          nvjpegOutputFormat output_format);

// Decodes batch of images, output in planar form. Output buffers should be large enough to be able to store 
// outputs of specified format. For each color plane of each image sizes could be retrieved for image using 
// nvjpegGetImageInfo() and minimum required memory buffer is nPlaneHeight*nPlanePitch. Call to 
// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
// parameter to this batch initialization function.
// 
// IN/OUT handle        : Library handle
// IN     data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
// IN     lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
//                        see nvjpegImageOutputPlanar and nvjpegImageOutputInterleaved for description
// IN/OUT stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatched(
          nvjpegHandle_t handle,
          const unsigned char *const *data,
          const unsigned int *lengths, 
          void *destinations,
          cudaStream_t stream);

// Same functionality as nvjpegDecodePlanarBatched but done in separate consecutive steps: 
// 1) nvjpegDecodePlanarBatchedCPU should be called [batch_size] times for each image in batch.
// Is thread safe and could be called by multiple threads simultaneously, by providing 
// thread_idx (thread_idx should be less than max_cpu_threads from nvjpegDecodeBatchedInitialize()
// 2) nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point
// 3) nvjpegDecodePlanarBatchedGPU 
// Actual amount of work done in each separate step depends on the selected backend. But in any way all 
// of those functions must be called in this specific order. If one of the steps returns error - 
// reset batch with nvjpegDecodeBatchedInitialize(). 
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedCPU(
          nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length,
          int image_idx,
          int thread_idx,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedMixed(
          nvjpegHandle_t handle,
          void *destinations,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedGPU(
          nvjpegHandle_t handle,
          cudaStream_t stream);

#if defined(__cplusplus)
  }
#endif
 
#endif /* NV_JPEG_HEADER */
