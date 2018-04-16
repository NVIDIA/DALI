 /* Copyright 2009-2018 NVIDIA Corporation.  All rights reserved. 
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to NVIDIA intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and 
  * conditions of a form of NVIDIA software license agreement by and 
  * between NVIDIA and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of NVIDIA is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
  * OF THESE LICENSED DELIVERABLES. 
  * 
  * U.S. Government End Users.  These Licensed Deliverables are a 
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
  * 1995), consisting of "commercial computer software" and "commercial 
  * computer software documentation" as such terms are used in 48 
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
  * U.S. Government End Users acquire the Licensed Deliverables with 
  * only those rights set forth herein. 
  * 
  * Any use of the Licensed Deliverables in individual and commercial 
  * software must include, in the user documentation and internal 
  * comments to the code, the above Disclaimer and U.S. Government End 
  * Users Notice. 
  */ 
  
#ifndef NV_JPEG_HEADER
#define NV_JPEG_HEADER

#define NVJPEGAPI

#include "cuda_runtime.h"

#if defined(__cplusplus)
  extern "C" {
#endif

/* nvJPEG status type returns */
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


// returned by getImageInfo
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

typedef enum
{
    NVJPEG_OUTPUT_UNCHANGED,   // return decoded image as it is, luma and chroma planes
    NVJPEG_OUTPUT_RGB,         // convert to planar RGB
    NVJPEG_OUTPUT_BGR,         // convert to planar BGR
    NVJPEG_OUTPUT_Y,           // return Y component only
    NVJPEG_OUTPUT_YUV,         // return luma and chroma (basically same as NVJPEG_OF_UNCHANGED)
} nvjpegOutputFormat;

// only hybrid backend (default) is supported right now
typedef enum 
{
    NVJPEG_BACKEND_DEFAULT = 0,
    NVJPEG_BACKEND_HYBRID,
    NVJPEG_BACKEND_GPU,
    NVJPEG_BACKEND_CPU,
} nvjpegBackend;

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


typedef int (*tDevMalloc)(void**, size_t);
typedef int (*tDevFree)(void*);
 
typedef struct 
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator;

struct nvjpegHandle;
typedef struct nvjpegHandle* nvjpegHandle_t;

// IN : backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN : allocator     : ptr to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree), but those allocations are persistent.
// OUT: handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreate(nvjpegBackend backend, nvjpegDevAllocator *allocator, nvjpegHandle_t *handle);

// IN : handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegDestroy(nvjpegHandle_t handle);
/** 
 * Retrieve the image info, including channel, width and height of each component.
 * If the image is 1-channel, only widthY and heightY are valid. The other two groups
 * are set to 0.
 * If the image is 3-channel, all three groups are valid.
 * The user should call this function to allocate the appropriate buffer
 * before calling the decoder.
 * 
 * \param data          Pointer to the buffer containing the jpeg image to be decoded. 
 * \param length         Length of the jpeg image buffer.
 * \param nComponent    Number of componenets of the image, currently only supports 1-channel or 3-channel.
 * \param nWidthY          Width of Y component.
 * \param nHeightY      Height to Y component.
 * \param nWidthCb      Width ofCbY component.
 * \param nHeightCb     Height to Cb component.
 * \param nWidthCr      Width ofCrY component.
 * \param nHeightCr     Height to Cr component.
 * \param subsampling   Chroma subsampling used in this JPEG
 *
 * \return 
 */ 
nvjpegStatus_t NVJPEGAPI nvjpegGetImageInfo(nvjpegHandle_t handle,
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
                   
/** 
 * Decoder path for a single image, output in planar form
 * Before calling this function, user needs to call the \ref nvjpegGetImageInfo
 * to determine the component and size information of the image and allocate the 
 * buffer accordingly. 
 * 
 * \param data          Pointer to the buffer containing the jpeg image to be decoded. 
 * \param length        Length of the jpeg image buffer.
 * \param destination   Structure with information about output buffers
 * \param output_format Output data format. See nvjpegOutputFormat
 * \param stream        CUDA stream where do all GPU work
 * 
 * \return NVJPEG_STATUS_SUCCESS if successful
 */
nvjpegStatus_t NVJPEGAPI nvjpegDecodePlanar(nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length, 
          nvjpegImageOutputPlanar destination,
          nvjpegOutputFormat output_format,
          cudaStream_t stream);

// experimental API with access to individual decoding phases. 
// Should be called one after another, if any returns error code - something is wrong (debug to cerr now)
nvjpegStatus_t NVJPEGAPI nvjpegDecodePlanarCPU(nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length);

nvjpegStatus_t NVJPEGAPI nvjpegDecodePlanarMemcpy(nvjpegHandle_t handle,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodePlanarGPU(nvjpegHandle_t handle,
          nvjpegImageOutputPlanar destination,
          nvjpegOutputFormat output_format,
          cudaStream_t stream);

// same as nvjpegDecodePlanar but instead of using multiple output planes puts 
// everything in single continuous output buffer.
// Useful when JPEG has chroma subsampling 4:4:4 or we want to convert to RGB
nvjpegStatus_t NVJPEGAPI nvjpegDecodeContinuous(nvjpegHandle_t handle,
          const unsigned char *data,
          unsigned int length, 
          unsigned char *destination,  
          nvjpegOutputFormat output_format,
          cudaStream_t stream);

/*@}*/

#if defined(__cplusplus)
  }
#endif
 
#endif /* NV_JPEG_HEADER */
