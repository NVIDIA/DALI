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

// Maximum number of channels nvjpeg decoder supports
#define NVJPEG_MAX_COMPONENT 4

#include "cuda_runtime_api.h"
#include "library_types.h"

#include "stdint.h"

#if defined(__cplusplus)
  extern "C" {
#endif

/* nvJPEG status enums, returned by nvJPEG API */
typedef enum
{
    NVJPEG_STATUS_SUCCESS            = 0,
    NVJPEG_STATUS_NOT_INITIALIZED    = 1,
    NVJPEG_STATUS_INVALID_PARAMETER  = 2,
    NVJPEG_STATUS_BAD_JPEG           = 3,
    NVJPEG_STATUS_JPEG_NOT_SUPPORTED = 4,
    NVJPEG_STATUS_ALLOCATOR_FAILURE  = 5,
    NVJPEG_STATUS_EXECUTION_FAILED   = 6,
    NVJPEG_STATUS_ARCH_MISMATCH      = 7,
    NVJPEG_STATUS_INTERNAL_ERROR     = 8,
} nvjpegStatus_t;


// Enumeration returned by getImageInfo identifies image chroma subsampling stored inside JPEG input stream
// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
// Initial release support: 4:4:4, 4:2:0, 4:2:2, Grayscale
typedef enum
{
    NVJPEG_CSS_444 = 0,
    NVJPEG_CSS_422 = 1,
    NVJPEG_CSS_420 = 2,
    NVJPEG_CSS_440 = 3,
    NVJPEG_CSS_411 = 4,
    NVJPEG_CSS_410 = 5,
    NVJPEG_CSS_GRAY = 6,
    NVJPEG_CSS_UNKNOWN = -1
} nvjpegChromaSubsampling_t;

// Parameter of this type specifies what type of output user wants for image decoding
typedef enum
{
    NVJPEG_OUTPUT_UNCHANGED   = 0, // return decoded image as it is - write planar output
    NVJPEG_OUTPUT_YUV         = 1, // return planar luma and chroma
    NVJPEG_OUTPUT_Y           = 2, // return luma component only, write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_RGB         = 4, // convert to planar RGB
    NVJPEG_OUTPUT_BGR         = 5, // convert to planar BGR
    NVJPEG_OUTPUT_RGBI        = 6, // convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_BGRI        = 7  // convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
} nvjpegOutputFormat_t;

// Implementation
// Initial release support: NVJPEG_BACKEND_DEFAULT, NVJPEG_BACKEND_HYBRID
typedef enum 
{
    NVJPEG_BACKEND_DEFAULT = 0,
    NVJPEG_BACKEND_HYBRID  = 1,
    NVJPEG_BACKEND_GPU     = 2,
} nvjpegBackend_t;

// Output descriptor.
// Data that is written to planes depends on output forman
typedef struct
{
    unsigned char * channel[NVJPEG_MAX_COMPONENT];
    unsigned int    pitch[NVJPEG_MAX_COMPONENT];
} nvjpegImage_t;

// Prototype for device memory allocation. 
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);

// Memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct 
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator_t;

// Opaque library handle identifier.
struct nvjpegHandle;
typedef struct nvjpegHandle* nvjpegHandle_t;

// Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
struct nvjpegJpegState;
typedef struct nvjpegJpegState* nvjpegJpegState_t;

// returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
nvjpegStatus_t NVJPEGAPI nvjpegGetProperty(libraryPropertyType type, int *value);

// Initalization of nvjpeg handle. This handle is used for all consecutive calls
// IN         backend       : Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.
// IN         allocator     : Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)
// INT/OUT    handle        : Codec instance, use for other calls
nvjpegStatus_t NVJPEGAPI nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t *allocator, nvjpegHandle_t *handle);

// Release the handle and resources.
// IN/OUT     handle: instance handle to release 
nvjpegStatus_t NVJPEGAPI nvjpegDestroy(nvjpegHandle_t handle);


// Initalization of decoding state
// IN         handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
nvjpegStatus_t NVJPEGAPI nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t *jpeg_handle);

// Release the jpeg image handle.
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
nvjpegStatus_t NVJPEGAPI nvjpegJpegStateDestroy(nvjpegJpegState_t jpeg_handle);
// 
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded. 
// IN         length      : Length of the jpeg image buffer.
// OUT        nComponent  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// OUT        subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling_t
// OUT        widths      : pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  
// OUT        heights     : pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded 
nvjpegStatus_t NVJPEGAPI nvjpegGetImageInfo(
          nvjpegHandle_t handle,
          const unsigned char *data, 
          size_t length,
          int *nComponents, 
          nvjpegChromaSubsampling_t *subsampling,
          int *widths,
          int *heights);
                   

// Decodes single image. Destination buffers should be large enough to be able to store 
// output of specified format. For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Pointer to the buffer containing the jpeg image to be decoded. 
// IN         length        : Length of the jpeg image buffer.
// IN         output_format : Output data format. See nvjpegOutputFormat_t for description
// IN/OUT     destination   : Pointer to structure with information about output buffers. See nvjpegImage_t description.
// IN/OUT     stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecode(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *data,
          size_t length, 
          nvjpegOutputFormat_t output_format,
          nvjpegImage_t *destination,
          cudaStream_t stream);

// Same functionality and parameters as for nvjpegDecodePlanar, but separated in steps: 
// 1) CPU processing
// 2) Mixed processing that requires interaction of both GPU and CPU. Any previous call 
// to nvjpegDecodeGPU() with same handle should be finished before this call, i.e. cudaStreamSycnhronize() could be used
// 3) GPU processing 
// Actual amount of work done in each separate step depends on the selected backend. But in any way all 
// of those functions must be called in this specific order. If one of the steps returns error - decode should be done from the beginning.
nvjpegStatus_t NVJPEGAPI nvjpegDecodePhaseOne(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *data,
          size_t length,
          nvjpegOutputFormat_t output_format,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodePhaseTwo(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodePhaseThree(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          nvjpegImage_t *destination,
          cudaStream_t stream);

//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

// Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT     handle          : Library handle
// INT/OUT    jpeg_handle     : Decoded jpeg image state handle
// IN         batch_size      : Size of the batch
// IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN         output_format   : Output data format. Will be the same for every image in batch
//
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedInitialize(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          int batch_size,
          int max_cpu_threads,
          nvjpegOutputFormat_t output_format);

// Decodes batch of images. Output buffers should be large enough to be able to store 
// outputs of specified format, see single image decoding description for details. Call to 
// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
// parameter to this batch initialization function.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
// IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
// IN/OUT     stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatched(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *const *data,
          const size_t *lengths, 
          nvjpegImage_t *destinations,
          cudaStream_t stream);

// Same functionality as nvjpegDecodePlanarBatched but done in separate consecutive steps: 
// 1) nvjpegDecodePlanarBatchedCPU should be called [batch_size] times for each image in batch.
// This function is thread safe and could be called by multiple threads simultaneously, by providing 
// thread_idx (thread_idx should be less than max_cpu_threads from nvjpegDecodeBatchedInitialize())
// 2) nvjpegDecodePlanarBatchedMixed. Any previous call to nvjpegDecodeBatchedGPU() should be done by this point
// 3) nvjpegDecodePlanarBatchedGPU 
// Actual amount of work done in each separate step depends on the selected backend. But in any way all 
// of those functions must be called in this specific order. If one of the steps returns error - 
// reset batch with nvjpegDecodeBatchedInitialize(). 
nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedPhaseOne(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          const unsigned char *data,
          size_t length,
          int image_idx,
          int thread_idx,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedPhaseTwo(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          cudaStream_t stream);

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedPhaseThree(
          nvjpegHandle_t handle,
          nvjpegJpegState_t jpeg_handle,
          nvjpegImage_t *destinations,
          cudaStream_t stream);

#if defined(__cplusplus)
  }
#endif
 
#endif /* NV_JPEG_HEADER */
