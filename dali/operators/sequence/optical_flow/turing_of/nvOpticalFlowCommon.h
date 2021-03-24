/*
* Copyright(c) 2020, NVIDIA CORPORATION.All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met :
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and / or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
* \file nvOpticalFlowCommon.h
*   NVIDIA GPUs - Turing and above contains a hardware-based optical flow engine
*   which provides fully-accelerated hardware-based optical flow and stereo estimation.
*   nvOpticalFlowCommon.h provides enums, structure definitions and function prototypes which are common across different devices,
*   nvOpticalFlowCommon.h uses #pragma directives to pack structure members with one byte alignment.
* \date 2018
*  nvOpticalFlowCommon.h provides common enums, structure definitions and function prototypes.
*/

#ifndef _NV_OPTICALFLOW_COMMON_H_
#define _NV_OPTICALFLOW_COMMON_H_
#if defined(_MSC_VER_) && (_MSC_VER_ < 1600)
#ifndef _STDINT
typedef __int32             int32_t;
typedef unsigned __int32    uint32_t;
typedef __int64             int64_t;
typedef unsigned __int64    uint64_t;
typedef signed char         int8_t;
typedef unsigned char       uint8_t;
typedef short               int16_t;
typedef unsigned short      uint16_t;
#endif
#else
#include <stdint.h>
#endif

#ifdef _WIN32
#define NVOFAPI __stdcall
#else
#define NVOFAPI
#endif
#define NV_OF_API_MAJOR_VERSION 2
#define NV_OF_API_MINOR_VERSION 0
#define NV_OF_API_VERSION  (uint16_t)((NV_OF_API_MAJOR_VERSION << 4) | NV_OF_API_MINOR_VERSION)
#define MIN_ERROR_STRING_SIZE 80

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

typedef struct NvOFHandle_st            *NvOFHandle;
typedef struct NvOFGPUBufferHandle_st   *NvOFGPUBufferHandle;
typedef struct NVOFPrivDataHandle_st    *NvOFPrivDataHandle;

/**
  *  Supported error codes
*/
typedef enum _NV_OF_STATUS
{
    /**
    * This indicates that API call returned with no errors.
    */
    NV_OF_SUCCESS,

    /**
    * This indicates that HW Optical flow functionality is not supported
    */
    NV_OF_ERR_OF_NOT_AVAILABLE,

    /**
    * This indicates that device passed by the client is not supported.
    */
    NV_OF_ERR_UNSUPPORTED_DEVICE,

    /**
    * This indicates that device passed to the API call is no longer available and
    * needs to be reinitialized.
    */
    NV_OF_ERR_DEVICE_DOES_NOT_EXIST,

    /**
    * This indicates that one or more of the pointers passed to the API call
    * is invalid.
    */
    NV_OF_ERR_INVALID_PTR,

    /**
    * This indicates that one or more of the parameter passed to the API call
    * is invalid.
    */
    NV_OF_ERR_INVALID_PARAM,

    /**
    * This indicates that an API call was made in wrong sequence/order.
    */
    NV_OF_ERR_INVALID_CALL,

    /**
    * This indicates that an invalid struct version was used by the client.
    */
    NV_OF_ERR_INVALID_VERSION,

    /**
    * This indicates that the API call failed because it was unable to allocate
    * enough memory to perform the requested operation.
    */
    NV_OF_ERR_OUT_OF_MEMORY,

    /**
    * This indicates that the OF session has not been initialized with
    * ::NvOFInit() or that initialization has failed.
    */
    NV_OF_ERR_NOT_INITIALIZED,

    /**
    * This indicates that an unsupported parameter was passed by the client.
    */
    NV_OF_ERR_UNSUPPORTED_FEATURE,

    /**
    * This indicates that an unknown internal error has occurred.
    */
   NV_OF_ERR_GENERIC,
} NV_OF_STATUS;

/**
*  Supported bool values
*/
typedef enum _NV_OF_BOOL
{
    NV_OF_FALSE = 0,                              /* < Represents false bool value */
    NV_OF_TRUE  = !NV_OF_FALSE                    /* < Represents true bool value */
} NV_OF_BOOL;

/**
* Supported optical flow and stereo disparity capability values.
*/
typedef enum _NV_OF_CAPS
{
    NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES,      /**< Indicates supported values of ::NV_OF_OUTPUT_VECTOR_GRID_SIZE,
                                                    ::NV_OF_INIT_PARAMS::outGridSize should be set with a supported output gridsize. */
    NV_OF_CAPS_SUPPORTED_HINT_GRID_SIZES,        /**< Indicates supported values of ::NV_OF_HINT_VECTOR_GRID_SIZE,
                                                    ::NV_OF_INIT_PARAMS::hintGridSize should be set with a supported hint gridsize. */
    NV_OF_CAPS_SUPPORT_HINT_WITH_OF_MODE,        /**< Indicates external hint support for ::NV_OF_MODE_OPTICALFLOW mode.
                                                    0: External hint not supported for ::NV_OF_MODE_OPTICALFLOW mode.
                                                    1: External hint is supported for ::NV_OF_MODE_OPTICALFLOW mode. */
    NV_OF_CAPS_SUPPORT_HINT_WITH_ST_MODE,        /**< Indicates external hint support for ::NV_OF_MODE_STEREODISPARITY mode.
                                                    0: External hint not supported for ::NV_OF_MODE_STEREODISPARITY mode.
                                                    1: External hint is supported for ::NV_OF_MODE_STEREODISPARITY mode. */
    NV_OF_CAPS_WIDTH_MIN,                        /**< Minimum input width supported. */
    NV_OF_CAPS_HEIGHT_MIN,                       /**< Minimum input height supported. */
    NV_OF_CAPS_WIDTH_MAX,                        /**< Maximum input width supported. */
    NV_OF_CAPS_HEIGHT_MAX,                       /**< Maximum input height supported. */
    NV_OF_CAPS_SUPPORT_ROI,                      /**< Indicates ROI support.
                                                    0: ROIs cannot be specified.
                                                    1: One or more ROIs can be specified. */
    NV_OF_CAPS_SUPPORT_ROI_MAX_NUM,              /**< Indicates maximum number of ROIs supported. */
    NV_OF_CAPS_SUPPORT_MAX
} NV_OF_CAPS;

/**
* Supported optical flow/stereo disparity performance levels.
*/
typedef enum _NV_OF_PERF_LEVEL
{
    NV_OF_PERF_LEVEL_UNDEFINED,
    NV_OF_PERF_LEVEL_SLOW = 5,                   /**< Slow perf level results in lowest performance and best quality */
    NV_OF_PERF_LEVEL_MEDIUM = 10,                /**< Medium perf level results in low performance and medium quality */
    NV_OF_PERF_LEVEL_FAST = 20,                  /**< Fast perf level results in high performance and low quality */
    NV_OF_PERF_LEVEL_MAX
} NV_OF_PERF_LEVEL;

/**
* Supported grid size for output buffer ::NV_OF_EXECUTE_PARAMS::outputBuffer.
* Client should set ::NV_OF_INIT_PARAMS::outGridSize with ::NV_OF_OUTPUT_VECTOR_GRID_SIZE values.
*/
typedef enum _NV_OF_OUTPUT_VECTOR_GRID_SIZE
{
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 = 1,          /**< Output buffer grid size is 1x1  */
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 = 2,          /**< Output buffer grid size is 2x2  */
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 = 4,          /**< Output buffer grid size is 4x4  */
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_MAX
} NV_OF_OUTPUT_VECTOR_GRID_SIZE;

/**
* Expected grid size for optional paramater ::NV_OF_EXECUTE_PARAMS::externalHints buffer.
* Client should set ::NV_OF_INIT_PARAMS::hintGridSize with ::NV_OF_HINT_VECTOR_GRID_SIZE values.
*/
typedef enum _NV_OF_HINT_VECTOR_GRID_SIZE
{
    NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED,
    NV_OF_HINT_VECTOR_GRID_SIZE_1 = 1,            /**< Hint buffer grid size is 1x1.*/
    NV_OF_HINT_VECTOR_GRID_SIZE_2 = 2,            /**< Hint buffer grid size is 2x2.*/
    NV_OF_HINT_VECTOR_GRID_SIZE_4 = 4,            /**< Hint buffer grid size is 4x4.*/
    NV_OF_HINT_VECTOR_GRID_SIZE_8 = 8,            /**< Hint buffer grid size is 8x8.*/
    NV_OF_HINT_VECTOR_GRID_SIZE_MAX
} NV_OF_HINT_VECTOR_GRID_SIZE;

/**
* ::NV_OF_MODE enum define values for Optical flow and Stereo disparity modes.
* Client need to set ::NV_OF_INIT_PARAMS::mode with ::NV_OF_MODE values.
* For the ::NV_OF_MODE_OPTICALFLOW mode, the buffer format for ::NV_OF_EXECUTE_PARAMS::externalHints
* and ::NV_OF_EXECUTE_PARAMS::outputBuffer is ::NV_OF_FLOW_VECTOR.
* For the ::NV_OF_MODE_STEREODISPARITY mode, the buffer format for ::NV_OF_EXECUTE_PARAMS::externalHints
* and ::NV_OF_EXECUTE_PARAMS::outputBuffer is ::NV_OF_STEREO_DISPARITY.
*/
typedef enum _NV_OF_MODE
{
    NV_OF_MODE_UNDEFINED,
    NV_OF_MODE_OPTICALFLOW,                       /**< Calculate optical flow between two frames. */
    NV_OF_MODE_STEREODISPARITY,                   /**< Calculate disparity between Stereo view pair. */
    NV_OF_MODE_MAX
} NV_OF_MODE;

/**
*  Supported buffer type for ::NvOFGPUBufferHandle allocation.
*  Client need to set NV_OF_CREATE_BUFFER::bufferUsage with ::NV_OF_BUFFER_USAGE enum values.
*/
typedef enum _NV_OF_BUFFER_USAGE
{
    NV_OF_BUFFER_USAGE_UNDEFINED,
    NV_OF_BUFFER_USAGE_INPUT,                    /**< Input buffer type is used to allocate ::NV_OF_INPUT_EXECUTE_PARAMS::inputFrame,
                                                      ::NV_OF_INPUT_EXECUTE_PARAMS::referenceFrame. */
    NV_OF_BUFFER_USAGE_OUTPUT,                   /**< Output buffer type is used to allocate ::NV_OF_OUTPUT_EXECUTE_PARAMS::outputBuffer. */
    NV_OF_BUFFER_USAGE_HINT,                     /**< Hint buffer type is used to allocate ::NV_OF_INPUT_EXECUTE_PARAMS::externalHints.*/
    NV_OF_BUFFER_USAGE_COST,                     /**< Cost buffer type is used to allocate ::NV_OF_OUTPUT_EXECUTE_PARAMS::outputCostBuffer.*/
    NV_OF_BUFFER_USAGE_MAX
} NV_OF_BUFFER_USAGE;

/**
* Supported buffer formats
*/
typedef enum _NV_OF_BUFFER_FORMAT
{
    NV_OF_BUFFER_FORMAT_UNDEFINED,
    NV_OF_BUFFER_FORMAT_GRAYSCALE8,               /**< Input buffer format with 8 bit planar format */
    NV_OF_BUFFER_FORMAT_NV12,                     /**< Input buffer format with 8 bit plannar, UV interleaved */
    NV_OF_BUFFER_FORMAT_ABGR8,                    /**< Input buffer format with 8 bit packed A8B8G8R8 */
    NV_OF_BUFFER_FORMAT_SHORT,                    /**< Output or hint buffer format for stereo disparity */
    NV_OF_BUFFER_FORMAT_SHORT2,                   /**< Output or hint buffer format for optical flow vector */
    NV_OF_BUFFER_FORMAT_UINT,                     /**< Legacy 32-bit Cost buffer format for optical flow vector / stereo disparity.
                                                       This cost buffer format is not performance efficient and results in additional GPU usage.
                                                       Hence users are strongly recommended to use the 8-bit cost buffer format.
                                                       Legacy 32-bit cost buffer format is also planned to be deprecated in future. */
    NV_OF_BUFFER_FORMAT_UINT8,                    /**< 8-bit Cost buffer format for optical flow vector / stereo disparity. */
    NV_OF_BUFFER_FORMAT_MAX
} NV_OF_BUFFER_FORMAT;

/**
* Supported stereo disparity range.  Avaialble for GPUs later than Turing
*/
typedef enum _NV_OF_STEREO_DISPARITY_RANGE
{
    NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED,
    NV_OF_STEREO_DISPARITY_RANGE_128 = 128,
    NV_OF_STEREO_DISPARITY_RANGE_256 = 256,
    NV_OF_STEREO_DISPARITY_RANGE_MAX,
} NV_OF_STEREO_DISPARITY_RANGE;

/**
* \struct NV_OF_FLOW_VECTOR
* Struct needed for optical flow. ::NV_OF_EXECUTE_OUTPUT_PARAMS::outputBuffer will be populated with optical flow
* in ::NV_OF_FLOW_VECTOR format for each ::NV_OF_INIT_PARAMS::outGridSize.
* Flow vectors flowx and flowy are 16-bit values with the lowest 5 bits holding fractional value,
* followed by a 10-bit integer value and the most significant bit being a sign bit.
*/
typedef struct _NV_OF_FLOW_VECTOR
{
    int16_t                         flowx;        /**< x component of flow in S10.5 format */
    int16_t                         flowy;        /**< y component of flow in S10.5 format */
} NV_OF_FLOW_VECTOR;

/**
* \struct NV_OF_STEREO_DISPARITY
* Struct needed for stereo /disparity. ::NV_OF_OUTPUT_EXECUTE_PARAMS::outputBuffer will be populated
* with stereo disparity in ::NV_OF_STEREO_DISPARITY format for each ::NV_OF_INIT_PARAMS::outGridSize.
* Stereo disparity is a 16-bit value with the lowest 5 bits holding fractional value,
* followed by a 11-bit unsigned integer value.
*/
typedef struct _NV_OF_STEREO_DISPARITY
{
    uint16_t                        disparity;    /**< Horizontal displacement[in pixels] in 11.5 format. */
} NV_OF_STEREO_DISPARITY;

/**
* \struct NV_OF_INIT_PARAMS
* Optical flow/stereo disparity session initialization parameters.
*/
typedef struct _NV_OF_INIT_PARAMS
{
    uint32_t                        width;                            /**< [in]: Specifies input buffer width */
    uint32_t                        height;                           /**< [in]: Specifies input buffer height */
    NV_OF_OUTPUT_VECTOR_GRID_SIZE   outGridSize;                      /**< [in]: Specifies flow vector grid size for ::NV_OF_EXECUTE_PARAMS::outputBuffer buffer.*/
    NV_OF_HINT_VECTOR_GRID_SIZE     hintGridSize;                     /**< [in]: Specifies flow vector grid size for ::NV_OF_EXECUTE_PARAMS::externalHints buffer.
                                                                                 This field is only considered if ::NV_OF_INIT_PARAMS::enableExternalHints is set.
                                                                                 hintGridSize should be equal or greater than outGridSize. */
    NV_OF_MODE                      mode;                             /**< [in]: Operating mode for NVOF. Set to a value defined by enum ::NV_OF_MODE. */
    NV_OF_PERF_LEVEL                perfLevel;                        /**< [in]: Specifies perf level. */
    NV_OF_BOOL                      enableExternalHints;              /**< [in]: Set to 1 to enable external hints for optical flow session. */
    NV_OF_BOOL                      enableOutputCost;                 /**< [in]: Set to 1 to enable output cost calculation for optical flow session. */
    NvOFPrivDataHandle              hPrivData;                        /**< [in]: Optical flow private data. It is reserved field and should be set to NULL. */
    NV_OF_STEREO_DISPARITY_RANGE    disparityRange;                   /**< [in]: Speicifies maximum disparity range.
                                                                                 Set to NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED for Turing GPUs. */
    NV_OF_BOOL                      enableRoi;                        /**< [in]: Set to 1 to enable estimation of optical flow/stereo for roi. */
} NV_OF_INIT_PARAMS;

/**
* \struct NV_OF_BUFFER_DESCRIPTOR
* Creation parameters for optical flow buffers.
*/
typedef struct _NV_OF_BUFFER_DESCRIPTOR
{
    uint32_t                        width;                           /**< [in]: Buffer width. */
    uint32_t                        height;                          /**< [in]: Buffer height. */
    NV_OF_BUFFER_USAGE              bufferUsage;                     /**< [in]: To specify buffer usage type.
                                                                     ::NV_OF_BUFFER_USAGE_OUTPUT buffer usage type accepts ::NV_OF_CREATE_BUFFER::width,
                                                                     ::NV_OF_BUFFER_DESCRIPTOR::height in ::NV_OF_INIT_PARAMS::outGridSize units.
                                                                     ::NV_OF_BUFFER_USAGE_HINT buffer usage type accepts ::NV_OF_BUFFER_DESCRIPTOR::width,
                                                                     ::NV_OF_BUFFER_DESCRIPTOR::height in ::NV_OF_INIT_PARAMS::hintGridSize units. */
    NV_OF_BUFFER_FORMAT             bufferFormat;                    /**< [in]: Buffer format. */

} NV_OF_BUFFER_DESCRIPTOR;

/**
* \struct NV_OF_ROI_RECT
* Specifies the co-ordinates of the Region Of Interest (ROI)
* ROI rects should satisfy below requirements:
*   1. NV_OF_ROI_RECT::start_x should align to (32 * NV_OF_INIT_PARAMS::outGridSize)
*   2. NV_OF_ROI_RECT::width should align to (32 * NV_OF_INIT_PARAMS::outGridSize)
*   3. NV_OF_ROI_RECT::start_y should align to (8 * max(NV_OF_INIT_PARAMS::outGridSize, 2))
*   4. NV_OF_ROI_RECT::height should align to (8 * NV_OF_INIT_PARAMS::outGridSize)
*   5. NV_OF_ROI_RECT::width >= 32 && NV_OF_ROI_RECT::height >= 16; maximum size 8192x8192
*   6. Whole ROI region should be inside of the image
* Optical flow/stereo disparity vectors out side of ROI are invalid and should not be used.
*/
struct NV_OF_ROI_RECT
{
    uint32_t                        start_x;                         /**< [in]: ROI start position in x-direction. */
    uint32_t                        start_y;                         /**< [in]: ROI start position in y-direction. */
    uint32_t                        width;                           /**< [in]: Width of ROI. */
    uint32_t                        height;                          /**< [in]: Height of ROI. */
};

/**
* \struct NV_OF_EXECUTE_INPUT_PARAMS
* Parameters which are sent per frame for optical flow/stereo disparity execution.
*/
typedef struct _NV_OF_EXECUTE_INPUT_PARAMS
{
    NvOFGPUBufferHandle             inputFrame;                      /**< [in]: If ::NV_OF_INIT_PARAMS::mode is ::NV_OF_MODE_OPTICALFLOW, this specifies the handle to the buffer containing the input frame.
                                                                                If ::NV_OF_INIT_PARAMS::mode is ::NV_OF_MODE_STEREODISPARITY, this specifies the handle to the buffer containing the rectified left view. */
    NvOFGPUBufferHandle             referenceFrame;                  /**< [in]: If ::NV_OF_INIT_PARAMS::mode is ::NV_OF_MODE_OPTICALFLOW, this specifies the handle to the buffer containing the reference frame.
                                                                                If ::NV_OF_INIT_PARAMS::mode is ::NV_OF_MODE_STEREODISPARITY, this specifies the handle to the buffer containing the rectified right view. */
    NvOFGPUBufferHandle             externalHints;                   /**< [in]: It is an optional input, This field will be considered if client had set ::NV_OF_INIT_PARAMS::enableExternalHint flag.
                                                                                Client can pass some available predictors as hints.
                                                                                Optical flow driver will search around those hints to optimize flow vectors quality.
                                                                                Expected hint buffer format is ::NV_OF_FLOW_VECTOR, ::NV_OF_STEREO_DISPARITY
                                                                                for ::NV_OF_MODE_OPTICALFLOW, ::NV_OF_MODE_STEREODISPARITY modes respectively for
                                                                                each ::NV_OF_INIT_PARAMS::hintGridSize in a frame. */
    NV_OF_BOOL                      disableTemporalHints;            /**< [in]: Temporal hints yield better accuracy flow vectors when running on successive frames of a continuous video (without major scene changes).
                                                                                When disableTemporalHints = 0, optical flow vectors from previous NvOFExecute call are automatically used as hints for the current NvOFExecute call.
                                                                                However, when running optical flow on pairs of images which are completely independent of each other, temporal hints are useless
                                                                                and in fact, they will degrade the quality. Therefore, it is recommended to set disableTemporalHints = 1 in this case.*/
    uint32_t                        padding;                         /**< [in]: Padding.  Must be set to 0. */
    NvOFPrivDataHandle              hPrivData;                       /**< [in]: Optical flow private data handle. It is reserved field and should be set to NULL. */
    uint32_t                        padding2;                        /**< [in]: Padding.  Must be set to 0. */
    uint32_t                        numRois;                         /**< [in]: Number of ROIs. */
    NV_OF_ROI_RECT*                 roiData;                         /**< [in]: Pointer to the NV_OF_ROI_RECTs data.  Size of this buffer should be atleast numROIs * sizeof(NV_OF_ROI_RECT). */
} NV_OF_EXECUTE_INPUT_PARAMS;

/**
* \struct NV_OF_EXECUTE_OUTPUT_PARAMS
* Parameters which are received per frame for optical flow/stereo disparity execution.
*/
typedef struct _NV_OF_EXECUTE_OUTPUT_PARAMS
{
    NvOFGPUBufferHandle            outputBuffer;                     /**< [in]: Specifies the pointer to optical flow or stereo disparity buffer handle.
                                                                                ::outputBuffer will be populated with optical flow in
                                                                                ::NV_OF_FLOW_VECTOR format or stereo disparity in
                                                                                ::NV_OF_STEREO_DISPARITY format for each
                                                                                ::NV_OF_VECTOR_GRID_SIZE::outGridSize in a frame.*/
    NvOFGPUBufferHandle            outputCostBuffer;                 /**< [in]: Specifies the pointer to output cost calculation buffer handle. */
    NvOFPrivDataHandle             hPrivData;                        /**< [in]: Optical flow private data handle. It is reserved field and should be set to NULL. */
} NV_OF_EXECUTE_OUTPUT_PARAMS;

/**
* \brief Initialize NVIDIA Video Optical Flow Interface and validates input params.
*
* Initializes NVIDIA Video Optical Flow Interface and validates input params.
* It also initializes NVIDIA Video Optical Flow driver with the init value passed in ::NV_OF_INIT_PARAMS
* structure.
*
* \param [in] hOf
*   Object of ::NvOFHandle type.
* \param [in] initParams
*   Pointer to the ::NV_OF_INIT_PARAMS structure.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_UNSUPPORTED_DEVICE \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_UNSUPPORTED_PARAM \n
* ::NV_OF_ERR_OUT_OF_MEMORY \n
* ::NV_OF_ERR_INVALID_PARAM \n
* ::NV_OF_ERR_INVALID_VERSION \n
* ::NV_OF_ERR_OF_NOT_INITIALIZED \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFINIT) (NvOFHandle hOf, const NV_OF_INIT_PARAMS *initParams);

/**
* \brief Kick off computation of optical flow between input and reference frame.
*
* This is asynchronous function call which kicks off computation of optical flow or stereo disparity
* between ::NV_OF_EXECUTE_INPUT_PARAMS::inputFrame and ::NV_OF_EXECUTE_INPUT_PARAMS::referenceFrame and returns
* after submitting  execute paramaters to optical flow engine.
* ::NV_OF_EXECUTE_OUTPUT_PARAMS::outputBuffer will be populated with optical flow or stereo disparity
* based on ::NV_OF_INIT_PARAMS:mode is NV_OF_MODE_OPTICALFLOW or NV_OF_MODE_STEREODISPARITY respectively.
*
* \param [in] hOf
*   Object of ::NvOFHandle type.
* \param [in] executeInParams
*   pointer to the ::NV_OF_EXECUTE_INPUT_PARAMS structure.
* \param [out] executeOutParams
*   pointer to the ::NV_OF_EXECUTE_OUTPUT_PARAMS structure.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_INVALID_DEVICE \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_UNSUPPORTED_PARAM \n
* ::NV_OF_ERR_OUT_OF_MEMORY \n
* ::NV_OF_ERR_INVALID_PARAM \n
* ::NV_OF_ERR_INVALID_VERSION \n
* ::NV_OF_ERR_OF_NOT_INITIALIZED \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFEXECUTE) (NvOFHandle hOf, const NV_OF_EXECUTE_INPUT_PARAMS *executeInParams, NV_OF_EXECUTE_OUTPUT_PARAMS *executeOutParams);

/**
* \brief Release optical flow API and driver resources.
*
* Releases resources and waits until all resources are gracefully released.
*
*  \param [in] hOf
*   Object of ::NvOFHandle type.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_OF_NOT_INITIALIZED \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFDESTROY) (NvOFHandle hOf);

/**
* \brief Populate error buffer with the description of last failure.
*
* Populates lastError[] with the description of last failure.
*
* \param [in] hOf
*   Object of ::NvOFHandle type.
* \param [in/out] lastError
*   lastError is a char array, minimum expected size of lastError[] is MIN_ERROR_STRING_SIZE characters.
*   After execution of this function call, lastError[] is populated with error string.
* \param [in/out] As an input parameter, "size" indicates the size of the array provided by the client.
*   After execution of this function call, "size" field indicates the number of characters written into
*   "lastError" excluding null character.
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_OF_NOT_INITIALIZED \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFGETLASTERROR) (NvOFHandle hOf, char lastError[], uint32_t *size);

/**
* \brief Populate capability array for specified ::NV_OF_CAPS value.
* This is to be called in two stages.
* It returns the number of capability values for specified ::NV_OF_CAPS value when
* queried with "capsVal" set to NULL.
* It populates capsVal array with capability values for specified ::NV_OF_CAPS value
* when queried with "capsVal" set to non-NULL value.
*
*  \param [in] hOf
*   Object of ::NvOFHandle type.
*  \param [in] capsParam
*   object of ::NV_OF_CAPS type.
*  \param [out] capsVal
*   Pointer to uint32_t, minimum expected size of capsVal is the "size" returned by the this function call
*   queried with "capsVal" set to NULL.
* \param [out] size
*   Pointer to uint32_t, which stores size of populated capsVal.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_OF_NOT_INITIALIZED \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFGETCAPS) (NvOFHandle hOf, NV_OF_CAPS capsParam, uint32_t *capsVal, uint32_t *size);

/**
* \brief Get the largest API version supported by the driver.
*
* This function can be used by clients to determine if the driver supports
* the API header the application was compiled with.
*
* \param [out] version
*   Pointer to the requested value. The 4 least significant bits in the returned
*   indicate the minor version and the rest of the bits indicate the major
*   version of the largest supported version.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
*/
NV_OF_STATUS NVOFAPI NvOFGetMaxSupportedApiVersion(uint32_t* version);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif
