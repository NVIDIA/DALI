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
* \file NvOpticalFlowCuda.h
*   NVIDIA GPUs - Turing and above contains a hardware-based optical flow engine
*   which provides fully-accelerated hardware-based optical flow and stereo estimation.
*   nvOpticalFlowCuda.h provides cuda specific enums, structure definitions and function pointers prototypes.
* \date 2018
*  This file contains CUDA specific enums, structure definitions and function prototypes.
*/

#ifndef _NV_OPTICALFLOW_CUDA_H_
#define _NV_OPTICALFLOW_CUDA_H_
#include "nvOpticalFlowCommon.h"
#include <cuda.h>
#define MAX_NUM_PLANES 3

#if defined(__cplusplus)

extern "C"
{
#endif /* __cplusplus */

/**
* Supported CUDA buffer types.
*/
typedef enum _NV_OF_CUDA_BUFFER_TYPE
{
    NV_OF_CUDA_BUFFER_TYPE_UNDEFINED,
    NV_OF_CUDA_BUFFER_TYPE_CUARRAY,           /**< Buffer type is CUarray */
    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,       /**< Buffer type is CUdeviceptr */
    NV_OF_CUDA_BUFFER_TYPE_MAX
} NV_OF_CUDA_BUFFER_TYPE;

/**
* \struct NV_BUFFER_STRIDE
* Horizontal and vertical strides of a plane.
*/
typedef struct _NV_OF_BUFFER_STRIDE
{
    uint32_t strideXInBytes;               /**< Horizontal stride. */
    uint32_t strideYInBytes;               /**< Vertical stride. */
} NV_OF_BUFFER_STRIDE;

/**
* \struct NV_OF_CUDA_BUFFER_STRIDE_INFO
* This structure stores buffer stride information which is populated in the ::nvOFGPUBufferGetStrideInfo() API.
*/
typedef struct _NV_OF_CUDA_BUFFER_STRIDE_INFO
{
    NV_OF_BUFFER_STRIDE strideInfo[MAX_NUM_PLANES];    /**< Stride information of each plane.*/
    uint32_t            numPlanes;                     /**< Number of planes. */
} NV_OF_CUDA_BUFFER_STRIDE_INFO;

/**
* \brief Create an instance of NvOFHandle object.
*
* This function creates an instance of NvOFHandle object and returns status.
* Client is expected to release NvOFHandle resource using Destroy function call.
*
* \param [in] cuContext
*   Should be set to cuda context created by Client.
* \param [out] NvOFHandle*
*   Pointer of class ::NvOFHandle object.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_OUT_OF_MEMORY \n
* ::NV_OF_ERR_INVALID_VERSION \n
* ::NV_OF_ERR_UNSUPPORTED_PARAM \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVCREATEOPTICALFLOWCUDA) (CUcontext device, NvOFHandle *hOf);

/**
* \brief Set input and output cuda stream for specified optical flow instance.
*
* Optical flow algorithm may optionally involve cuda preprocessing on the input buffers and post
* processing on the output flow vectors. This function is used to set input and output cuda stream
* to pipeline and synchronize the cuda preprocessing and post processing tasks with OF HW engine.
* Client should call this function before Execute function to update input and/or output streams otherwise
* Execute function will either use preset input, output streams or default streams(If streams are never set before).
*
* \param [in] hOf
*   Object of ::NvOFHandle type.
* \param [in] inputStream
*   CUstream type object which is used to process ::NV_OF_EXECUTE_PARAMS::inputFrame,
*   ::NV_OF_EXECUTE_PARAMS::referenceFrame and optional NV_OF_EXECUTE_PARAMS::externalHints.
* \param [in] outputStream
*  CUstream type object which is used to process ::NV_OF_EXECUTE_PARAMS::outputBuffer and
*  optional NV_OF_EXECUTE_PARAMS::costBuffer.
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
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFSETIOCUDASTREAMS) (NvOFHandle hOf, CUstream inputStream, CUstream outputStream);

/**
* \brief Create ::NvOFGPUBufferHandle resource.
*
* This function creates ::NvOFGPUBufferHandle resource for specified cuda bufferType.
*
* \param [in] hOf
*   Pointer to the NvOFHandle.
* \param [in] createBufferParams
*   pointer of the ::NV_OF_CREATE_BUFFER.
* \param [out] ofGpuBuffer
*   Output pointer of ::NvOFGPUBufferHandle type.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
* ::NV_OF_ERR_DEVICE_DOES_NOT_EXIST \n
* ::NV_OF_ERR_OUT_OF_MEMORY \n
* ::NV_OF_ERR_INVALID_PARAM \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFCREATEGPUBUFFERCUDA) (NvOFHandle hOf, const NV_OF_BUFFER_DESCRIPTOR *bufferDesc,
                                                         NV_OF_CUDA_BUFFER_TYPE bufferType, NvOFGPUBufferHandle *hOfGpuBuffer);

/**
* \brief Return CUarray object associated with ::NvOFGPUBufferHandle type resource.
*
* \param [in] ofGpuBuffer
*  Object of type NvOFGPUBufferHandle, created by a call to NvOFCreateGPUBufferCuda() with bufferType set to ::NV_OF_CUDA_BUFFER_TYPE_CUARRAY.
*
* \return
* Object of CUarray type.
* If ofGpubuffer corresponds to a GPU buffer that was not created with buffer type NV_OF_CUDA_BUFFER_TYPE_CUARRAY,
* this function returns NULL
*/
typedef CUarray(NVOFAPI* PFNNVOFGPUBUFFERGETCUARRAY) (NvOFGPUBufferHandle ofGpuBuffer);

/**
* \brief Return CUdeviceptr object associated with ::NvOFGPUBufferHandle type resource.
*
* \param [in] ofGpuBuffer
*  Object of type NvOFGPUBufferHandle, created by a call to NvOFCreateGPUBufferCuda() with bufferType set to ::NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR.
*
* \return
* Object of the CUdeviceptr type.
* If ofGpubuffer corresponds to a GPU buffer that was not created with buffer type NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
* this function returns 0
*/
typedef CUdeviceptr(NVOFAPI* PFNNVOFGPUBUFFERGETCUDEVICEPTR) (NvOFGPUBufferHandle ofGpuBuffer);

/**
* \brief Populates buffer information associated with ::NvOFGPUBufferHandle type resource.
*
* Populates structure ::NV_OF_CUDA_BUFFER_STRIDE_INFO with the horizontal and vertical stride details of all the planes.
* \param [in] ofGpuBuffer
*   Object of type NvOFGPUBufferHandle, created by a call to NvOFCreateGPUBufferCuda().
* \param [out] strideInfo
*   pointer to the ::NV_OF_CUDA_BUFFER_STRIDE_INFO.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_PTR \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNVOFGPUBUFFERGETSTRIDEINFO) (NvOFGPUBufferHandle ofGpuBuffer, NV_OF_CUDA_BUFFER_STRIDE_INFO *strideInfo);

/**
* \brief Destroy NvOFGPUBufferHandle object and associated resources.
*
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_GENERIC \n
*/
typedef NV_OF_STATUS(NVOFAPI* PFNNVOFDESTROYGPUBUFFERCUDA) (NvOFGPUBufferHandle buffer);

/**
* \struct NV_OF_CUDA_API_FUNCTION_LIST
* This is structure of function pointers  which are populated by ::NvOFAPICreateInstanceCuda() API.
* Defination of each cuda specific function pointer is defined above.
*/
typedef struct _NV_OF_CUDA_API_FUNCTION_LIST
{
    PFNNVCREATEOPTICALFLOWCUDA                                 nvCreateOpticalFlowCuda;
    PFNNVOFINIT                                                nvOFInit;
    PFNNVOFCREATEGPUBUFFERCUDA                                 nvOFCreateGPUBufferCuda;
    PFNNVOFGPUBUFFERGETCUARRAY                                 nvOFGPUBufferGetCUarray;
    PFNNVOFGPUBUFFERGETCUDEVICEPTR                             nvOFGPUBufferGetCUdeviceptr;
    PFNVOFGPUBUFFERGETSTRIDEINFO                               nvOFGPUBufferGetStrideInfo;
    PFNNVOFSETIOCUDASTREAMS                                    nvOFSetIOCudaStreams;
    PFNNVOFEXECUTE                                             nvOFExecute;
    PFNNVOFDESTROYGPUBUFFERCUDA                                nvOFDestroyGPUBufferCuda;
    PFNNVOFDESTROY                                             nvOFDestroy;
    PFNNVOFGETLASTERROR                                        nvOFGetLastError;
    PFNNVOFGETCAPS                                             nvOFGetCaps;
} NV_OF_CUDA_API_FUNCTION_LIST;

/**
* \brief ::NvOFAPICreateInstanceCuda() API is the entry point to the NvOFAPI interface.
*
* ::NvOFAPICreateInstanceCuda() API populates functionList with function pointers to the API routines implemented by the
 * NvOFAPI interface.
*
* \return
* ::NV_OF_SUCCESS \n
* ::NV_OF_ERR_INVALID_VERSION \n
* :: NV_OF_ERR_INVALID_PTR \n
*/
NV_OF_STATUS NVOFAPI NvOFAPICreateInstanceCuda(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST  *functionList);
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif
