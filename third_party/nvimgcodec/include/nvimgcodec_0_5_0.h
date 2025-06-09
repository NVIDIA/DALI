/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

/** 
 * @brief The nvImageCodec library and extension API
 * 
 * @file nvimgcodec.h
 *   
 */

#ifndef NVIMGCODEC_HEADER
#define NVIMGCODEC_HEADER

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include "nvimgcodec_version.h"

#ifndef NVIMGCODECAPI
    #ifdef _WIN32
        #define NVIMGCODECAPI __declspec(dllexport)
    #elif __GNUC__ >= 4
        #define NVIMGCODECAPI __attribute__((visibility("default")))
    #else
        #define NVIMGCODECAPI
    #endif
#endif

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @brief Maximal size of codec name
 */
#define NVIMGCODEC_MAX_CODEC_NAME_SIZE 256

/**
 * @brief Defines device id as current device
 */
#define NVIMGCODEC_DEVICE_CURRENT -1

/**
 * @brief Defines device id as a CPU only 
 */
#define NVIMGCODEC_DEVICE_CPU_ONLY -99999

/**
 * @brief Maximal number of dimensions.
 */
#define NVIMGCODEC_MAX_NUM_DIM 5

/**
 * @brief Maximum number of image planes.
 */
#define NVIMGCODEC_MAX_NUM_PLANES 32

/**
 * @brief Maximum number of JPEG2000 resolutions.
 */
#define NVIMGCODEC_JPEG2K_MAXRES 33

    /**
     * @brief Opaque nvImageCodec library instance type.
     */
    struct nvimgcodecInstance;

    /**
     * @brief Handle to opaque nvImageCodec library instance type.
     */
    typedef struct nvimgcodecInstance* nvimgcodecInstance_t;

    /**
     * @brief Opaque Image type.
     */
    struct nvimgcodecImage;

    /**
     * @brief Handle to opaque Image type.
     */
    typedef struct nvimgcodecImage* nvimgcodecImage_t;

    /**
     * @brief Opaque Code Stream type.
     */
    struct nvimgcodecCodeStream;

    /**
     * @brief Handle to opaque Code Stream type.
     */
    typedef struct nvimgcodecCodeStream* nvimgcodecCodeStream_t;

    /**
     * @brief Opaque Parser type.
     */
    struct nvimgcodecParser;

    /**
     * @brief Handle to opaque Parser type.
     */
    typedef struct nvimgcodecParser* nvimgcodecParser_t;

    /**
     * @brief Opaque Encoder type.
     */
    struct nvimgcodecEncoder;

    /**
     * @brief Handle to opaque Encoder type.
     */
    typedef struct nvimgcodecEncoder* nvimgcodecEncoder_t;

    /**
     * @brief Opaque Decoder type.
     */
    struct nvimgcodecDecoder;

    /**
     * @brief Handle to opaque Decoder type.
     */
    typedef struct nvimgcodecDecoder* nvimgcodecDecoder_t;

    /**
     * @brief Opaque Debug Messenger type.
     */
    struct nvimgcodecDebugMessenger;

    /**
     * @brief Handle to opaque Debug Messenger type.
     */
    typedef struct nvimgcodecDebugMessenger* nvimgcodecDebugMessenger_t;

    /**
     * @brief Opaque Extension type.
     */
    struct nvimgcodecExtension;

    /**
     * @brief Handle to opaque Extension type.
     */
    typedef struct nvimgcodecExtension* nvimgcodecExtension_t;

    /**
     * @brief Opaque Future type.
     */
    struct nvimgcodecFuture;

    /**
     * @brief Handle to opaque Future type.
     */
    typedef struct nvimgcodecFuture* nvimgcodecFuture_t;

    /**
     * @brief Structure types supported by the nvImageCodec API.
     * 
     * Each value corresponds to a particular structure with a type member and matching  structure name.
     */
    typedef enum
    {
        NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES,
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR,
        NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR,
        NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
        NVIMGCODEC_STRUCTURE_TYPE_REGION,
        NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
        NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO,
        NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO,
        NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO,
        NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
        NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA,
        NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC,
        NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
        NVIMGCODEC_STRUCTURE_TYPE_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecStructureType_t;

    /**
     * @brief The nvImageCodec properties.
     * 
     * @see nvimgcodecGetProperties
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        uint32_t version;         /**< The nvImageCodec library version. */
        uint32_t ext_api_version; /**< The nvImageCodec extension API version. */
        uint32_t cudart_version;  /**< The version of CUDA Runtime with which nvImageCodec library was built. */

    } nvimgcodecProperties_t;

    /** 
     * @brief Function type for device memory resource allocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer where to write pointer to allocated memory.
     * @param [in] size How many bytes to allocate.
     * @param [in] stream CUDA stream    
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcodecDeviceMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for device memory deallocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer to memory buffer to be deallocated.
     *                 If NULL, the operation must do nothing, successfully.
     * @param [in] size How many bytes was allocated (size passed during allocation).
     * @param [in] stream CUDA stream   
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcodecDeviceFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for host pinned memory resource allocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer where to write pointer to allocated memory.
     * @param [in] size How many bytes to allocate.
     * @param [in] stream CUDA stream    
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcodecPinnedMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for host pinned memory deallocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer to memory buffer to be deallocated.
     *                 If NULL, the operation must do nothing, successfully.
     * @param [in] size How many bytes was allocated (size passed during allocation). 
     * @param [in] stream CUDA stream   
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcodecPinnedFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    /**
     * @brief Device memory allocator.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecDeviceMalloc_t device_malloc; /**< Allocate memory on the device. */
        nvimgcodecDeviceFree_t device_free;     /**< Frees memory on the device.*/
        void* device_ctx;                       /**< When invoking the allocators, this context will 
                                                    be pass as input to allocator functions.*/
        size_t device_mem_padding;              /**< Any device memory allocation 
                                                    would be padded to the multiple of specified number of bytes */
    } nvimgcodecDeviceAllocator_t;

    /** 
     * @brief Host pinned memory allocator. 
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecPinnedMalloc_t pinned_malloc; /**< Allocate host pinned memory: memory directly 
                                                    accessible by both CPU and cuda-enabled GPU. */
        nvimgcodecPinnedFree_t pinned_free;     /**< Frees host pinned memory.*/
        void* pinned_ctx;                       /**< When invoking the allocators, this context will
                                                    be pass as input to allocator functions.*/
        size_t pinned_mem_padding;              /**< Any pinned host memory allocation
                                                    would be padded to the multiple of specified number of bytes */
    } nvimgcodecPinnedAllocator_t;

    /** 
     * @brief The return status codes of the nvImageCodec API
     */
    typedef enum
    {
        NVIMGCODEC_STATUS_SUCCESS = 0,
        NVIMGCODEC_STATUS_NOT_INITIALIZED = 1,
        NVIMGCODEC_STATUS_INVALID_PARAMETER = 2,
        NVIMGCODEC_STATUS_BAD_CODESTREAM = 3,
        NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED = 4,
        NVIMGCODEC_STATUS_ALLOCATOR_FAILURE = 5,
        NVIMGCODEC_STATUS_EXECUTION_FAILED = 6,
        NVIMGCODEC_STATUS_ARCH_MISMATCH = 7,
        NVIMGCODEC_STATUS_INTERNAL_ERROR = 8,
        NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED = 9,
        NVIMGCODEC_STATUS_MISSED_DEPENDENCIES = 10,
        NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED = 11,
        NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER = 12,
        NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM = 13,
        NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED = 14,
        NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE = 15,
        NVIMGCODEC_STATUS_EXTENSION_ARCH_MISMATCH = 16,
        NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR = 17,
        NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED = 18,
        NVIMGCODEC_STATUS_EXTENSION_INCOMPLETE_BITSTREAM = 19,
        NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED = 20,
        NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR = 21,
        NVIMGCODEC_STATUS_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecStatus_t;

    /**
     * @brief Describes type sample of data. 
     * 
     * Meaning of bits:
     * 0 bit      -> 0 - unsigned, 1- signed
     * 1..7 bits  -> define type
     * 8..15 bits -> type bitdepth
     * 
     */
    typedef enum
    {
        NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN = 0,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT8 = 0x0801,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 = 0x0802,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT16 = 0x1003,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16 = 0x1004,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT32 = 0x2005,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32 = 0x2006,
        NVIMGCODEC_SAMPLE_DATA_TYPE_INT64 = 0x4007,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64 = 0x4008,
        NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16 = 0x1009,
        NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32 = 0x200B,
        NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64 = 0x400D,
        NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED = -1,
        NVIMGCODEC_SAMPLE_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecSampleDataType_t;

    /** 
     * @brief Chroma subsampling.
    */
    typedef enum
    {
        NVIMGCODEC_SAMPLING_NONE = 0,
        NVIMGCODEC_SAMPLING_444 = NVIMGCODEC_SAMPLING_NONE,
        NVIMGCODEC_SAMPLING_422 = 2,
        NVIMGCODEC_SAMPLING_420 = 3,
        NVIMGCODEC_SAMPLING_440 = 4,
        NVIMGCODEC_SAMPLING_411 = 5,
        NVIMGCODEC_SAMPLING_410 = 6,
        NVIMGCODEC_SAMPLING_GRAY = 7,
        NVIMGCODEC_SAMPLING_410V = 8,
        NVIMGCODEC_SAMPLING_UNSUPPORTED = -1,
        NVIMGCODEC_SAMPLING_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecChromaSubsampling_t;

    /**
     * @brief Provides information how color components are matched to channels in given order and channels are matched to planes.
     */
    typedef enum
    {
        NVIMGCODEC_SAMPLEFORMAT_UNKNOWN = 0,
        NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED = 1, //**< unchanged planar */
        NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED = 2, //**< unchanged interleaved */
        NVIMGCODEC_SAMPLEFORMAT_P_RGB = 3,       //**< planar RGB */
        NVIMGCODEC_SAMPLEFORMAT_I_RGB = 4,       //**< interleaved RGB */
        NVIMGCODEC_SAMPLEFORMAT_P_BGR = 5,       //**< planar BGR */
        NVIMGCODEC_SAMPLEFORMAT_I_BGR = 6,       //**< interleaved BGR */
        NVIMGCODEC_SAMPLEFORMAT_P_Y = 7,         //**< Y component only */
        NVIMGCODEC_SAMPLEFORMAT_P_YUV = 9,       //**< YUV planar format */
        NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED = -1,
        NVIMGCODEC_SAMPLEFORMAT_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecSampleFormat_t;

    /** 
     * @brief Defines color specification.
     */
    typedef enum
    {
        NVIMGCODEC_COLORSPEC_UNKNOWN = 0,
        NVIMGCODEC_COLORSPEC_UNCHANGED = NVIMGCODEC_COLORSPEC_UNKNOWN,
        NVIMGCODEC_COLORSPEC_SRGB = 1,
        NVIMGCODEC_COLORSPEC_GRAY = 2,
        NVIMGCODEC_COLORSPEC_SYCC = 3,
        NVIMGCODEC_COLORSPEC_CMYK = 4,
        NVIMGCODEC_COLORSPEC_YCCK = 5,
        NVIMGCODEC_COLORSPEC_UNSUPPORTED = -1,
        NVIMGCODEC_COLORSPEC_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecColorSpec_t;

    /** 
     *  @brief Defines orientation of an image.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        int rotated; /**< Rotation angle in degrees (clockwise). Only multiples of 90 are allowed. */
        int flip_x;  /**< Flip horizontal 0 or 1*/
        int flip_y;  /**< Flip vertical 0 or 1*/
    } nvimgcodecOrientation_t;

    /**
     * @brief Defines plane of an image.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        uint32_t width;                         /**< Plane width. First plane defines width of image. */
        uint32_t height;                        /**< Plane height. First plane defines height of image.*/
        size_t row_stride;                      /**< Number of bytes need to offset to next row of plane. */
        uint32_t num_channels;                  /**< Number of channels. Color components, are always first
                                                    but there can be more channels than color components.*/
        nvimgcodecSampleDataType_t sample_type; /**< Sample data type. @see  nvimgcodecSampleDataType_t */
        uint8_t precision;                      /**< Value 0 means that precision is equal to sample type bitdepth */
    } nvimgcodecImagePlaneInfo_t;

    /**
     * @brief Defines region of an image.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        int ndim;                          /**< Number of dimensions, 0 value means no region. */
        int start[NVIMGCODEC_MAX_NUM_DIM]; /**< Region start position at the particular dimension. */
        int end[NVIMGCODEC_MAX_NUM_DIM];   /**< Region end position at the particular dimension. */
    } nvimgcodecRegion_t;

    /**
     * @brief Defines buffer kind in which image data is stored.
     */
    typedef enum
    {
        NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN = 0,
        NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE = 1, /**< GPU-accessible with planes in pitch-linear layout. */
        NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST = 2,   /**< Host-accessible with planes in pitch-linear layout. */
        NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED = -1,
        NVIMGCODEC_IMAGE_BUFFER_KIND_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecImageBufferKind_t;

    /**
     * @brief Defines information about an image.
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        char codec_name[NVIMGCODEC_MAX_CODEC_NAME_SIZE]; /**< Information about codec used. Only valid when used with code stream. */

        nvimgcodecColorSpec_t color_spec;                 /**< Image color specification. */
        nvimgcodecChromaSubsampling_t chroma_subsampling; /**< Image chroma subsampling. Only valid with chroma components. */
        nvimgcodecSampleFormat_t sample_format; /**< Defines how color components are matched to channels in given order and channels
                                                    are matched to planes. */
        nvimgcodecOrientation_t orientation;    /**< Image orientation. */
        nvimgcodecRegion_t region;              /**< Region of interest. */

        uint32_t num_planes;                                              /**< Number of image planes. */
        nvimgcodecImagePlaneInfo_t plane_info[NVIMGCODEC_MAX_NUM_PLANES]; /**< Array with information about image planes. */

        void* buffer;                            /**< Pointer to buffer in which image data is stored. */
        size_t buffer_size;                      /**< Size of buffer in which image data is stored. */
        nvimgcodecImageBufferKind_t buffer_kind; /**< Buffer kind in which image data is stored.*/

        cudaStream_t cuda_stream; /**< CUDA stream to synchronize with */
    } nvimgcodecImageInfo_t;

    /** 
     * @brief JPEG Encoding
     *  
     * Currently parseable JPEG encodings (SOF markers)
     * https://www.w3.org/Graphics/JPEG/itu-t81.pdf
     * Table B.1 Start of Frame markers
     */
    typedef enum
    {
        NVIMGCODEC_JPEG_ENCODING_UNKNOWN = 0x0,
        NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT = 0xc0,
        NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN = 0xc1,
        NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN = 0xc2,
        NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN = 0xc3,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN = 0xc5,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN = 0xc6,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN = 0xc7,
        NVIMGCODEC_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS = 0xc8,
        NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC = 0xc9,
        NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC = 0xca,
        NVIMGCODEC_JPEG_ENCODING_LOSSLESS_ARITHMETIC = 0xcb,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC = 0xcd,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC = 0xce,
        NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC = 0xcf,
        NVIMGCODEC_JPEG_ENCODING_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecJpegEncoding_t;

    /** 
     * @brief Defines image information related to JPEG format.
     * 
     * This structure extends information provided in nvimgcodecImageInfo_t
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecJpegEncoding_t encoding; /**< JPEG encoding type. */
    } nvimgcodecJpegImageInfo_t;

    /**
     * @brief Defines image information related to JPEG2000 format.
     *
     * This structure extends information provided in nvimgcodecImageInfo_t
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        uint32_t num_tiles_y;                  /**< Number of tile rows. */
        uint32_t num_tiles_x;                  /**< Number of tile columns. */
        uint32_t tile_height;                  /**< Height of the tile. */
        uint32_t tile_width;                   /**< Width of the tile. */
    } nvimgcodecTileGeometryInfo_t;

    /**
     * @brief Defines decoding/encoding backend kind.
     */
    typedef enum
    {
        NVIMGCODEC_BACKEND_KIND_CPU_ONLY = 1,       /**< Decoding/encoding is executed only on CPU. */
        NVIMGCODEC_BACKEND_KIND_GPU_ONLY = 2,       /**< Decoding/encoding is executed only on GPU. */
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU = 3, /**< Decoding/encoding is executed on both CPU and GPU.*/
        NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY = 4,    /**< Decoding/encoding is executed on GPU dedicated hardware engine. */
    } nvimgcodecBackendKind_t;

    /**
     * @brief Defines how to interpret the load hint parameter.
     */
    typedef enum
    {
        NVIMGCODEC_LOAD_HINT_POLICY_IGNORE = 1, /**< Load hint is not taken into account. */
        NVIMGCODEC_LOAD_HINT_POLICY_FIXED = 2,  /**< Load hint is used to calculate the backend batch size once */
        NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME =
            3, /**< Load hint is used as an initial hint, and it is recalculated on every iteration to reduce the idle time of threads */
    } nvimgcodecLoadHintPolicy_t;


    /** 
     * @brief Defines decoding/encoding backend parameters.
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        /** 
         * Hint to calculate the fraction of the batch items that will be picked by this backend.
         * This is just a hint and a particular implementation can choose to ignore it.
         * Different policies can be selected, see `nvimgcodecLoadHintPolicy_t`
         */
        float load_hint;

        /**
         * If true, the backend load will be adapted on every iteration to minize idle time of the threads.
         */
        nvimgcodecLoadHintPolicy_t load_hint_policy;
    } nvimgcodecBackendParams_t;

    /** 
     * @brief Defines decoding/encoding backend.
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecBackendKind_t kind;     /**< Decoding/encoding backend kind. */
        nvimgcodecBackendParams_t params; /**< Decoding/encoding backend parameters. */
    } nvimgcodecBackend_t;

    /**
     * @brief Processing status bitmask for decoding or encoding . 
     */
    typedef enum
    {
        NVIMGCODEC_PROCESSING_STATUS_UNKNOWN = 0x0,
        NVIMGCODEC_PROCESSING_STATUS_SUCCESS = 0x1,   /**< Processing finished with success. */
        NVIMGCODEC_PROCESSING_STATUS_SATURATED = 0x2, /**< Decoder/encoder could potentially process 
                                                                          image but is saturated. 
                                                                          @see nvimgcodecBackendParams_t load_hint. */

        NVIMGCODEC_PROCESSING_STATUS_FAIL = 0x3,                    /**< Processing failed because unknown reason. */
        NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED = 0x7,         /**< Processing failed because compressed image stream is corrupted. */
        NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED = 0xb,       /**< Processing failed because codec is unsupported */
        NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED = 0x13,    /**< Processing failed because no one from allowed
                                                                          backends is supported. */
        NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED = 0x23,   /**< Processing failed because codec encoding is unsupported. */
        NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED = 0x43, /**< Processing failed because image resolution is unsupported. */
        NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED =
            0x83, /**< Processing failed because some feature of compressed stream is unsupported */

        //These values below describe cases when processing could be possible but with different image format or parameters
        NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED = 0x5,     /**< Color specification unsupported. */
        NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED = 0x9,    /**< Apply orientation was enabled but it is unsupported. */
        NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED = 0x11,           /**< Decoding region of interest is unsupported. */
        NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED = 0x21,      /**< Selected unsupported chroma subsampling . */
        NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED = 0x41,   /**< Selected unsupported sample type. */
        NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED = 0x81, /**< Selected unsupported sample format. */
        NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED = 0x101,   /**< Unsupported number of planes to decode/encode. */
        NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED = 0x201, /**< Unsupported number of channels to decode/encode. */

        NVIMGCODEC_PROCESSING_STATUS_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecProcessingStatus;

    /**
     * @brief Processing status type which combine processing status bitmasks
    */
    typedef uint32_t nvimgcodecProcessingStatus_t;

    /**
     * @brief Decode parameters
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        int apply_exif_orientation; /**<  Apply exif orientation if available. Valid values 0 or 1. */
        int enable_roi;             /**<  Enables region of interest. Valid values 0 or 1. */

    } nvimgcodecDecodeParams_t;

    /**
     * @brief Encode parameters
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        /** 
         * Float value of quality which interpretation depends of particular codec.
         * 
         * For JPEG codec it is expected to be integer values between 1 and 100, where 100 is the highest quality.
         *
         * For WebP codec, value greater than 100 means lossless.
         *
         * For OpenCV JPEG2000 backend this value is multiplied by 10 and used as IMWRITE_JPEG2000_COMPRESSION_X1000,
         * when compression is irreversible.
         *
         * @warning For nvJPEG2000 backend it is unsupported and target_psnr should be used instead.
         */
        float quality;

        /** 
         * Float value of target PSNR (Peak Signal to Noise Ratio)
         * 
         * @warning It is valid only for lossy encoding.
         * @warning It not supported by all codec.
        */
        float target_psnr;
    } nvimgcodecEncodeParams_t;

    /**
     * @brief Progression orders defined in the JPEG2000 standard.
     */
    typedef enum
    {
        NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP = 0, //**< Layer-Resolution-Component-Position progression order. */
        NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP = 1, //**< Resolution-Layer-Component-Position progression order. */
        NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL = 2, //**< Resolution-Position-Component-Layer progression order. */
        NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL = 3, //**< Position-Component-Resolution-Layer progression order. */
        NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL = 4, //**< Component-Position-Resolution-Layer progression order. */
        NVIMGCODEC_JPEG2K_PROG_ORDER_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecJpeg2kProgOrder_t;

    /**
     * @brief JPEG2000 code stream type
     */
    typedef enum
    {
        NVIMGCODEC_JPEG2K_STREAM_J2K = 0, /**< Corresponds to the JPEG2000 code stream.*/
        NVIMGCODEC_JPEG2K_STREAM_JP2 = 1, /**< Corresponds to the .jp2 container.*/
        NVIMGCODEC_JPEG2K_STREAM_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecJpeg2kBitstreamType_t;

    /** 
     * @brief JPEG2000 Encode parameters
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecJpeg2kBitstreamType_t stream_type; /**< JPEG2000 code stream type. */
        nvimgcodecJpeg2kProgOrder_t prog_order;      /**< JPEG2000 progression order. */
        uint32_t num_resolutions;                    /**< Number of resolutions. */
        uint32_t code_block_w;                       /**< Code block width. Allowed values 32, 64 */
        uint32_t code_block_h;                       /**< Code block height. Allowed values 32, 64 */
        int irreversible;                            /**< Sets whether or not to use irreversible encoding. Valid values 0 or 1. */
    } nvimgcodecJpeg2kEncodeParams_t;

    /**
     * @brief JPEG Encode parameters
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        /**
         * Sets whether or not to use optimized Huffman. Valid values 0 or 1.
         * 
         * @note  Using optimized Huffman produces smaller JPEG bitstream sizes with the same quality, but with slower performance.
         */
        int optimized_huffman;
    } nvimgcodecJpegEncodeParams_t;

    /**
     * @brief Bitmask specifying which severities of events cause a debug messenger callback
     */
    typedef enum
    {
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_NONE = 0x00000000,
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE = 0x00000001,   /**< Diagnostic message useful for developers */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG = 0x00000010,   /**< Diagnostic message useful for developers */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO = 0x00000100,    /**< Informational message like the creation of a resource */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING = 0x00001000, /**< Message about behavior that is not necessarily an error,
                                                                but very likely a bug in your application */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR = 0x00010000,   /**< Message about behavior that is invalid and may cause
                                                                improper execution or result of operation (e.g. can't open file)
                                                                but not application */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL = 0x00100000,   /**< Message about behavior that is invalid and may cause crashes
                                                                and forcing to shutdown application */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ALL = 0x0FFFFFFF,     /**< Used in case filtering out by message severity */
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT =
            NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL,
        NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecDebugMessageSeverity_t;

    /**
     * @brief Bitmask specifying which category of events cause a debug messenger callback
     */
    typedef enum
    {
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_NONE = 0x00000000,
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL =
            0x00000001, /**< Some event has happened that is unrelated to the specification or performance */
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_VALIDATION = 0x00000010,  /**< Something has happened that indicates a possible mistake */
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_PERFORMANCE = 0x00000100, /**< Potential non-optimal use */
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL = 0x0FFFFFFF,         /**< Used in case filtering out by message category */
        NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecDebugMessageCategory_t;

    /**
     * @brief Describing debug message passed to debug callback function
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        const char* message;         /**< Null-terminated string detailing the trigger conditions */
        uint32_t internal_status_id; /**< It is internal codec status id */
        const char* codec;           /**< Codec name if codec is rising message or NULL otherwise (e.g framework) */
        const char* codec_id;        /**< Codec id if codec is rising message or NULL otherwise */
        uint32_t codec_version;      /**< Codec version if codec is rising message or 0 otherwise */
    } nvimgcodecDebugMessageData_t;

    /**
     * @brief Debug callback function type.
     * 
     * @param message_severity [in] Message severity
     * @param message_category [in] Message category
     * @param callback_data [in] Debug message data 
     * @param user_data [in] Pointer that was specified during the setup of the callback 
     * @returns 1 if message should not be passed further to other callbacks and 0 otherwise 
     */
    typedef int (*nvimgcodecDebugCallback_t)(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* callback_data, void* user_data);

    /**
     * @brief Debug messenger description.
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        uint32_t message_severity;               /**< Bitmask of message severity to listen for e.g. error or warning.  */
        uint32_t message_category;               /**< Bitmask of message category to listen for e.g. general or performance related. */
        nvimgcodecDebugCallback_t user_callback; /**< Debug callback function */
        void* user_data;                         /**< Pointer to user data which will be passed back to debug callback function. */
    } nvimgcodecDebugMessengerDesc_t;

    /** 
     * @brief Executor description.
     *
     * Codec plugins can use executor available via execution parameters to schedule execution of asynchronous task.  
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        const void* struct_next;               /**< Is NULL or a pointer to an extension structure type. */

        void* instance; /**< Executor instance pointer which will be passed back in functions */

        /**
         * @brief Schedule execution of asynchronous task.
         * 
         * @param instance [in] Pointer to nvimgcodecExecutorDesc_t instance. 
         * @param device_id [in] Device id on which task will be executed.
         * @param sample_idx [in] Index of batch sample to process task on; It will be passed back as an argument in task function. 
         * @param task_context [in] Pointer to task context which will be passed back as an argument in task function.
         * @param task [in] Pointer to task function to schedule.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*schedule)(void* instance, int device_id, int sample_idx, void* task_context,
            void (*task)(int thread_id, int sample_idx, void* task_context));

        /**
         * @brief Starts the execution of all the queued work
         */
        nvimgcodecStatus_t (*run)(void* instance, int device_id);

        /**
         * @brief Blocks until all work issued earlier is complete
         * @remarks It must be called only after `run`
         */
        nvimgcodecStatus_t (*wait)(void* instance, int device_id);

        /** 
         * @brief Gets number of threads.
         * 
         * @param instance [in] Pointer to nvimgcodecExecutorDesc_t instance. 
         * @return Number of threads in executor.
        */
        int (*getNumThreads)(void* instance);
    } nvimgcodecExecutorDesc_t;

    /** 
     * @brief Execution parameters
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        nvimgcodecDeviceAllocator_t* device_allocator; /**< Custom allocator for device memory */
        nvimgcodecPinnedAllocator_t* pinned_allocator; /**< Custom allocator for pinned memory */
        int max_num_cpu_threads;                       /**< Max number of CPU threads in default executor 
                                                           (0 means default value equal to number of cpu cores) */
        nvimgcodecExecutorDesc_t* executor;            /**< Points an executor. If NULL default executor will be used. 
                                                           @note At plugin level API it always points to executor, either custom or default. */
        int device_id;                                 /**< Device id to process decoding on. It can be also specified 
                                                           using defines NVIMGCODEC_DEVICE_CURRENT or NVIMGCODEC_DEVICE_CPU_ONLY. */
        int pre_init;                                  /**< If true, all relevant resources are initialized at creation of the instance */
        int skip_pre_sync;                             /**< If true, synchronization between user stream and per-thread streams is skipped before
                                                           decoding (we only synchronize after decoding). This can be useful when we are sure that
                                                           there are no actions that need synchronization (e.g. a CUDA async allocation on
                                                           the user stream) */
        int num_backends;                              /**< Number of allowed backends passed (if any)
                                                           in backends parameter. For 0, all backends are allowed.*/
        const nvimgcodecBackend_t* backends;           /**< Points a nvimgcodecBackend_t array with defined allowed backends.
                                                           For nullptr, all backends are allowed. */
    } nvimgcodecExecutionParams_t;

    /**
     * @brief Input/Output stream description.
     * 
     * This abstracts source or sink for code stream bytes.
     *  
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance; /**< I/O stream description instance pointer which will be passed back in functions */

        /**
         * @brief Reads all requested data from the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of read bytes.
         * @param buf [in]   Pointer to output buffer
         * @param bytes [in] Number of bytes to read
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*read)(void* instance, size_t* output_size, void* buf, size_t bytes);

        /**
         * @brief Writes all requested data to the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of written bytes.
         * @param buf [in]   Pointer to input buffer
         * @param bytes [in] Number of bytes to write
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*write)(void* instance, size_t* output_size, void* buf, size_t bytes);

        /**
         * @brief Writes one character to the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of written bytes.
         * @param ch [in] Character to write.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);

        /**
         * @brief Skips `count` objects in the stream
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param count [in] Number bytes to skip
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*skip)(void* instance, size_t count);

        /**
         * @brief Moves the read pointer in the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param offset  [in] Offset to move.
         * @param whence  [in] Beginning - SEEK_SET, SEEK_CUR or SEEK_END.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*seek)(void* instance, ptrdiff_t offset, int whence);

        /**
         * @brief Retrieves current position, in bytes from the beginning, in the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param offset  [in/out] Pointer where to return current position.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*tell)(void* instance, ptrdiff_t* offset);

        /**
         * @brief Retrieves the length, in bytes, of the stream.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param size  [in/out] Pointer where to return length of the stream.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*size)(void* instance, size_t* size);

        /**
         * @brief Provides expected bytes which are going to be written.  
         * 
         *  This function gives possibility to pre/re-allocate map function.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param bytes [in] Number of expected bytes which are going to be written.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*reserve)(void* instance, size_t bytes);

        /**
         * @brief Requests all data to be written to the output.
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*flush)(void* instance);

        /**
         * @brief Maps data into host memory  
         * 
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.
         * @param buffer [in/out] Points where to return pointer to mapped data. If data cannot be mapped, NULL will be returned.
         * @param offset [in] Offset in the stream to begin mapping.
         * @param size [in] Length of the mapping
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*map)(void* instance, void** buffer, size_t offset, size_t size);

        /**
         * @brief Unmaps previously mapped data
         *  
         * @param instance [in] Pointer to nvimgcodecIoStreamDesc_t instance.         * 
         * @param buffer [in] Pointer to mapped data
         * @param size [in] Length of data to unmap 
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*unmap)(void* instance, void* buffer, size_t size);
    } nvimgcodecIoStreamDesc_t;

    /**
     * @brief Code stream description.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance; /**< Code stream description instance pointer which will be passed back in functions */
        uint64_t id;  /** <Generated id that uniquely identifies the instance */

        nvimgcodecIoStreamDesc_t* io_stream; /**< I/O stream which works as a source or sink of code stream bytes */

        /**
         * @brief Retrieves image information.
         * 
         * @param instance [in] Pointer to nvimgcodecCodeStreamDesc_t instance.
         * @param image_info [in/out] Points where to return image information.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*getImageInfo)(void* instance, nvimgcodecImageInfo_t* image_info);
    } nvimgcodecCodeStreamDesc_t;

    /**
     * @brief Image description.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance; /**< Image instance pointer which will be passed back in functions */

        /**
         * @brief Retrieves image info information.
         * 
         * @param instance [in] Pointer to nvimgcodecImageDesc_t instance.
         * @param image_info [in/out] Points where to return image information.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*getImageInfo)(void* instance, nvimgcodecImageInfo_t* image_info);

        /**
         * @brief Informs that host side of processing of image is ready.
         * 
         * @param instance [in] Pointer to nvimgcodecImageDesc_t instance.
         * @param processing_status [in] Processing status.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
         */
        nvimgcodecStatus_t (*imageReady)(void* instance, nvimgcodecProcessingStatus_t processing_status);
    } nvimgcodecImageDesc_t;

    /**
     * @brief Parser description.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void*
            instance; /**<nvimgcodecStatus_t (*size)(void* instance, size_t* size); Parser description instance pointer which will be passed back in functions */
        const char* id;    /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec; /**< Codec name e.g. jpeg2000 */

        /** 
         * @brief Checks whether parser can parse given code stream.
         * 
         * @param instance [in] Pointer to nvimgcodecParserDesc_t instance.
         * @param result [in/out] Points where to return result of parsing check. Valid values 0 or 1.
         * @param code_stream [in] Code stream to parse check.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*canParse)(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream);

        /**
         * Creates parser.
         * 
         * @param [in] Pointer to nvimgcodecParserDesc_t instance.
         * @param [in/out] Points where to return handle to created parser.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*create)(void* instance, nvimgcodecParser_t* parser);

        /** 
         * Destroys parser.
         * 
         * @param parser [in] Parser handle to destroy.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*destroy)(nvimgcodecParser_t parser);

        /**
         * @brief Parses given code stream and returns image information.
         * 
         * @param parser [in] Parser handle.
         * @param image_info [in/out] Points where to return image information.
         * @param code_stream [in] Code stream to parse.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*getImageInfo)(
            nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);
    } nvimgcodecParserDesc_t;

    /**
     * @brief Encoder description.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                       /**< Encoder description instance pointer which will be passed back in functions */
        const char* id;                       /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec;                    /**< Codec name e.g. jpeg2000 */
        nvimgcodecBackendKind_t backend_kind; /**< What kind of backend this encoder is using */

        /**
         * @brief Creates encoder.
         * 
         * @param instance [in] Pointer to nvimgcodecEncoderDesc_t instance.
         * @param encoder [in/out] Points where to return handle to created encoder.
         * @param exec_params [in] Points an execution parameters.
         * @param options [in] String with optional, space separated, list of parameters for encoders, in format 
         *                     "<encoder_id>:<parameter_name>=<parameter_value>".
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*create)(
            void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

        /** 
         * Destroys encoder.
         * 
         * @param encoder [in] Encoder handle to destroy.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*destroy)(nvimgcodecEncoder_t encoder);

        /**
         * @brief Checks whether encoder can encode given image to code stream with provided parameters.
         * 
         * @param encoder [in] Encoder handle.
         * @param code_stream [in] Encoded stream.
         * @param image [in] Image descriptor.
         * @param params [in] Encode parameters which will be used with check.
         * @param thread_idx [in] Index of the caller thread (can be from 0 to the executor's number of threads, or -1 for non-threaded execution)
         * @return nvimgcodecProcessingStatus_t - Processing status
         */
        nvimgcodecProcessingStatus_t (*canEncode)(
            nvimgcodecEncoder_t encoder,
            const nvimgcodecCodeStreamDesc_t* code_stream,
            const nvimgcodecImageDesc_t* image,
            const nvimgcodecEncodeParams_t* params,
            int thread_idx);

        /**
         * @brief Encode given image to code stream with provided parameters.
         * 
         * @param encoder [in] Encoder handle.
         ** @param image [in] Image descriptor.
         * @param code_stream [in] Encoded stream.
         * @param params [in] Encode parameters.
         * @param thread_idx [in] Index of the caller thread (can be from 0 to the executor's number of threads, or -1 for non-threaded execution)
         * @return nvimgcodecProcessingStatus_t - Processing status
         */
        nvimgcodecStatus_t (*encode)(
            nvimgcodecEncoder_t encoder,
            const nvimgcodecCodeStreamDesc_t* code_stream,
            const nvimgcodecImageDesc_t* image,
            const nvimgcodecEncodeParams_t* params,
            int thread_idx);
    } nvimgcodecEncoderDesc_t;

    /**
     * Decoder description.
    */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                       /**< Decoder description instance pointer which will be passed back in functions */
        const char* id;                       /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec;                    /**< Codec name e.g. jpeg2000 */
        nvimgcodecBackendKind_t backend_kind; /**< Backend kind */

        /**
         * @brief Creates decoder.
         * 
         * @param instance [in] Pointer to nvimgcodecDecoderDesc_t instance.
         * @param decoder [in/out] Points where to return handle to created decoder.
         * @param exec_params [in] Points an execution parameters.
         * @param options [in] String with optional, space separated, list of parameters for decoders, in format 
         *                     "<encoder_id>:<parameter_name>=<parameter_value>".
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*create)(
            void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

        /**
         * @brief Destroys decoder.
         * 
         * @param decoder [in] Decoder handle to destroy.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*destroy)(nvimgcodecDecoder_t decoder);

        /**
         * @brief Checks whether decoder can decode given code stream to image with provided parameters.
         * 
         * @param decoder [in] Decoder handle to use for check.
         * @param info [in] Image information, including requested format.
         * @param code_stream [in] Encoded stream.
         * @param params [in] Decode parameters which will be used with check.
         * @param thread_idx [in] Index of the caller thread (can be from 0 to the executor's number of threads, or -1 for non-threaded execution)
         * @return nvimgcodecStatus_t
         */
        nvimgcodecProcessingStatus_t (*canDecode)(
            nvimgcodecDecoder_t decoder,
            const nvimgcodecImageDesc_t* image,
            const nvimgcodecCodeStreamDesc_t* code_stream,
            const nvimgcodecDecodeParams_t* params,
            int thread_idx);

        /**
         * @brief Decode given code stream to image with provided parameters.
         * 
         * @param decoder [in] Decoder handle to use for decoding.
         * @param image [in/out] Image descriptor.
         * @param code_stream [in] Encoded stream.
         * @param params [in] Decode parameters.
         * @param thread_idx [in] Index of the caller thread (can be from 0 to the executor's number of threads, or -1 for non-threaded execution)
         * @return nvimgcodecStatus_t
         */
        nvimgcodecStatus_t (*decode)(
            nvimgcodecDecoder_t decoder,
            const nvimgcodecImageDesc_t* image,
            const nvimgcodecCodeStreamDesc_t* code_stream,
            const nvimgcodecDecodeParams_t* params,
            int thread_idx);

        /**
         * @brief Decode given batch of code streams to images with provided parameters.
         * @param decoder [in] Decoder handle to use for decoding.
         * @param images [in/out] Pointer to array of pointers of batch size with image descriptors.
         * @param code_streams [in] Pointer to array of batch size of pointers to encoded stream instances.
         * @param batch_size [in] Number of items in batch to decode.
         * @param params [in] Decode parameters.
         * @param thread_idx [in] Index of the caller thread (can be from 0 to the executor's number of threads, or -1 for non-threaded execution)
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*decodeBatch)(
            nvimgcodecDecoder_t decoder,
            const nvimgcodecImageDesc_t** images,
            const nvimgcodecCodeStreamDesc_t** code_streams,
            int batch_size,
            const nvimgcodecDecodeParams_t* params,
            int thread_idx);

        /**
         * @brief  Retrieve preferred minibatch size. The library will try to use batch sizes that are multiples of this value.
         * @param decoder [in] Decoder handle to use.
         * @param batch_size [out] Preferred minibatch size.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*getMiniBatchSize)(nvimgcodecDecoder_t decoder, int* batch_size);

    } nvimgcodecDecoderDesc_t;

    /**
     * @brief Defines decoder or encoder priority in codec.
     * 
     * For each codec there can be more decoders and encoders registered. Every decoder and encoder is registered with defined priority.
     * Decoding process starts with selecting highest priority decoder and checks whether it can decode particular code stream. In case
     * decoding could not be handled by selected decoder, there is fallback mechanism which selects next in priority decoder. There can be 
     * more decoders registered with the same priority. In such case decoders with the same priority are selected in order of registration.
     */
    typedef enum
    {
        NVIMGCODEC_PRIORITY_HIGHEST = 0,
        NVIMGCODEC_PRIORITY_VERY_HIGH = 100,
        NVIMGCODEC_PRIORITY_HIGH = 200,
        NVIMGCODEC_PRIORITY_NORMAL = 300,
        NVIMGCODEC_PRIORITY_LOW = 400,
        NVIMGCODEC_PRIORITY_VERY_LOW = 500,
        NVIMGCODEC_PRIORITY_LOWEST = 1000,
        NVIMGCODEC_PRIORITY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcodecPriority_t;

    /**
     * @brief Pointer to logging function.
     * 
     * @param instance [in] Plugin framework instance pointer
     * @param message_severity [in] Message severity e.g. error or warning.
     * @param message_category [in]  Message category e.g. general or performance related.
     * @param data [in] Debug message data i.e. message string, status, codec etc.
     */
    typedef nvimgcodecStatus_t (*nvimgcodecLogFunc_t)(void* instance, const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data);

    /**
     * @brief Plugin Framework
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance;           /**< Plugin framework instance pointer which will be passed back in functions */
        const char* id;           /**< Plugin framework named identifier e.g. nvImageCodec */
        uint32_t version;         /**< Plugin framework version. */
        uint32_t ext_api_version; /**< The nvImageCodec extension API version. */
        uint32_t cudart_version;  /**< The version of CUDA Runtime with which plugin framework was built. */
        nvimgcodecLogFunc_t log;  /**< Pointer to logging function. @see nvimgcodecLogFunc_t */

        /**
         * @brief Registers encoder plugin.
         * 
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to encoder description.
         * @param priority [in] Priority of encoder. @see nvimgcodecPriority_t
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*registerEncoder)(void* instance, const nvimgcodecEncoderDesc_t* desc, float priority);

        /**
         * @brief Unregisters encoder plugin.
         *
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to encoder description to unregister.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*unregisterEncoder)(void* instance, const nvimgcodecEncoderDesc_t* desc);

        /**
         * @brief Registers decoder plugin.
         * 
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to decoder description.
         * @param priority [in] Priority of decoder. @see nvimgcodecPriority_t
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*registerDecoder)(void* instance, const nvimgcodecDecoderDesc_t* desc, float priority);

        /**
         * @brief Unregisters decoder plugin.
         *
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to decoder description to unregister.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*unregisterDecoder)(void* instance, const nvimgcodecDecoderDesc_t* desc);

        /**
         * @brief Registers parser plugin.
         * 
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to parser description.
         * @param priority [in] Priority of decoder. @see nvimgcodecPriority_t
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*registerParser)(void* instance, const nvimgcodecParserDesc_t* desc, float priority);

        /**
         * @brief Unregisters parser plugin.
         *
         * @param instance [in] Pointer to nvimgcodecFrameworkDesc_t instance.
         * @param desc [in] Pointer to parser description to unregister.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
         */
        nvimgcodecStatus_t (*unregisterParser)(void* instance, const nvimgcodecParserDesc_t* desc);

    } nvimgcodecFrameworkDesc_t;

    /**
     * @brief Extension description
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        void* instance;           /**< Extension instance pointer which will be passed back in functions */
        const char* id;           /**< Extension named identifier e.g. nvjpeg_ext */
        uint32_t version;         /**< Extension version. Used when registering extension to check if there are newer.*/
        uint32_t ext_api_version; /**< The version of nvImageCodec extension API with which the extension was built. */

        /**
         * @brief Creates extension.
         * 
         * @param instance [in] Pointer to nvimgcodecExtensionDesc_t instance.
         * @param extension [in/out] Points where to return handle to created extension. 
         * @param framework [in] Pointer to framework description which can be use to register plugins.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*create)(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework);

        /** 
         * Destroys extension.
         * 
         * @param extension [in] Extension handle to destroy.
         * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
        */
        nvimgcodecStatus_t (*destroy)(nvimgcodecExtension_t extension);
    } nvimgcodecExtensionDesc_t;

    /**
     * @brief Extension module entry function type
     * 
     * @param ext_desc [in/out] Points a nvimgcodecExtensionDesc_t handle in which the extension description is returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    typedef nvimgcodecStatus_t (*nvimgcodecExtensionModuleEntryFunc_t)(nvimgcodecExtensionDesc_t* ext_desc);

    /**
     * @brief Extension shared module exported entry function.
     * 
     * @param ext_desc [in/out] Points a nvimgcodecExtensionDesc_t handle in which the extension description is returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecExtensionModuleEntry(nvimgcodecExtensionDesc_t* ext_desc);

    /**
     * @brief Provides nvImageCodec library properties.
     * 
     * @param properties [in/out] Points a nvimgcodecProperties_t handle in which the properties are returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecGetProperties(nvimgcodecProperties_t* properties);

    /** 
     * @brief The nvImageCodec library instance create information structure.
     */
    typedef struct
    {
        nvimgcodecStructureType_t struct_type; /**< The type of the structure. */
        size_t struct_size;                    /**< The size of the structure, in bytes. */
        void* struct_next;                     /**< Is NULL or a pointer to an extension structure type. */

        int load_builtin_modules;           /**< Load default modules. Valid values 0 or 1. */
        int load_extension_modules;         /**< Discover and load extension modules on start. Valid values 0 or 1. */
        const char* extension_modules_path; /**< There may be several paths separated by ':' on Linux or ';' on Windows */
        int create_debug_messenger;         /**< Create debug messenger during instance creation. Valid values 0 or 1. */
        /** Pointer to description to use when creating debug messenger. If NULL, default internal description will be used,
         *  together with following message_severity and message_category fields. */
        const nvimgcodecDebugMessengerDesc_t* debug_messenger_desc;
        uint32_t message_severity; /**< Severity for default debug messenger */
        uint32_t message_category; /**< Message category for default debug messenger */
    } nvimgcodecInstanceCreateInfo_t;

    /**
     * @brief Creates an instance of the library using the input arguments.
     * 
     * @param instance [in/out] Points a nvimgcodecInstance_t handle in which the resulting instance is returned.
     * @param create_info [in] Pointer to a nvimgcodecInstanceCreateInfo_t structure controlling creation of the instance.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecInstanceCreate(
        nvimgcodecInstance_t* instance, const nvimgcodecInstanceCreateInfo_t* create_info);

    /**
     * @brief Destroys the nvImageCodec library instance.
     * 
     * @param instance [in] The library instance handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecInstanceDestroy(nvimgcodecInstance_t instance);

    /**
     * @brief Creates library extension.
     *  
     * @param instance [in] The library instance handle the extension will be used with.
     * @param extension [in/out] Points a nvimgcodecExtension_t handle in which the resulting extension is returned.
     * @param extension_desc [in] Pointer to a nvimgcodecExtensionDesc_t structure which defines extension to create.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecExtensionCreate(
        nvimgcodecInstance_t instance, nvimgcodecExtension_t* extension, nvimgcodecExtensionDesc_t* extension_desc);

    /**
     * @brief Destroys library extension.
     * 
     * @param extension [in] The extension handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecExtensionDestroy(nvimgcodecExtension_t extension);

    /**
     * @brief Creates a debug messenger.
     *  
     * @param instance [in] The library instance handle the messenger will be used with.
     * @param dbg_messenger [in/out] Points a nvimgcodecDebugMessenger_t handle in which the resulting debug messenger is returned.
     * @param messenger_desc [in]  Pointer to nvimgcodecDebugMessengerDesc_t structure which defines debug messenger to create.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDebugMessengerCreate(
        nvimgcodecInstance_t instance, nvimgcodecDebugMessenger_t* dbg_messenger, const nvimgcodecDebugMessengerDesc_t* messenger_desc);

    /**
     * @brief Destroys debug messenger.
     * 
     * @param dbg_messenger [in] The debug messenger handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDebugMessengerDestroy(nvimgcodecDebugMessenger_t dbg_messenger);

    /**
     * @brief Waits for processing items to be finished.
     *  
     * @param future [in] Handle to future object created by decode or encode functions.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     * @warning Please note that when future is ready, it only means that all host work was done and it can be that
     *          some work was scheduled to be executed on device (depending on codec). To further synchronize work on 
     *          device, there is cuda_stream field available in nvimgcodecImageInfo_t which can be used to specify 
     *          cuda_stream to synchronize with.
     * @see  nvimgcodecImageInfo_t cuda_stream field.
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecFutureWaitForAll(nvimgcodecFuture_t future);

    /**
     * @brief Destroys future.
     * 
     * @param future [in] The future handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecFutureDestroy(nvimgcodecFuture_t future);

    /**
     * @brief Receives processing statuses of batch items scheduled for decoding or encoding 
     * 
     * @param future [in] The future handle returned by decode or encode function for given batch items.
     * @param processing_status [in/out] Points a nvimgcodecProcessingStatus_t handle in which the processing statuses is returned.
     * @param size [in/out]  Points a size_t in which the size of processing statuses returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecFutureGetProcessingStatus(
        nvimgcodecFuture_t future, nvimgcodecProcessingStatus_t* processing_status, size_t* size);

    /**
     * @brief Creates Image which wraps sample buffer together with format information.
     * 
     * @param instance [in] The library instance handle the image will be used with.
     * @param image [in/out] Points a nvimgcodecImage_t handle in which the resulting image is returned.
     * @param image_info [in] Points a nvimgcodecImageInfo_t struct which describes sample buffer together with format.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecImageCreate(
        nvimgcodecInstance_t instance, nvimgcodecImage_t* image, const nvimgcodecImageInfo_t* image_info);

    /**
     * @brief Destroys image.
     * 
     * @param image [in] The image handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecImageDestroy(nvimgcodecImage_t image);

    /**
     * @brief Retrieves image information from provided opaque image object. 
     *  
     * @param image [in] The image handle to retrieve information from.
     * @param image_info [in/out] Points a nvimgcodecImageInfo_t handle in which the image information is returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecImageGetImageInfo(nvimgcodecImage_t image, nvimgcodecImageInfo_t* image_info);

    /**
     * @brief Creates code stream which wraps file source of compressed data 
     *  
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcodecCodeStream_t handle in which the resulting code stream is returned.
     * @param file_name [in] File name with compressed image data to wrap.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromFile(
        nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name);

    /**
     * @brief Creates code stream which wraps host memory source of compressed data.
     * 
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcodecCodeStream_t handle in which the resulting code stream is returned.
     * @param data [in] Pointer to buffer with compressed data.
     * @param length [in] Length of compressed data in provided buffer.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromHostMem(
        nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const unsigned char* data, size_t length);

    /**
     * @brief Creates code stream which wraps file sink for compressed data with given format.
     * 
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcodecCodeStream_t handle in which the resulting code stream is returned.
     * @param file_name [in] File name sink for compressed image data to wrap.
     * @param image_info [in] Points a nvimgcodecImageInfo_t struct which describes output image format.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamCreateToFile(
        nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name, const nvimgcodecImageInfo_t* image_info);

    /**
     * @brief Function type to resize and provide host buffer.
     * 
     * @param ctx [in] Pointer to context provided together with function.
     * @param req_size [in] Requested size of buffer.
     * @return Pointer to requested buffer.
     * 
     * @note This function can be called multiple times and requested size can be lower at the end so buffer can be shrinked.
     */
    typedef unsigned char* (*nvimgcodecResizeBufferFunc_t)(void* ctx, size_t req_size);

    /**
     * @brief Creates code stream which wraps host memory sink for compressed data with given format.
     *  
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcodecCodeStream_t handle in which the resulting code stream is returned.
     * @param ctx [in] Pointer to user defined context with which get buffer function will be called back.
     * @param resize_buffer_func [in] Points a nvimgcodecResizeBufferFunc_t function handle which will be used to resize and providing host output buffer.
     * @param image_info [in] Points a nvimgcodecImageInfo_t struct which describes output image format.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamCreateToHostMem(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream,
        void* ctx, nvimgcodecResizeBufferFunc_t resize_buffer_func, const nvimgcodecImageInfo_t* image_info);

    /**
     * @brief Destroys code stream.
     * 
     * @param code_stream [in] The code stream handle to destroy 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamDestroy(nvimgcodecCodeStream_t code_stream);

    /**
     * @brief Retrieves compressed image information from code stream. 
     *  
     * @param code_stream [in] The code stream handle to retrieve information from.
     * @param image_info [in/out] Points a nvimgcodecImageInfo_t handle in which the image information is returned.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes}
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecCodeStreamGetImageInfo(
        nvimgcodecCodeStream_t code_stream, nvimgcodecImageInfo_t* image_info);

    /**
     * @brief Creates generic image decoder.
     * 
     * @param instance  [in] The library instance handle the decoder will be used with.
     * @param decoder  [in/out] Points a nvimgcodecDecoder_t handle in which the decoder is returned.
     * @param exec_params [in] Points an execution parameters.
     * @param options [in] String with optional space separated list of parameters for specific decoders in format 
     *                     "<decoder_id>:<parameter_name>=<parameter_value>". For example  "nvjpeg:fancy_upsampling=1"
     * @return nvimgcodecStatus_t - An error code as specified in
     {
        @link nvimgcodecStatus_t API Return Status Codes
     }
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCreate(
        nvimgcodecInstance_t instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    /**
     * @brief Destroys decoder.
     * 
     * @param decoder [in] The decoder handle to destroy
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderDestroy(nvimgcodecDecoder_t decoder);

    /**
     * @brief Checks if decoder can decode provided code stream to given output images with specified parameters.
     *  
     * @param decoder [in] The decoder handle to use for checks. 
     * @param streams [in] Pointer to input nvimgcodecCodeStream_t array to check decoding with.
     * @param images [in] Pointer to output nvimgcodecImage_t array to check decoding with.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcodecDecodeParams_t struct to check decoding with.
     * @param processing_status [in/out] Points a nvimgcodecProcessingStatus_t handle in which the processing statuses is returned.
     * @param force_format [in] Valid values 0 or 1. If 1 value, and high priority codec does not support provided format it will
     *                          fallback to lower priority codec for further checks. For 0 value, when high priority codec does not
     *                          support provided format or parameters but it can process input in general, it will stop check and
     *                          return processing status with flags which shows what format or parameters need to be changed to 
     *                          avoid fallback to lower priority codec. 
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCanDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams,
        const nvimgcodecImage_t* images, int batch_size, const nvimgcodecDecodeParams_t* params,
        nvimgcodecProcessingStatus_t* processing_status, int force_format);

    /**
     * @brief Decode batch of provided code streams to given output images with specified parameters.
     *  
     * @param decoder [in] The decoder handle to use for decoding. 
     * @param streams [in] Pointer to input nvimgcodecCodeStream_t array to decode.
     * @param images [in] Pointer to output nvimgcodecImage_t array to decode to.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcodecDecodeParams_t struct to decode with.
     * @param future [in/out] Points a nvimgcodecFuture_t handle in which the future is returned. 
     *               The future object can be used to waiting and getting processing statuses.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     * 
     * @see nvimgcodecFutureGetProcessingStatus
     * @see nvimgcodecFutureWaitForAll
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams,
        const nvimgcodecImage_t* images, int batch_size, const nvimgcodecDecodeParams_t* params, nvimgcodecFuture_t* future);

    /**
     * @brief Creates generic image encoder.
     *  
     * @param instance [in] The library instance handle the encoder will be used with.
     * @param encoder [in/out] Points a nvimgcodecEncoder_t handle in which the decoder is returned.
     * @param exec_params [in] Points an execution parameters.
     * @param options [in] String with optional, space separated, list of parameters for specific encoders, in format 
     *                     "<encoder_id>:<parameter_name>=<parameter_value>."
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCreate(
        nvimgcodecInstance_t instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    /**
     * @brief Destroys encoder.
     *  
     * @param encoder [in] The encoder handle to destroy
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderDestroy(nvimgcodecEncoder_t encoder);

    /**
     * @brief Checks if encoder can encode provided images to given output code streams with specified parameters.
     *  
     * @param encoder [in] The encoder handle to use for checks. 
     * @param images [in] Pointer to input nvimgcodecImage_t array to check encoding with.
     * @param streams [in] Pointer to output nvimgcodecCodeStream_t array to check encoding with.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcodecEncodeParams_t struct to check decoding with.
     * @param processing_status [in/out] Points a nvimgcodecProcessingStatus_t handle in which the processing statuses is returned.
     * @param force_format [in] Valid values 0 or 1. If 1 value, and high priority codec does not support provided format it will 
     *                          fallback to lower priority codec for further checks. For 0 value, when high priority codec does not
     *                          support provided format or parameters but it can process input in general, it will stop check and
     *                          return processing status with flags which shows what format or parameters need to be changed to 
     *                          avoid fallback to lower priority codec.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCanEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images,
        const nvimgcodecCodeStream_t* streams, int batch_size, const nvimgcodecEncodeParams_t* params,
        nvimgcodecProcessingStatus_t* processing_status, int force_format);

    /**
     * @brief Encode batch of provided images to given output code streams with specified parameters.
     * 
     * @param encoder [in] The encoder handle to use for encoding. 
     * @param images [in] Pointer to input nvimgcodecImage_t array to encode.
     * @param streams [in] Pointer to output nvimgcodecCodeStream_t array to encode to.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to  nvimgcodecEncodeParams_t struct to encode with.
     * @param future  [in/out] Points a nvimgcodecFuture_t handle in which the future is returned. 
     *                 The future object can be used to waiting and getting processing statuses.
     * @return nvimgcodecStatus_t - An error code as specified in {@link nvimgcodecStatus_t API Return Status Codes} 
     * 
     * @see nvimgcodecFutureGetProcessingStatus
     * @see nvimgcodecFutureWaitForAll
     */
    NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images,
        const nvimgcodecCodeStream_t* streams, int batch_size, const nvimgcodecEncodeParams_t* params, nvimgcodecFuture_t* future);

#if defined(__cplusplus)
}
#endif

#endif
