/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

//Compatibility: CV-CUDA v0.4.0 Beta

#pragma once

#include <map>
#include <tuple>

#include <nvimgcodec.h>

#include <nvcv/ImageData.hpp>
#include <nvcv/ImageFormat.hpp>

namespace nvimgcodec { namespace adapter { namespace nvcv { namespace {

#define CHECK_NVCV(call)                                     \
    {                                                        \
        NVCVStatus _e = (call);                              \
        if (_e != NVCV_SUCCESS) {                            \
            std::stringstream _error;                        \
            _error << "NVCV Types failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());          \
        }                                                    \
    }

constexpr auto ext2loc_buffer_kind(NVCVImageBufferType in_kind)
{
    switch (in_kind) {
    case NVCV_IMAGE_BUFFER_NONE:
        return NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED;
    case NVCV_IMAGE_BUFFER_STRIDED_CUDA:
        return NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    case NVCV_IMAGE_BUFFER_STRIDED_HOST:
        return NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        return NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED;
    default:
        return NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED;
    }
}

constexpr auto ext2loc_color_spec(NVCVColorSpec color_spec, NVCVColorModel color_model, int num_total_channels)
{
    if (color_model == NVCV_COLOR_MODEL_YCbCr && color_spec == NVCV_COLOR_SPEC_sYCC)
        return NVIMGCODEC_COLORSPEC_SYCC;
    else if (color_model == NVCV_COLOR_MODEL_RGB && color_spec == NVCV_COLOR_SPEC_sRGB)
        return NVIMGCODEC_COLORSPEC_SRGB;
    else if (color_model == NVCV_COLOR_MODEL_YCbCr && num_total_channels == 1)
        return NVIMGCODEC_COLORSPEC_GRAY;    
    else
        return NVIMGCODEC_COLORSPEC_UNSUPPORTED;    
};

constexpr auto ext2loc_css(NVCVChromaSubsampling in)
{
    switch (in) {
    case NVCV_CSS_444:
        return NVIMGCODEC_SAMPLING_444;
    case NVCV_CSS_422:
        return NVIMGCODEC_SAMPLING_422;
    case NVCV_CSS_420:
        return NVIMGCODEC_SAMPLING_420;
    case NVCV_CSS_440:
       return NVIMGCODEC_SAMPLING_440;
    case NVCV_CSS_411:
        return NVIMGCODEC_SAMPLING_411;
    //case NVCV_CSS_410:
    //    return NVIMGCODEC_SAMPLING_410;
    // case :
    //    return NVIMGCODEC_SAMPLING_GRAY; //TODO
    case NVCV_CSS_410R:
      return NVIMGCODEC_SAMPLING_410V;
    default:
        return NVIMGCODEC_SAMPLING_UNSUPPORTED;
    }
}

constexpr auto ext2loc_sample_type(NVCVDataKind data_kind, int32_t bpp)
{
    switch (data_kind) {
    case NVCV_DATA_KIND_SIGNED:
        if (bpp <= 8)
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
        else if ((bpp > 8) && (bpp <= 16))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
        else
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;

    case NVCV_DATA_KIND_UNSIGNED:
        if (bpp <= 8)
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        else if ((bpp > 8) && (bpp <= 16))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
        else
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;

    case NVCV_DATA_KIND_FLOAT:
        if (bpp <= 16)
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
        else
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;

    case NVCV_DATA_KIND_COMPLEX:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    default:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

constexpr auto ext2loc_sample_type(NVCVDataType data_type)
{
    switch (data_type) {
    case NVCV_DATA_TYPE_S8:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
    case NVCV_DATA_TYPE_U8:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    case NVCV_DATA_TYPE_S16:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
    case NVCV_DATA_TYPE_U16:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    case NVCV_DATA_TYPE_S32:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
    case NVCV_DATA_TYPE_U32:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
    case NVCV_DATA_TYPE_S64:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
    case NVCV_DATA_TYPE_U64:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
    //TODO
    //case NVCV_DATA_TYPE_F16:
    //    return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
    case NVCV_DATA_TYPE_F32:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
    case NVCV_DATA_TYPE_F64:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
    default:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

constexpr auto loc2ext_dtype(nvimgcodecSampleDataType_t in)
{
    switch (in) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED:
        return NVCV_DATA_TYPE_NONE;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
        return NVCV_DATA_TYPE_S8;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        return NVCV_DATA_TYPE_U8;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        return NVCV_DATA_TYPE_S16;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        return NVCV_DATA_TYPE_U16;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
        return NVCV_DATA_TYPE_S32;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
        return NVCV_DATA_TYPE_U32;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        return NVCV_DATA_TYPE_S64;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
        return NVCV_DATA_TYPE_U64;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16:
        return NVCV_DATA_TYPE_NONE; //TODO
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        return NVCV_DATA_TYPE_F32;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
        return NVCV_DATA_TYPE_F64;
    default:
        return NVCV_DATA_TYPE_NONE;
    }
}

constexpr auto ext2loc_sample_format(int32_t num_planes, NVCVSwizzle swizzle, NVCVColorSpec color_spec)
{
    if (color_spec == NVCV_COLOR_SPEC_sRGB) {
        if (swizzle == NVCV_SWIZZLE_XYZ0)
            return num_planes > 1 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        else if (swizzle == NVCV_SWIZZLE_ZYX0)
            return num_planes > 1 ? NVIMGCODEC_SAMPLEFORMAT_P_BGR : NVIMGCODEC_SAMPLEFORMAT_I_BGR;
        else
            return NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED;
    } else if (color_spec == NVCV_COLOR_SPEC_sYCC) {
        if (swizzle == NVCV_SWIZZLE_XYZ0)
            return num_planes > 1 ? NVIMGCODEC_SAMPLEFORMAT_P_YUV : NVIMGCODEC_SAMPLEFORMAT_P_Y;
        else
            return NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED;
    } else {
        if (swizzle == NVCV_DETAIL_MAKE_SWZL(1, 1, 1, 1)) //TODO confirm this
            return num_planes > 1 ? NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED : NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
        else
            return NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED;
    }
}

constexpr auto loc2ext_buffer_kind(nvimgcodecImageBufferKind_t in_kind)
{
    switch (in_kind) {
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST:
        return NVCV_IMAGE_BUFFER_STRIDED_HOST;
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE:
        return NVCV_IMAGE_BUFFER_STRIDED_CUDA;
    default:
        return NVCV_IMAGE_BUFFER_NONE;
    }
}

constexpr auto loc2ext_css(nvimgcodecChromaSubsampling_t in)
{
    switch (in) {
    case NVIMGCODEC_SAMPLING_444:
        return NVCV_CSS_444;
    case NVIMGCODEC_SAMPLING_422:
        return NVCV_CSS_422;
    case NVIMGCODEC_SAMPLING_420:
        return NVCV_CSS_420;
    case NVIMGCODEC_SAMPLING_440:
        return NVCV_CSS_440;
    case NVIMGCODEC_SAMPLING_411:
        return NVCV_CSS_411;
    case NVIMGCODEC_SAMPLING_410:
        return NVCV_CSS_410;
    case NVIMGCODEC_SAMPLING_GRAY:
        return NVCV_CSS_NONE;
    case NVIMGCODEC_SAMPLING_410V:
        return NVCV_CSS_410R;
    default:
        return NVCV_CSS_NONE;
    }
}

constexpr auto loc2ext_color_spec(nvimgcodecColorSpec_t in)
{
    switch (in) {
    case NVIMGCODEC_COLORSPEC_UNKNOWN:
        return NVCV_COLOR_SPEC_UNDEFINED;
    case NVIMGCODEC_COLORSPEC_SRGB:
        return NVCV_COLOR_SPEC_sRGB;
    case NVIMGCODEC_COLORSPEC_SYCC:
        return NVCV_COLOR_SPEC_sYCC;
    case NVIMGCODEC_COLORSPEC_GRAY:
    case NVIMGCODEC_COLORSPEC_CMYK:
    case NVIMGCODEC_COLORSPEC_YCCK:
        return NVCV_COLOR_SPEC_UNDEFINED;
    case NVIMGCODEC_COLORSPEC_UNSUPPORTED:
        return NVCV_COLOR_SPEC_UNDEFINED;
    default:
        return NVCV_COLOR_SPEC_UNDEFINED;
    }
}

constexpr auto loc2ext_color_model(nvimgcodecColorSpec_t in)
{
    switch (in) {
    case NVIMGCODEC_COLORSPEC_UNKNOWN:
        return NVCV_COLOR_MODEL_UNDEFINED;
    case NVIMGCODEC_COLORSPEC_SRGB:
        return NVCV_COLOR_MODEL_RGB;
    case NVIMGCODEC_COLORSPEC_GRAY:
        return NVCV_COLOR_MODEL_YCbCr;
    case NVIMGCODEC_COLORSPEC_CMYK:
        return NVCV_COLOR_MODEL_CMYK;
    case NVIMGCODEC_COLORSPEC_YCCK:
        return NVCV_COLOR_MODEL_YCCK;
    case NVIMGCODEC_COLORSPEC_SYCC:
        return NVCV_COLOR_MODEL_YCbCr;
    case NVIMGCODEC_COLORSPEC_UNSUPPORTED:
        return NVCV_COLOR_MODEL_UNDEFINED;
    default:
        return NVCV_COLOR_MODEL_UNDEFINED;
    }
}

constexpr auto loc2ext_data_kind(nvimgcodecSampleDataType_t in)
{
    switch (in) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        return NVCV_DATA_KIND_UNSIGNED;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        return NVCV_DATA_KIND_SIGNED;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        return NVCV_DATA_KIND_FLOAT;
    default:
        return NVCV_DATA_KIND_UNSIGNED;
    }
}

constexpr auto loc2ext_swizzle(nvimgcodecSampleFormat_t in)
{
    switch (in) {
    case NVIMGCODEC_SAMPLEFORMAT_UNKNOWN:
    case NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED:
        return NVCV_SWIZZLE_0000;
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        return NVCV_SWIZZLE_XYZW;
    case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        return NVCV_SWIZZLE_XYZW;
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        return NVCV_SWIZZLE_XYZ0;
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        return NVCV_SWIZZLE_XYZ0;
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        return NVCV_SWIZZLE_ZYX0;
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        return NVCV_SWIZZLE_ZYX0;
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        return NVCV_SWIZZLE_X000;
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
        return NVCV_SWIZZLE_XYZ0;
    default:
        return NVCV_SWIZZLE_0000;
    }
}

constexpr unsigned char loc2ext_bpp(nvimgcodecSampleDataType_t in)
{
    switch (in) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED:
        return 0;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        return 8;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        return 16;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        return 32;
    default:
        return 0;
    }
}

constexpr auto loc2ext_packing(const nvimgcodecImageInfo_t& image_info)
{
    switch (image_info.sample_format) {
    case NVIMGCODEC_SAMPLEFORMAT_UNKNOWN:
    case NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED:
        return std::make_tuple(NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(
            NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type) * image_info.plane_info[0].num_channels,
                image_info.plane_info[0].num_channels));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        NVCVPacking packing1 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[1].sample_type), 1));
        NVCVPacking packing2 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[2].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), std::move(packing1), std::move(packing2), NVCV_PACKING_0);
    }
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type) * 3, 3));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCODEC_SAMPLEFORMAT_P_Y: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    default:
        return std::make_tuple(NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
}

nvimgcodecStatus_t ImageData2Imageinfo(nvimgcodecImageInfo_t* image_info, const NVCVImageData& image_data)
{
    try {
        image_info->buffer_kind = ext2loc_buffer_kind(image_data.bufferType);
        if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        NVCVDataKind data_kind;
        CHECK_NVCV(nvcvImageFormatGetDataKind(image_data.format, &data_kind));

        NVCVSwizzle swizzle;
        CHECK_NVCV(nvcvImageFormatGetSwizzle(image_data.format, &swizzle));
        if (swizzle == NVCV_SWIZZLE_0000)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        

        const NVCVImageBufferStrided& strided = image_data.buffer.strided;
        image_info->num_planes = strided.numPlanes;
        auto ptr = strided.planes[0].basePtr;
        if (ptr == nullptr)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        image_info->buffer = reinterpret_cast<void*>(ptr);

        int num_total_channels = 0;
        for (int32_t p = 0; p < strided.numPlanes; ++p) {
            if (strided.planes[p].basePtr != ptr) //Accept only contignous memory
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            image_info->plane_info[p].width = strided.planes[p].width;
            image_info->plane_info[p].height = strided.planes[p].height;
            image_info->plane_info[p].row_stride = strided.planes[p].rowStride;
            int32_t bpp;
            CHECK_NVCV(nvcvImageFormatGetPlaneBitsPerPixel(image_data.format, p, &bpp));
            image_info->plane_info[p].sample_type = ext2loc_sample_type(data_kind, bpp);
            image_info->plane_info[p].precision = bpp;
            if (image_info->plane_info[p].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            int32_t num_channels;
            
            CHECK_NVCV(nvcvImageFormatGetPlaneNumChannels(image_data.format, p, &num_channels));
            num_total_channels += num_channels;
            NVCVExtraChannelInfo exChannelInfo;
            CHECK_NVCV(nvcvImageFormatGetExtraChannelInfo(image_data.format, &exChannelInfo));
            // cvcuda supports different data kind for regular and extra channels but nvimgcodecs does not.
            if (p == 0 && ((bpp != exChannelInfo.bitsPerPixel) || (data_kind != exChannelInfo.datakind))) 
                return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
            // TODO uncomment this later when alpha type is supported in nvimgcodecs
            //NVCVAlphaType alpha_type; 
            //CHECK_NVCV(nvcvImageFormatGetAlphaType(image_data.format, &alpha_type));
            
            image_info->plane_info[p].num_channels = num_channels + (p == 0) ? exChannelInfo.numChannels : 0;
            ptr += image_info->plane_info[p].height * image_info->plane_info[p].row_stride;
        }

        NVCVColorSpec color_spec;
        NVCVColorModel color_model;
        CHECK_NVCV(nvcvImageFormatGetColorSpec(image_data.format, &color_spec));
        CHECK_NVCV(nvcvImageFormatGetColorModel(image_data.format, &color_model));        
        image_info->color_spec = ext2loc_color_spec(color_spec, color_model, num_total_channels);
        if (image_info->color_spec == NVIMGCODEC_COLORSPEC_UNSUPPORTED)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        NVCVChromaSubsampling css;
        CHECK_NVCV(nvcvImageFormatGetChromaSubsampling(image_data.format, &css));
        image_info->chroma_subsampling = ext2loc_css(css);
        if (image_info->chroma_subsampling == NVIMGCODEC_SAMPLING_UNSUPPORTED)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        image_info->sample_format = ext2loc_sample_format(strided.numPlanes, swizzle, color_spec);
        if (image_info->sample_format == NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t ImageInfo2ImageData(NVCVImageData* image_data, const nvimgcodecImageInfo_t& image_info)
{
    image_data->bufferType = loc2ext_buffer_kind(image_info.buffer_kind);
    if (image_data->bufferType == NVCV_IMAGE_BUFFER_NONE) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    NVCVImageBufferStrided& strided = image_data->buffer.strided;
    strided.numPlanes = image_info.num_planes;
    int num_total_channels = 0;    
    NVCVByte* ptr = reinterpret_cast<NVCVByte*>(image_info.buffer);
    for (int32_t p = 0; p < strided.numPlanes; ++p) {
        strided.planes[p].width = image_info.plane_info[p].width;
        strided.planes[p].height = image_info.plane_info[p].height;
        strided.planes[p].rowStride = image_info.plane_info[p].row_stride;
        strided.planes[p].basePtr = ptr;
        ptr += image_info.plane_info[p].height * image_info.plane_info[p].row_stride;
        num_total_channels += image_info.plane_info[p].num_channels;
    }

    if (image_info.color_spec == NVIMGCODEC_COLORSPEC_UNSUPPORTED || image_info.color_spec == NVIMGCODEC_COLORSPEC_UNKNOWN)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    auto color_spec = loc2ext_color_spec(image_info.color_spec);            
    auto color_model = loc2ext_color_model(image_info.color_spec);    
    auto css = loc2ext_css(image_info.chroma_subsampling);
    auto data_kind = loc2ext_data_kind(image_info.plane_info[0].sample_type);
    auto swizzle = loc2ext_swizzle(image_info.sample_format);
    if (swizzle == NVCV_SWIZZLE_0000)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    NVCVPacking packing0, packing1, packing2, packing3;
    std::tie(packing0, packing1, packing2, packing3) = loc2ext_packing(image_info);
    if (packing0 == NVCV_PACKING_0 && packing1 == NVCV_PACKING_0 && packing2 == NVCV_PACKING_0 && packing3 == NVCV_PACKING_0)
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    
    // more than 4 channels are only supported in the case of interleaved/packed format. 
    // planar format with >4 channels are yet not supported in cvcuda    
    NVCVExtraChannelInfo exChannelInfo;
    // 4 regular channels (color + alpha/black) are supported
    exChannelInfo.numChannels = image_info.plane_info[0].num_channels - 4;
    exChannelInfo.numChannels = exChannelInfo.numChannels < 0 ? 0 : exChannelInfo.numChannels;
    exChannelInfo.bitsPerPixel = image_info.plane_info[0].precision;
    exChannelInfo.datakind = data_kind;
    // placeholders until nvimgcodecs supports these
    exChannelInfo.channelType = NVCV_EXTRA_CHANNEL_U; 
    NVCVAlphaType alpha_type = NVCV_ALPHA_ASSOCIATED; 
    
    if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC)        
    {
        CHECK_NVCV(nvcvMakeYCbCrImageFormat(&(image_data->format), color_spec, css, NVCV_MEM_LAYOUT_PITCH_LINEAR, data_kind, swizzle, packing0, packing1, packing2, packing3, alpha_type, &exChannelInfo));
    }        
    else if (image_info.color_spec == NVIMGCODEC_COLORSPEC_GRAY)
    {
        // if image is gray scale, then we require planes 1,2,3 to have NVCV_PACKING0        
        if (!(num_total_channels == 1 && packing1 == NVCV_PACKING_0 && packing2 == NVCV_PACKING_0 &&
              packing3 == NVCV_PACKING_0 && swizzle == NVCV_SWIZZLE_X000 && css == NVCV_CSS_NONE)) 
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        CHECK_NVCV(nvcvMakeYCbCrImageFormat(&(image_data->format), NVCV_COLOR_SPEC_UNDEFINED, css, NVCV_MEM_LAYOUT_PITCH_LINEAR, data_kind, swizzle, packing0, packing1, packing2, packing3, alpha_type, &exChannelInfo));
    }
    else if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB)
    {
        CHECK_NVCV(nvcvMakeColorImageFormat(&(image_data->format), color_model, color_spec, NVCV_MEM_LAYOUT_PITCH_LINEAR, data_kind, swizzle, packing0, packing1, packing2, packing3, alpha_type, &exChannelInfo));
    }
    else if (image_info.color_spec == NVIMGCODEC_COLORSPEC_YCCK || 
            image_info.color_spec == NVIMGCODEC_COLORSPEC_CMYK)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
    else
    {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TensorData2ImageInfo(nvimgcodecImageInfo_t* image_info, const NVCVTensorData& tensor_data)
{
    try {
        if (tensor_data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        if (tensor_data.rank > 4)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        const NVCVTensorBufferStrided& strided = tensor_data.buffer.strided;
        if (strided.basePtr == nullptr)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        image_info->buffer = static_cast<void*>(strided.basePtr);
        image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        auto sample_type = ext2loc_sample_type(tensor_data.dtype);
        if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_NHWC) == 0 && tensor_data.shape[3] == 3) {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
            image_info->plane_info[0].height = tensor_data.shape[1];
            image_info->plane_info[0].width = tensor_data.shape[2];
            image_info->plane_info[0].row_stride = tensor_data.shape[2] * strided.strides[2];
            image_info->plane_info[0].sample_type = sample_type;
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_HWC) == 0 && tensor_data.shape[2] == 3) {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
            image_info->plane_info[0].height = tensor_data.shape[0];
            image_info->plane_info[0].width = tensor_data.shape[1];
            image_info->plane_info[0].row_stride = tensor_data.shape[1] * strided.strides[1];
            image_info->plane_info[0].sample_type = sample_type;
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_NCHW) == 0 && tensor_data.shape[1] == 3) {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
            image_info->plane_info[0].height = tensor_data.shape[2];
            image_info->plane_info[0].width = tensor_data.shape[3];
            image_info->plane_info[0].row_stride = tensor_data.shape[3] * strided.strides[3];
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_CHW) == 0 && tensor_data.shape[0] == 3) {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
            image_info->plane_info[0].height = tensor_data.shape[1];
            image_info->plane_info[0].width = tensor_data.shape[2];
            image_info->plane_info[0].row_stride = tensor_data.shape[2] * strided.strides[2];
        } else
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        if (image_info->sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
            image_info->num_planes = 1;
            image_info->plane_info[0].num_channels = 3;
        } else if (image_info->sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            image_info->num_planes = 3;
            for (auto p = 0; p < image_info->num_planes; ++p) {
                image_info->plane_info[p].height = image_info->plane_info[0].height;
                image_info->plane_info[p].width = image_info->plane_info[0].width;
                image_info->plane_info[p].row_stride = image_info->plane_info[0].row_stride;
                image_info->plane_info[p].num_channels = 1;
                image_info->plane_info[p].sample_type = sample_type;
            }
        }
        image_info->buffer_size = image_info->plane_info[0].row_stride * image_info->plane_info[0].height * image_info->num_planes;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t ImageInfo2TensorData(NVCVTensorData* tensor_data, const nvimgcodecImageInfo_t& image_info)
{
    try {
        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        tensor_data->bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        NVCVTensorBufferStrided& strided = tensor_data->buffer.strided;
        strided.basePtr = static_cast<NVCVByte*>(image_info.buffer);
        tensor_data->rank = 4;
        tensor_data->dtype = loc2ext_dtype(image_info.plane_info[0].sample_type);
        if (tensor_data->dtype == NVCV_DATA_TYPE_NONE)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        int32_t bpp;
        CHECK_NVCV(nvcvDataTypeGetBitsPerPixel(tensor_data->dtype, &bpp));
        int32_t bytes = (bpp + 7) / 8;
        if (tensor_data->dtype == NVCV_DATA_TYPE_NONE)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            tensor_data->layout = NVCV_TENSOR_NCHW;
            tensor_data->shape[0] = 1;
            tensor_data->shape[1] = 3;
            tensor_data->shape[2] = image_info.plane_info[0].height;
            tensor_data->shape[3] = image_info.plane_info[0].width;

            strided.strides[3] = bytes;
            strided.strides[2] = image_info.plane_info[0].row_stride;
            strided.strides[1] = strided.strides[2] * tensor_data->shape[2];
            strided.strides[0] = strided.strides[1] * tensor_data->shape[1];
        } else if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
            tensor_data->layout = NVCV_TENSOR_NHWC;
            tensor_data->shape[0] = 1;
            tensor_data->shape[1] = image_info.plane_info[0].height;
            tensor_data->shape[2] = image_info.plane_info[0].width;
            tensor_data->shape[3] = 3;

            strided.strides[3] = bytes;
            strided.strides[2] = strided.strides[3] * tensor_data->shape[3];
            strided.strides[1] = image_info.plane_info[0].row_stride;
            strided.strides[0] = strided.strides[1] * tensor_data->shape[1];
        } else {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

}}}} // namespace nvimgcodec::adapter::nvcv
