/* Copyright 2014-2016 NVIDIA Corporation.  All rights reserved.
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

#include "codec_jpeg.h"

#include "debug.h"
#include "input_stream_jpeg.h"

#include <cuda_runtime.h>
#include <nppi_compression_functions.h>

#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <memory>
#include <iomanip>

#define USE_EXPERIMENTAL_DCT

template<class T>
inline void alignUp(T & x, T alignement) 
{
    x = ((x - 1) & ~(alignement - 1)) + alignement;
}

// ----------------------------------------------------------------------------
// ComponentSpecification
//

ComponentSpecification::ComponentSpecification(unsigned char nComponentIdentifier,
        unsigned char vSampleFactors,
        unsigned char nQuantizationTableDestinationSelector): nComponentIdendifier_(nComponentIdentifier)
                                                            , vSamplingFactors_(vSampleFactors)
                                                            , nQuantizationTableDestinationSelector_(nQuantizationTableDestinationSelector)
{ ; }

void
ComponentSpecification::setComponentIdentifier(unsigned char nComponentIdentifier)
{
    nComponentIdendifier_ = nComponentIdentifier;
}

unsigned char
ComponentSpecification::componentIdentifier()
    const
{
    return nComponentIdendifier_;
}

void
ComponentSpecification::setSamplingFactors(unsigned char vSamplingFactors)
{
    vSamplingFactors_ = vSamplingFactors;
}

unsigned char
ComponentSpecification::samplingFactors()
    const
{
    return vSamplingFactors_;
}

void
ComponentSpecification::setHorizontalSamplingFactor(int nHorizontalSamplingFactor)
{
    if (0 <= nHorizontalSamplingFactor && nHorizontalSamplingFactor < 16)
        vSamplingFactors_ = (nHorizontalSamplingFactor << 4) | (vSamplingFactors_ & 0x0F);
    else
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Invalid Horizontal Sampling Factor");
}

int
ComponentSpecification::horizontalSamplingFactor()
    const
{
    return static_cast<int>(vSamplingFactors_ >> 4);
}

void
ComponentSpecification::setVerticalSamplingFactor(int nVerticalSamplingFactor)
{
    if (0 <= nVerticalSamplingFactor && nVerticalSamplingFactor < 16)
        vSamplingFactors_ = (vSamplingFactors_ & 0xF0) | nVerticalSamplingFactor;
    else
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Invalid Vertical Sampling Factor");
}

int
ComponentSpecification::verticalSamplingFactor()
    const
{
    return static_cast<int>(vSamplingFactors_ & 0x0F);
}

void
ComponentSpecification::setQuantizationTableDestinationSelector(unsigned char nQuantizationTableDestinationSelector)
{
    nQuantizationTableDestinationSelector_ = nQuantizationTableDestinationSelector;
}

unsigned char 
ComponentSpecification::quantizationTableDestinationSelector()
    const
{
    return nQuantizationTableDestinationSelector_;
}


// ----------------------------------------------------------------------------
// Frameheader
//

FrameHeader::FrameHeader(): eEncoding_(BASELINE_DCT)
                          , nSamplePrecision_(8)
                          , nHeight_(0)
                          , nWidth_(0)
                          , nComponents_(0)
{ ; }

FrameHeader::FrameHeader(const FrameHeader & rFrameHeader): eEncoding_(rFrameHeader.eEncoding_)
                                                          , nSamplePrecision_(rFrameHeader.nSamplePrecision_)
                                                          , nHeight_(rFrameHeader.nHeight_)
                                                          , nWidth_(rFrameHeader.nWidth_)
                                                          , nComponents_(rFrameHeader.nComponents_)
                                                          , aComponentSpecifications_(rFrameHeader.aComponentSpecifications_)
{ ; }

FrameHeader & 
FrameHeader::operator= (const FrameHeader & rFrameHeader)
{
    if (&rFrameHeader == this)
        return *this;
        
    eEncoding_                  = rFrameHeader.eEncoding_;
    nSamplePrecision_           = rFrameHeader.nSamplePrecision_;
    nHeight_                    = rFrameHeader.nHeight_;
    nWidth_                     = rFrameHeader.nWidth_;
    nComponents_                = rFrameHeader.nComponents_;
    aComponentSpecifications_   = rFrameHeader.aComponentSpecifications_;
    
    return *this;
}
 
void
FrameHeader::reset()
{
    eEncoding_          = BASELINE_DCT;
    nSamplePrecision_   = 8;
    nHeight_            = 0;
    nWidth_             = 0;
    nComponents_        = 0;
}
   
void
FrameHeader::setWidth(unsigned short nWidth)
{
    nWidth_ = nWidth;
}

unsigned short
FrameHeader::width()
    const
{
    return nWidth_;
}

void
FrameHeader::setHeight(unsigned short nHeight)
{
    nHeight_ = nHeight;
}

unsigned short
FrameHeader::height()
    const
{
    return nHeight_;
}

unsigned short 
FrameHeader::width(int iComponent)
    const
{
    // According to ITU-T81 (JPEG Spec) Appendix A.1.1 width of the component 
    // is ceil(X * Hi / Hmax).
    return static_cast<unsigned short>(DivUp(width() * horizontalSamplingFactor(iComponent), maximumHorizontalSamplingFactor()));
}

unsigned short
FrameHeader::height(int iComponent)
    const
{
    // According to ITU-T81 (JPEG Spec) Appendix A.1.1 height of the component 
    // is ceil(X * Hi / Hmax).
    return static_cast<unsigned short>(DivUp(height() * verticalSamplingFactor(iComponent), maximumVerticalSamplingFactor()));
}

void
FrameHeader::setSamplePrecision(unsigned char nSamplePrecision)
{
    nSamplePrecision_ = nSamplePrecision;
}

unsigned char 
FrameHeader::samplePrecision()
    const
{
    return nSamplePrecision_;
}
    
void
FrameHeader::setEncoding(teEncoding eEncoding)
{
    eEncoding_ = eEncoding;
}

FrameHeader::teEncoding
FrameHeader::encoding()
    const
{
    return eEncoding_;
}

void
FrameHeader::setComponents(unsigned char nComponents)
{
    nComponents_ = nComponents;
    aComponentSpecifications_.resize(nComponents_);
}

int
FrameHeader::components()
    const
{
    return static_cast<int>(this->nComponents_);
}

int 
FrameHeader::componentIndexForIdentifier(int nIdentifier) 
    const 
{
    for( int i=0; i < 3; ++i ) 
    {
        if (componentIdentifier(i) == nIdentifier) 
            return i;
    }
    
    return -1;
}

void
FrameHeader::setComponentIdentifier(int iComponent, unsigned char nComponentIdentifier)
{
    componentSpecification(iComponent).setComponentIdentifier(nComponentIdentifier);
}

unsigned char
FrameHeader::componentIdentifier(int iComponent)
    const
{
    return componentSpecification(iComponent).componentIdentifier();
}

void
FrameHeader::setSamplingFactors(int iComponent, unsigned char vSamplingFactors)
{
    componentSpecification(iComponent).setSamplingFactors(vSamplingFactors);
}

unsigned char
FrameHeader::samplingFactors(int iComponent)
    const
{
    return componentSpecification(iComponent).samplingFactors();
}

void
FrameHeader::setHorizontalSamplingFactor(int iComponent, int nHorizontalSamplingFactor)
{
    componentSpecification(iComponent).setHorizontalSamplingFactor(nHorizontalSamplingFactor);
}

int
FrameHeader::horizontalSamplingFactor(int iComponent)
    const
{
    return componentSpecification(iComponent).horizontalSamplingFactor();
}

int
FrameHeader::maximumHorizontalSamplingFactor()
    const
{
    int nMaximumHorizontalSamplingFactor = horizontalSamplingFactor(0);
    for (int iComponent = 1; iComponent < components(); ++iComponent)
        if (nMaximumHorizontalSamplingFactor < horizontalSamplingFactor(iComponent))
            nMaximumHorizontalSamplingFactor = horizontalSamplingFactor(iComponent);
    
    return nMaximumHorizontalSamplingFactor;
}

void
FrameHeader::setVerticalSamplingFactor(int iComponent, int nVerticalSamplingFactor)
{
    componentSpecification(iComponent).setVerticalSamplingFactor(nVerticalSamplingFactor);
}

int
FrameHeader::verticalSamplingFactor(int iComponent)
    const
{
    return componentSpecification(iComponent).verticalSamplingFactor();
}

int
FrameHeader::maximumVerticalSamplingFactor()
    const
{
    int nMaximumVerticalSamplingFactor = verticalSamplingFactor(0);
    for (int iComponent = 1; iComponent < components(); ++iComponent)
        if (nMaximumVerticalSamplingFactor < verticalSamplingFactor(iComponent))
            nMaximumVerticalSamplingFactor = verticalSamplingFactor(iComponent);
    
    return nMaximumVerticalSamplingFactor;
}

void
FrameHeader::setQuantizationTableDestinationSelector(int iComponent, unsigned char nQuantizationTableDestinationSelector)
{
    componentSpecification(iComponent).setQuantizationTableDestinationSelector(nQuantizationTableDestinationSelector);
}

unsigned char 
FrameHeader::quantizationTableDestinationSelector(int iComponent)
    const
{
    return componentSpecification(iComponent).quantizationTableDestinationSelector();
}

ComponentSpecification & 
FrameHeader::componentSpecification(int iComponent)
{
    if (0 <= iComponent && iComponent < components())
        return aComponentSpecifications_[iComponent];
    else
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Component Index Out-of-Range");
}

const
ComponentSpecification & 
FrameHeader::componentSpecification(int iComponent)
    const
{
    if (0 <= iComponent && iComponent < components())
        return aComponentSpecifications_[iComponent];
    else
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Component Index Out-of-Range");
}

FrameHeader::teEncoding
FrameHeader::GetEncoding(int nMarker)
{
    switch (nMarker)
    {
    case START_OF_FRAME_HUFFMAN_BASELINE_DCT_JPEG:
        return BASELINE_DCT;
    case START_OF_FRAME_HUFFMAN_EXTENDET_SEQUENTIAL_DCT_JPEG:
        return EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
    case START_OF_FRAME_HUFFMAN_PROGRESSIVE_DCT_JPEG:
        return PROGRESSIVE_DCT_HUFFMAN;
    case START_OF_FRAME_HUFFMAN_LOSSLESS_JPEG:
        return LOSSLESS_HUFFMAN;
    case START_OF_FRAME_ARITHMETIC_EXTENDED_SEQUENTIAL_DCT_JPEG:
        return EXTENDED_SEQUENTIAL_DCT_ARITHMETIC;
    case START_OF_FRAME_ARITHMETIC_PROGRESSIVE_DCT_JPEG:
        return PROGRESSIVE_DCT_ARITHMETIC;
    case START_OF_FRAME_ARITHMETIC_LOSSLESS_JPEG:
        return LOSSESS_ARITHMETIC;
    default:
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Cannot convert marker to encoding.");
    }
}
        

// ----------------------------------------------------------------------------
// QuantizationTable
//
QuantizationTable::QuantizationTable()
    : nPrecision(PRECISION_8_BIT),
      nIdentifier(0)
{
    for (unsigned short * pElement = aTable.highp; pElement != aTable.highp + 64; ++pElement)
        *pElement = 0;
}

QuantizationTable::QuantizationTable(const QuantizationTable & rQuantizationTable)
    : nPrecision(rQuantizationTable.nPrecision),
      nIdentifier(rQuantizationTable.nIdentifier)
{
    switch(nPrecision)
    {
    case PRECISION_8_BIT:
        std::copy(rQuantizationTable.aTable.lowp, rQuantizationTable.aTable.lowp + 64, aTable.lowp);
        break;

    case PRECISION_16_BIT:
        std::copy(rQuantizationTable.aTable.highp, rQuantizationTable.aTable.highp + 64, aTable.highp);
        break;
    }
}

QuantizationTable & 
QuantizationTable::operator= (const QuantizationTable & rQuantizationTable)
{
    if (&rQuantizationTable == this)
        return *this;
        
    nPrecision = rQuantizationTable.nPrecision;
    nIdentifier = rQuantizationTable.nIdentifier;

    switch(nPrecision)
    {
    case PRECISION_8_BIT:
        std::copy(rQuantizationTable.aTable.lowp, rQuantizationTable.aTable.lowp + 64, aTable.lowp);
        break;

    case PRECISION_16_BIT:
        std::copy(rQuantizationTable.aTable.highp, rQuantizationTable.aTable.highp + 64, aTable.highp);
        break;
    }
    
    return *this;
}
    

const int QuantizationTable::ZIGZAG_ORDER[] = {
    0,  1,  5,  6,  14, 15, 27, 28,
    2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43,
    9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

const unsigned char QuantizationTable::LUMINANCE_BASE_TABLE[] = {
    0x10, 0x0b, 0x0a, 0x10, 0x18, 0x28, 0x33, 0x3d,
    0x0c, 0x0c, 0x0e, 0x13, 0x1a, 0x3a, 0x3c, 0x37,
    0x0e, 0x0d, 0x10, 0x18, 0x28, 0x39, 0x45, 0x38,
    0x0e, 0x11, 0x16, 0x1d, 0x33, 0x57, 0x50, 0x3e,
    0x12, 0x16, 0x25, 0x38, 0x44, 0x6d, 0x67, 0x4d,
    0x18, 0x23, 0x37, 0x40, 0x51, 0x68, 0x71, 0x5c,
    0x31, 0x40, 0x4e, 0x57, 0x67, 0x79, 0x78, 0x65,
    0x48, 0x5c, 0x5f, 0x62, 0x70, 0x64, 0x67, 0x63
};

const unsigned char QuantizationTable::CHROMINANCE_BASE_TABLE[] = {
    0x11, 0x12, 0x18, 0x2f, 0x63, 0x63, 0x63, 0x63,
    0x12, 0x15, 0x1a, 0x42, 0x63, 0x63, 0x63, 0x63,
    0x18, 0x1a, 0x38, 0x63, 0x63, 0x63, 0x63, 0x63,
    0x2f, 0x42, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
    0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
    0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
    0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
    0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63
};


void 
QuantizationTable::setQualityQuantizationTable(int quality, const unsigned char* pBaseTable, bool allow16Bit)
{
    if(quality  <= 0) quality = 1;

    if(quality > 100) quality = 100;

    quality = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);

    nPrecision = PRECISION_8_BIT;

    for(unsigned i = 0; i < 64; i++){
        int val = (pBaseTable[i] * quality + 50) / 100;

        if(val <= 0) val = 1;
        if(val > 255) {
            if (allow16Bit)
            {
                nPrecision = PRECISION_16_BIT;
                break;
            } else {
                val = 255;
            }
        }

        aTable.lowp[ZIGZAG_ORDER[i]] = static_cast<unsigned char>(val);
    }

    if (nPrecision == PRECISION_16_BIT)
    {
        for(unsigned i = 0; i < 64; i++){
            int val = (pBaseTable[i] * quality + 50) / 100;

            if(val <= 0) val = 1;
            if(val > 65535) val = 65535;

            aTable.highp[ZIGZAG_ORDER[i]] = static_cast<unsigned short>(val);
        }
    }
}

void 
QuantizationTable::setQualityLuminance(int quality, bool allow16Bit)
{
    setQualityQuantizationTable(quality, LUMINANCE_BASE_TABLE, allow16Bit);
}

void 
QuantizationTable::setQualityChrominance(int quality, bool allow16Bit)
{
    setQualityQuantizationTable(quality, CHROMINANCE_BASE_TABLE, allow16Bit);
}

// ----------------------------------------------------------------------------
// CodeJPEGHuffmanTable
//
CodecJPEGHuffmanTable::CodecJPEGHuffmanTable(): nClassAndIdentifier(0)
{
    for (unsigned char * pCode = aCodes; pCode < aCodes + 16; ++pCode)
        *pCode = 0;
        
    for (unsigned char * pValue = aTable; pValue < aTable + 256; ++pValue)
        *pValue = 0;
}

CodecJPEGHuffmanTable::CodecJPEGHuffmanTable(const CodecJPEGHuffmanTable & rHuffmanTable): nClassAndIdentifier(0)
{
    std::copy(rHuffmanTable.aCodes, rHuffmanTable.aCodes + 16, aCodes);
    std::copy(rHuffmanTable.aTable, rHuffmanTable.aTable + 256, aTable);
}

CodecJPEGHuffmanTable &
CodecJPEGHuffmanTable::operator= (const CodecJPEGHuffmanTable & rHuffmanTable)
{
    if (&rHuffmanTable == this)
        return *this;
        
    nClassAndIdentifier = rHuffmanTable.nClassAndIdentifier;
        
    std::copy(rHuffmanTable.aCodes, rHuffmanTable.aCodes + 16, aCodes);
    std::copy(rHuffmanTable.aTable, rHuffmanTable.aTable + 256, aTable);

    return *this;
}


void CodecJPEGHuffmanTable::print(int count) const
{
    std::cout << "Class&ID: " << std::setw(2) << std::setfill('0') << std::hex << int(nClassAndIdentifier) << std::endl;
    std::cout << "aCodes: ";
    for(int i = 0; i<16; i++)
    {
        std::cout << std::setw(2) << std::setfill('0') << std::hex << int( aCodes[i] ) << ", ";
    }
    std::cout << "\naTable: " ;
    for(int i = 0; i<count; i++)
    {
        std::cout << std::setw(2) << std::setfill('0') << std::hex << int( aTable[i] ) << ", "; 
    }
    std::cout << std::endl;
}

const unsigned char CodecJPEGHuffmanTable::DEFAULT_LUMINANCE_AC_TABLE[] = {
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01,
    0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1,
    0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6,
    0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
    0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa
};

const unsigned char CodecJPEGHuffmanTable::DEFAULT_LUMINANCE_DC_TABLE[] = {
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 
};

const unsigned char CodecJPEGHuffmanTable::DEFAULT_CHROMINANCE_AC_TABLE[] = {
    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02,
    0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52,
    0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4,
    0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2,
    0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
    0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa,   
};

const unsigned char CodecJPEGHuffmanTable::DEFAULT_CHROMINANCE_DC_TABLE[] = {
    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 
};

void 
CodecJPEGHuffmanTable::setDefaultLuminanceDC() 
{
    memcpy(aCodes, DEFAULT_LUMINANCE_DC_TABLE, sizeof(DEFAULT_LUMINANCE_DC_TABLE));
}

void 
CodecJPEGHuffmanTable::setDefaultLuminanceAC() 
{
    memcpy(aCodes, DEFAULT_LUMINANCE_AC_TABLE, sizeof(DEFAULT_LUMINANCE_AC_TABLE));
}

void 
CodecJPEGHuffmanTable::setDefaultChrominanceDC() 
{
    memcpy(aCodes, DEFAULT_CHROMINANCE_DC_TABLE, sizeof(DEFAULT_CHROMINANCE_DC_TABLE));
}

void 
CodecJPEGHuffmanTable::setDefaultChrominanceAC() 
{
    memcpy(aCodes, DEFAULT_CHROMINANCE_AC_TABLE, sizeof(DEFAULT_CHROMINANCE_AC_TABLE));
}

// ----------------------------------------------------------------------------
// Scan
//
Scan::Scan(): nRestartInterval_(0)
{ 
    for (int iTable = 0; iTable < 8; ++iTable)
        apHuffmanTables_[iTable] = 0;   
}

void Scan::copyTablesAndRestartState(const Scan &rScan)
{ 
    // Initialize with Tables and Restart Interval from passed Scan
    for (int iTable = 0; iTable < 8; ++iTable)
        apHuffmanTables_[iTable] = rScan.apHuffmanTables_[iTable]; 
        
    nRestartInterval_ = rScan.nRestartInterval_;
}

void
Scan::setRestartInterval(int nRestartInterval)
{
    nRestartInterval_ = nRestartInterval;
}

int
Scan::restartInterval()
    const
{
    return nRestartInterval_;
}

ScanHeader &
Scan::scanHeader()
{
    return oScanHeader_;
}

const
ScanHeader &
Scan::scanHeader()
    const
{
    return oScanHeader_;
}

int
Scan::components()
    const
{
    return scanHeader().nComponents;
}

CodecJPEGHuffmanTable &
Scan::huffmanTableDC(int iIndex)
{
    if (iIndex < 0 || iIndex >= 4)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "DC Huffman Table Index Out-Of-Range");
        
    if (!apHuffmanTables_[iIndex])
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Huffman Table Not Found");
        
    return *apHuffmanTables_[iIndex];
}

const
CodecJPEGHuffmanTable &
Scan::huffmanTableDC(int iIndex)
    const
{
    if (iIndex < 0 || iIndex >= 4)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "DC Huffman Table Index Out-Of-Range");
        
    if (!apHuffmanTables_[iIndex])
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Huffman Table Not Found");
        
    return *apHuffmanTables_[iIndex];
}

CodecJPEGHuffmanTable &
Scan::huffmanTableAC(int iIndex)
{
    if (iIndex < 0 || iIndex >= 4)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "DC Huffman Table Index Out-Of-Range");
        
    if (!apHuffmanTables_[iIndex + 4])
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Huffman Table Not Found");
        
    return *apHuffmanTables_[iIndex + 4];
}

const
CodecJPEGHuffmanTable &
Scan::huffmanTableAC(int iIndex)
    const
{
    if (iIndex < 0 || iIndex >= 4)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "DC Huffman Table Index Out-Of-Range");
        
    if (!apHuffmanTables_[iIndex + 4])
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Huffman Table Not Found");
        
    return *apHuffmanTables_[iIndex + 4];
}

int
Scan::huffmanTableSelectorDC(int iComponent)
    const
{
    // todo: input validation
    return scanHeader().aHuffmanTablesSelector[iComponent] >> 4;
}

int 
Scan::huffmanTableSelectorAC(int iComponent)
    const
{
    // todo: input validation
    return scanHeader().aHuffmanTablesSelector[iComponent] & 0x0f;
}

const
CodecJPEGHuffmanTable &
Scan::huffmanTableForComponentDC(int iComponent)
    const
{
    return huffmanTableDC(huffmanTableSelectorDC(iComponent));
}

const
CodecJPEGHuffmanTable &
Scan::huffmanTableForComponentAC(int iComponent)
    const
{
    return huffmanTableAC(huffmanTableSelectorAC(iComponent));
}

void
Scan::addComment(const std::string & rComment)
{
    aComments_.push_back(rComment);
}

void
Scan::addApplicationData(int iIndex, const std::string & rData)
{
    if (iIndex < 0 || iIndex > 15)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Application-Data Index Out-of-Range");
    aApplicationData_[iIndex].push_back(rData);
}

void
Scan::setBuffer(unsigned char const * buf, size_t len)
{
    aBuffer_ = buf;
    nBufferLength_ = len;
}

const
unsigned char *
Scan::bufferData()
    const
{
    return aBuffer_;
}

int
Scan::bufferSize()
    const
{
    return nBufferLength_;  
}
