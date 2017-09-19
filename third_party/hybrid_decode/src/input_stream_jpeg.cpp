/* Copyright 2013-2016 NVIDIA Corporation.  All rights reserved.
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
#include "input_stream_jpeg.h"

#include "codec_jpeg.h"
#include "debug.h"

#include <iostream>
#include <iterator>
#include <vector>
#include <string>

const int InputStreamJPEG::gnTypicalFileSize = 4 * 1024 * 1024; // 4MB

InputStreamJPEG::InputStreamJPEG(unsigned char * aBuffer, size_t nBufferSize)
    : nBufferSize_(nBufferSize), aBuffer_(aBuffer), pCurrent_(aBuffer) {}

InputStreamJPEG::tStreamPosition
InputStreamJPEG::current()
{
    return pCurrent_ - aBuffer_;
}

void
InputStreamJPEG::seek(tStreamPosition nPosition)
{
    pCurrent_ = aBuffer_ + nPosition;
}

void
InputStreamJPEG::read(unsigned char * pBuffer, size_t nLength)
{
    if (pCurrent_ + nLength >= aBuffer_ + nBufferSize_)
        throw std::runtime_error("Read size exceeds buffer size.");
        
	std::copy(pCurrent_, pCurrent_ + nLength, pBuffer);
	pCurrent_ += nLength;
}

void
InputStreamJPEG::read(unsigned short * pBuffer, size_t nLength)
{
    if (pCurrent_ + nLength >= aBuffer_ + nBufferSize_)
        throw std::runtime_error("Read size exceeds buffer size.");

    // copy and fix endianess
    for(size_t i = 0; i < nLength; i++)
    {
        pBuffer[i] = (pCurrent_[i * 2] << 8) | (pCurrent_[i * 2 + 1]);
    }

    pCurrent_ += nLength;
}

const unsigned char *
InputStreamJPEG::getBufferAtOffset(size_t offset)
    const
{
    return aBuffer_ + offset;
}

// Parse the input until a valid marker is found.
//  The function skips byte-stuffed 0xFF 0x00 pattersn as well as
// marker padding bytes (redundant 0xFF bytes leading up to a valid marker.
//  The byte with the actual marker information is returned as an int. A 
// return value of -1 indicates that the end of stream was reached.
//  This implementation is optimized for very short search lengths. In the 
// vast majority of cases, nextMarker is positioned very close to the next 
// marker and the startup cost of faster search methods outweighs their
// faster searching.
//
int 
InputStreamJPEG::nextMarker()
{
    unsigned char nChar;
    read(nChar);
    
    do
    {
        while (nChar != 0xffu && pCurrent_ != aBuffer_ + nBufferSize_)
        {
            read(nChar);
        }

        if (pCurrent_ == aBuffer_ + nBufferSize_)
            return -1;

        read(nChar);
    } while (nChar == 0 || nChar == 0xFFu);
    
    return nChar;
}

// Parse the input until a valid marker is found.
//  The function skips byte-stuffed 0xFF 0x00 pattersn as well as
// marker padding bytes (redundant 0xFF bytes leading up to a valid marker.
//  The byte with the actual marker information is returned as an int. A 
// return value of -1 indicates that the end of stream was reached.
//  This implementation is optimized for very long search lengths. The JPEG
// parser encounters long search lengths for the START_OF_SCAN_JPEG state, i.e.
// when parsing the scan data.
//
int 
InputStreamJPEG::nextMarkerFar()
{
    unsigned char nChar;
    
    do
    {
        pCurrent_ = Find(pCurrent_, aBuffer_ + nBufferSize_, 0xffu);
        // if we found a 0xFF we advance pCurrent_ to the next 
        // character.
        if (pCurrent_ != aBuffer_ + nBufferSize_)
            ++pCurrent_;
        else
            return -1;

        read(nChar);
    } while (nChar == 0 || nChar == 0xFFu);
    
    return nChar;
}

void 
InputStreamJPEG::readComment(std::string & rComment)
{
    unsigned short nDataLength;
    read(nDataLength);
    nDataLength = std::min((size_t) nDataLength, nBufferSize_ - (size_t)(pCurrent_ - aBuffer_));

    std::string sTemp(reinterpret_cast<char *>(pCurrent_), nDataLength - 2);
    rComment.swap(sTemp);
    pCurrent_ += nDataLength - 2;
}

void 
InputStreamJPEG::readApplicationData(std::string & rData)
{
    unsigned short nDataLength;
    read(nDataLength);
    nDataLength = std::min((size_t) nDataLength, nBufferSize_ - (size_t)(pCurrent_ - aBuffer_));

    std::string sTemp(reinterpret_cast<char *>(pCurrent_), nDataLength - 2);
    rData.swap(sTemp);
    pCurrent_ += nDataLength - 2;
}

void 
InputStreamJPEG::skipMarkerData()
{
    unsigned short nDataLength;
    read(nDataLength);

    pCurrent_ += nDataLength - 2;
}


void 
InputStreamJPEG::readRestartInterval(int & rRestartInterval)
{
    unsigned short nSegmentLength;
    unsigned short nRestartInterval;
    
    read(nSegmentLength);
    read(nRestartInterval);
    
    rRestartInterval = static_cast<int>(nRestartInterval);
}

void 
InputStreamJPEG::readFrameHeader(FrameHeader & rFrameHeader)
{
    unsigned short nLength;
    read(nLength);
    
    unsigned char nSamplePrecision;
    read(nSamplePrecision);
    rFrameHeader.setSamplePrecision(nSamplePrecision);
    
    unsigned short nWidth, nHeight;
    read(nHeight);
    read(nWidth);

    if (nHeight == 0 || nWidth == 0)
        throw std::runtime_error("Bad JPEG. (case A)");

    rFrameHeader.setHeight(nHeight);
    rFrameHeader.setWidth(nWidth);
    
    unsigned char nComponents;
    read(nComponents);

    if (nComponents != 1 && nComponents != 3)
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "The CODEC only supports 1 and 3 channel JPEGs.");

    rFrameHeader.setComponents(nComponents);

    for (int c = 0; c < rFrameHeader.components(); ++c)
    {
        unsigned char nComponentIdentifier;
        unsigned char vSamplingFactors;
        unsigned char nQuantizationTableSelector;
        
        read(nComponentIdentifier);
        read(vSamplingFactors);
        read(nQuantizationTableSelector);

        rFrameHeader.setComponentIdentifier(c, nComponentIdentifier);
        rFrameHeader.setSamplingFactors(c, vSamplingFactors);

        // sampling factors allowed
        //
        // 4:4:4  =>  1x1
        // 4:4:0  =>  1x2
        // 4:2:2  =>  2x1
        // 4:2:0  =>  2x2
        // 4:1:1  =>  4x1
        // 4:1:0  =>  4x2

	
        if (rFrameHeader.horizontalSamplingFactor(c) < 1 ||
                rFrameHeader.horizontalSamplingFactor(c) > 4 ||
                rFrameHeader.verticalSamplingFactor(c) < 1 ||
                rFrameHeader.verticalSamplingFactor(c) > 2)
        {
            throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "The CODEC doesn't support sampling factors above 2.");
        }

        rFrameHeader.setQuantizationTableDestinationSelector(c, nQuantizationTableSelector);
    }
}

void 
InputStreamJPEG::readQuantizationTables(QuantizationTable aQuantizationTables[], QuantizationTable * apQuantizationTables[], int numTables)
{
    unsigned short nLength;
    read(nLength);
    nLength -= 2;

    while (nLength > 0)
    {
        unsigned char nPrecisionAndIdentifier;
        read(nPrecisionAndIdentifier);
        nLength--;

        int nPrecision = (nPrecisionAndIdentifier & 0xf0) >> 4;
        int nIdentifier = nPrecisionAndIdentifier & 0x0f;

        if (nIdentifier >= numTables)
            throw std::runtime_error("Bad JPEG. (case B)");

        aQuantizationTables[nIdentifier].nIdentifier = nIdentifier;

        switch(nPrecision)
        {
        case 0:
            aQuantizationTables[nIdentifier].nPrecision = QuantizationTable::PRECISION_8_BIT;
            read(aQuantizationTables[nIdentifier].aTable.lowp, 64);
            nLength -= 64;
            break;

        case 1:
            cout << "found 16-bit quant table" << endl;
            aQuantizationTables[nIdentifier].nPrecision = QuantizationTable::PRECISION_16_BIT;
            read(aQuantizationTables[nIdentifier].aTable.highp, 64);
            nLength -= 64 * 2;
            break;
        }

        // make the pointer array point to the new quatization table struct
        apQuantizationTables[nIdentifier] = &aQuantizationTables[nIdentifier];
    }
}

void 
InputStreamJPEG::readHuffmanTables(CodecJPEGHuffmanTable aHuffmanTables[], CodecJPEGHuffmanTable * apHuffmanTables[], int numTables)
{
    unsigned short nLength;
    read(nLength);
    nLength -= 2;

    while (nLength > 0)
    {
        unsigned char nClassAndIdentifier;
        read(nClassAndIdentifier);
        int nClass = nClassAndIdentifier >> 4; // AC or DC
        int nIdentifier = nClassAndIdentifier & 0x0f;
        int nIdx = nClass * 4 + nIdentifier;

        if (nIdx > numTables)
            throw std::runtime_error("Bad JPEG. (case C)");

        aHuffmanTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

        // Number of Codes for Bit Lengths [1..16]
        int nCodeCount = 0;

        for (int i = 0; i < 16; ++i)
        {
            read(aHuffmanTables[nIdx].aCodes[i]);
            if( aHuffmanTables[nIdx].aCodes[i] < 0 )
                throw std::runtime_error("Bad JPEG. (case E)");
            nCodeCount += aHuffmanTables[nIdx].aCodes[i];
        }

        if (nCodeCount > sizeof(aHuffmanTables[nIdx].aTable))
            throw std::runtime_error("Bad JPEG. (case D)");

        read(aHuffmanTables[nIdx].aTable, nCodeCount);
        nLength -= 17 + nCodeCount;
        
        // make pointer table point to the new Huffman table
        apHuffmanTables[nIdx] = &aHuffmanTables[nIdx];
    }
}

void
InputStreamJPEG::readScanHeader(ScanHeader & rScanHeader)
{
    unsigned short nLength;
    read(nLength);

    read(rScanHeader.nComponents);

    if (rScanHeader.nComponents != 1 && rScanHeader.nComponents != 3)
        throw std::runtime_error("Bad JPEG (case G).");

    for (int c = 0; c < rScanHeader.nComponents; ++c)
    {
        read(rScanHeader.aComponentSelector[c]);
        read(rScanHeader.aHuffmanTablesSelector[c]);
    }

    read(rScanHeader.nSs);
    read(rScanHeader.nSe);
    read(rScanHeader.nA);
}

void
InputStreamJPEG::read(unsigned char & rChar)
{
    if (pCurrent_ >= aBuffer_ + nBufferSize_)
        throw std::runtime_error("Read size exceeds buffer size.");

    rChar = *pCurrent_++;
}

void
InputStreamJPEG::read(unsigned short & rShort)
{
    rShort = readBigEndian<unsigned short>();
}

size_t 
InputStreamJPEG::DivUp(size_t nDividend, size_t nDivisor)
{
    return (nDividend + nDivisor - 1) / nDivisor;
}

unsigned char * 
InputStreamJPEG::AlignUp(unsigned char * pPointer, size_t nAlignment)
{
    return static_cast<unsigned char *>(0) + DivUp(pPointer - static_cast<unsigned char *>(0), nAlignment) * nAlignment;
}

unsigned char * 
InputStreamJPEG::AlignDown(unsigned char * pPointer, size_t nAlignment)
{
    return static_cast<unsigned char *>(0) + ((pPointer - static_cast<unsigned char *>(0)) / nAlignment) * nAlignment;
}

#define HAS_ZERO(V) (((V) - 0x01010101UL) & ~(V) & 0x80808080UL)
#define HAS_VALUE(X,N) (HAS_ZERO((X) ^ (~0UL/255 * (N))))

unsigned char *
InputStreamJPEG::Find(unsigned char * pBegin, unsigned char * pEnd, unsigned char nChar)
{
    unsigned int * pBeginAligned = reinterpret_cast<unsigned int *>(AlignUp(pBegin, sizeof(unsigned int)));
    unsigned int * pEndAligned   = reinterpret_cast<unsigned int *>(AlignDown(pEnd, sizeof(unsigned int)));
    
    if (pBeginAligned < pEndAligned)
    {
        while (pBegin < reinterpret_cast<unsigned char *>(pBeginAligned))
        {
            if (*pBegin == nChar)
                return pBegin;
                
            ++pBegin;
        }
        
        while (pBeginAligned < pEndAligned)
        {
            if (HAS_VALUE(*pBeginAligned, nChar))
            {
                pBegin = reinterpret_cast<unsigned char *>(pBeginAligned);
                while (pBegin < reinterpret_cast<unsigned char *>(pBeginAligned + 1))
                {
                    if (*pBegin == nChar)
                        return pBegin;
                        
                    ++pBegin;
                }
            }
            
            ++pBeginAligned;
        }
        
        pBegin = reinterpret_cast<unsigned char *>(pEndAligned);
        while (pBegin < pEnd)
        {
            if (*pBegin == nChar)
                return pBegin;
                
            ++pBegin;
        }
    }
    else
        return std::find(pBegin, pEnd, nChar);
        
    // ToDo: debug assert that pBegin == pEnd
    return pBegin;
}

