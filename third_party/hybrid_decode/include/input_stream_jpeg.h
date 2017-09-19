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
#ifndef INPUT_JPEG_STREAM_H_
#define INPUT_JPEG_STREAM_H_

#include "host_buffer.h"

#include <string>
#include <istream>
#include <ostream>
#include <algorithm>
#include <vector>

class FrameHeader;
class QuantizationTable;
class CodecJPEGHuffmanTable;
struct ScanHeader;

enum MarkerJPEG
{
    START_OF_FRAME_HUFFMAN_BASELINE_DCT_JPEG = 0xC0,
    SOF1 = START_OF_FRAME_HUFFMAN_BASELINE_DCT_JPEG,
    
    START_OF_FRAME_HUFFMAN_EXTENDET_SEQUENTIAL_DCT_JPEG = 0xC1,
    SOF2 = START_OF_FRAME_HUFFMAN_EXTENDET_SEQUENTIAL_DCT_JPEG,
    
    START_OF_FRAME_HUFFMAN_PROGRESSIVE_DCT_JPEG = 0xC2,
    SOF3 = START_OF_FRAME_HUFFMAN_PROGRESSIVE_DCT_JPEG,
    
    START_OF_FRAME_HUFFMAN_LOSSLESS_JPEG = 0xC3,
    SOF4 = START_OF_FRAME_HUFFMAN_LOSSLESS_JPEG,
    
    START_OF_FRAME_HUFFMAN_DIFFERENTIAL_SEQUENTIAL_DCT_JPEG = 0xC5, 
    SOF5 = START_OF_FRAME_HUFFMAN_DIFFERENTIAL_SEQUENTIAL_DCT_JPEG,
    
    START_OF_FRAME_HUFFMAN_DIFFERENTIAL_PROGRESSIVE_DCT_JPEG = 0xC6,
    SOF6 = START_OF_FRAME_HUFFMAN_DIFFERENTIAL_PROGRESSIVE_DCT_JPEG,
    
    START_OF_FRAME_HUFFMAN_DIFFERENTIAL_LOSSLESS_JPEG = 0xC7,
    SOF7 = START_OF_FRAME_HUFFMAN_DIFFERENTIAL_LOSSLESS_JPEG,
    
    RESERVED_FRAME_FOR_EXTENSIONS_JPEG = 0xC8,
    JPG = RESERVED_FRAME_FOR_EXTENSIONS_JPEG,
    
    START_OF_FRAME_ARITHMETIC_EXTENDED_SEQUENTIAL_DCT_JPEG = 0xC9,
    SOF9 = START_OF_FRAME_ARITHMETIC_EXTENDED_SEQUENTIAL_DCT_JPEG,
    
    START_OF_FRAME_ARITHMETIC_PROGRESSIVE_DCT_JPEG = 0xCA,
    SOF10 = START_OF_FRAME_ARITHMETIC_PROGRESSIVE_DCT_JPEG,
    
    START_OF_FRAME_ARITHMETIC_LOSSLESS_JPEG = 0xCB,
    SOF11 = START_OF_FRAME_ARITHMETIC_LOSSLESS_JPEG,

    START_OF_FRAME_ARITHMETIC_SEQUENTIAL_DCT_JPEG = 0xCD,
    SOF13 = START_OF_FRAME_ARITHMETIC_SEQUENTIAL_DCT_JPEG,
    
    START_OF_FRAME_ARITHMETIC_DIFFERENTIAL_PROGRESSIVE_DCT = 0xCE,
    SOF14 = START_OF_FRAME_ARITHMETIC_DIFFERENTIAL_PROGRESSIVE_DCT,
    
    START_OF_FRAME_ARITHMETIC_DIFFERENTIAL_LOSSLESS_JPEG = 0xCF,
    SOF15 = START_OF_FRAME_ARITHMETIC_DIFFERENTIAL_LOSSLESS_JPEG,
    
    DEFINE_HUFFMAN_TABLE_JPEG = 0xC4,
    DHT = DEFINE_HUFFMAN_TABLE_JPEG,
    
    DEFINE_ARITHMETIC_CODING_CONDITIONING = 0xCC,
    DAC = DEFINE_ARITHMETIC_CODING_CONDITIONING,
    
    RESTART_0_JPEG = 0xD0,
    RST0 = RESTART_0_JPEG,
    
    RESTART_1_JPEG = 0xD1,
    RST1 = RESTART_1_JPEG,
    
    RESTART_2_JPEG = 0xD2,
    RST2 = RESTART_2_JPEG,
    
    RESTART_3_JPEG = 0xD3,
    RST3 = RESTART_3_JPEG,
    
    RESTART_4_JPEG = 0xD4,
    RST4 = RESTART_4_JPEG,
    
    RESTART_5_JPEG = 0xD5,
    RST5 = RESTART_5_JPEG,
    
    RESTART_6_JPEG = 0xD6,
    RST6 = RESTART_6_JPEG,
    
    RESTART_7_JPEG = 0xD7,
    RST7 = RESTART_7_JPEG,
    
    START_0F_IMAGE_JPEG = 0xD8,
    SOI = START_0F_IMAGE_JPEG,
    
    END_OF_IMAGE_JPEG = 0xD9, 
    E0F = END_OF_IMAGE_JPEG,
    
    START_OF_SCAN_JPEG = 0xDA,
    SOS = START_OF_SCAN_JPEG,
    
    DEFINE_QUANTIZATION_TABLE_JPEG = 0xDB,
    DQT = DEFINE_QUANTIZATION_TABLE_JPEG,
    
    DEFINE_NUMBER_OF_LINES_JPEG = 0xDC,
    DNL = DEFINE_NUMBER_OF_LINES_JPEG, 
    
    DEFINE_RESTART_INTERVAL_JPEG = 0xDD,
    DRI = DEFINE_RESTART_INTERVAL_JPEG,
    
    DEFINE_HIERARCHICAL_PROGRESSION_JPEG = 0xDE,
    DHP = DEFINE_HIERARCHICAL_PROGRESSION_JPEG,
    
    EXPAND_REFERENCE_COMPONENT_JPEG = 0xDF,
    EXP = EXPAND_REFERENCE_COMPONENT_JPEG,
    
    APPLICATION_SEGMENT_0_JPEG = 0xE0,
    APP0 = APPLICATION_SEGMENT_0_JPEG,
    
    APPLICATION_SEGMENT_1_JPEG = 0xE1,
    APP1 = APPLICATION_SEGMENT_1_JPEG,
    
    APPLICATION_SEGMENT_2_JPEG = 0xE2,
    APP2 = APPLICATION_SEGMENT_2_JPEG,
    
    APPLICATION_SEGMENT_3_JPEG = 0xE3,
    APP3 = APPLICATION_SEGMENT_3_JPEG,
    
    APPLICATION_SEGMENT_4_JPEG = 0xE4,
    APP4 = APPLICATION_SEGMENT_4_JPEG,
    
    APPLICATION_SEGMENT_5_JPEG = 0xE5,
    APP5 = APPLICATION_SEGMENT_5_JPEG,
    
    APPLICATION_SEGMENT_6_JPEG = 0xE6,
    APP6 = APPLICATION_SEGMENT_6_JPEG,
    
    APPLICATION_SEGMENT_7_JPEG = 0xE7,
    APP7 = APPLICATION_SEGMENT_7_JPEG,
    
    APPLICATION_SEGMENT_8_JPEG = 0xE8,
    APP8 = APPLICATION_SEGMENT_8_JPEG,
    
    APPLICATION_SEGMENT_9_JPEG = 0xE9,
    APP9 = APPLICATION_SEGMENT_9_JPEG,
    
    APPLICATION_SEGMENT_10_JPEG = 0xEA,
    APP10 = APPLICATION_SEGMENT_10_JPEG,
    
    APPLICATION_SEGMENT_11_JPEG = 0xEB,
    APP11 = APPLICATION_SEGMENT_11_JPEG,
    
    APPLICATION_SEGMENT_12_JPEG = 0xEC,
    APP12 = APPLICATION_SEGMENT_12_JPEG,
    
    APPLICATION_SEGMENT_13_JPEG = 0xED,
    APP13 = APPLICATION_SEGMENT_13_JPEG,
    
    APPLICATION_SEGMENT_14_JPEG = 0xEE,
    APP14 = APPLICATION_SEGMENT_14_JPEG,
    
    APPLICATION_SEGMENT_15_JPEG = 0xEF,
    APP15 = APPLICATION_SEGMENT_15_JPEG,
    
    EXTENSION_0_JPEG = 0xF0,
    JPG0 = EXTENSION_0_JPEG,
    
    EXTENSION_1_JPEG = 0xF1,
    JPG1 = EXTENSION_1_JPEG,
    
    EXTENSION_2_JPEG = 0xF2,
    JPG2 = EXTENSION_2_JPEG,
    
    EXTENSION_3_JPEG = 0xF3,
    JPG3 = EXTENSION_3_JPEG,
    
    EXTENSION_4_JPEG = 0xF4,
    JPG4 = EXTENSION_4_JPEG,
    
    EXTENSION_5_JPEG = 0xF5,
    JPG5 = EXTENSION_5_JPEG,
    
    EXTENSION_6_JPEG = 0xF6,
    JPG6 = EXTENSION_6_JPEG,
    
    EXTENSION_7_JPEG = 0xF7,
    JPG7 = EXTENSION_7_JPEG,
    
    EXTENSION_8_JPEG = 0xF8,
    JPG8 = EXTENSION_8_JPEG,
    
    EXTENSION_9_JPEG = 0xF9,
    JPG9 = EXTENSION_9_JPEG,
    
    EXTENSION_10_JPEG = 0xFA,
    JPG10 = EXTENSION_10_JPEG,
    
    EXTENSION_11_JPEG = 0xFB,
    JPG11 = EXTENSION_1_JPEG,
    
    EXTENSION_12_JPEG = 0xFC,
    JPG12 = EXTENSION_12_JPEG,
    
    EXTENSION_13_JPEG = 0xFD,
    JPG13 = EXTENSION_13_JPEG,
    
    COMMENT_JPEG = 0xFE,
    COM = COMMENT_JPEG,
};

const int N_APPLICATION_SEGMENTS_JPEG = 16;

class InputStreamJPEG
{
public:
    typedef size_t tStreamPosition;

    static const int gnTypicalFileSize; // 4MB

public:
    explicit
    InputStreamJPEG(unsigned char * aBuffer, size_t nBufferSize);
    
    tStreamPosition
    current();
    
    void
    seek(tStreamPosition nPosition);
    
    void
    read(unsigned char * pBuffer, size_t nLength);
    
    void
    read(unsigned short *pBuffer, size_t nLength);

    const unsigned char *
    getBufferAtOffset(size_t offset)
    const;

    int 
    nextMarker();

    int 
    nextMarkerFar();

    void 
    skipMarkerData();

    void 
    readComment(std::string & rComment);
    
    void 
    readApplicationData(std::string & rData);
    
    void 
    readFrameHeader(FrameHeader & rFrameHeader);

    void 
    readRestartInterval(int & rRestartInterval);

    void 
    readQuantizationTables(QuantizationTable aQuantizationTables[], QuantizationTable * apQuantizationTables[], int numTables);

    void 
    readHuffmanTables(CodecJPEGHuffmanTable aHuffmanTables[], CodecJPEGHuffmanTable * apHuffmanTables[], int numTables);

    void
    readScanHeader(ScanHeader & rScanHeader);

protected:
    void
    read(unsigned char & rChar);
    
    void
    read(unsigned short & rShort);
    
    template<class T>
    T 
    readBigEndian()
    {
        unsigned char src[sizeof(T)];
        read(src, sizeof(T));
        unsigned char dst[sizeof(T)];
        std::reverse_copy(src, src + sizeof(T), dst);
        return *reinterpret_cast<T *>(dst);
    }

    static
    size_t 
    DivUp(size_t nDividend, size_t nDivisor);
    
    static
    unsigned char * 
    AlignUp(unsigned char * pPointer, size_t nAlignment);

    static
    unsigned char * 
    AlignDown(unsigned char * pPointer, size_t nAlignment);
    
    static
    unsigned char *
    Find(unsigned char * pBegin, unsigned char * pEnd, unsigned char nChar);
    
private:
    /// Size of input JPEG file
    size_t          nBufferSize_;

    /// Beginning of input JPEG file, points to HostBuffer oImageFileBUffer_ owned by CodecJPEG.
    /// In constructor, that HostBuffer is resized so it contains additional padding
    /// needed by NPP functions.
    unsigned char * aBuffer_;

    /// Current position in JPEG file
    unsigned char * pCurrent_;
};

#endif // INPUT_JPEG_STREAM_H_
