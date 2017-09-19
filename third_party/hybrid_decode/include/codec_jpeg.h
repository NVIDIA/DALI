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
#ifndef CODEC_JPEG_H_
#define CODEC_JPEG_H_

#include "common.h"
#include "host_buffer.h"

class ComponentSpecification
{
public:
    //explicit
    ComponentSpecification(unsigned char nComponentIdentifier = 0,
                           unsigned char vSampleFactors = 0,
                           unsigned char nQuantizationTableDestinationSelector = 0);

    void
    setComponentIdentifier(unsigned char nComponentIdentifier);
    
    unsigned char
    componentIdentifier()
    const;
    
    void
    setSamplingFactors(unsigned char vSamplingFactors);
    
    unsigned char
    samplingFactors()
    const;
    
    void
    setHorizontalSamplingFactor(int nHorizontalSamplingFactor);
    
    int
    horizontalSamplingFactor()
    const;
    
    void
    setVerticalSamplingFactor(int nVerticalSamplingFactor);
    
    int
    verticalSamplingFactor()
    const;
    
    void
    setQuantizationTableDestinationSelector(unsigned char nQuantizationTableDestinationSelector);
    
    unsigned char 
    quantizationTableDestinationSelector()
    const;
    
private:
    unsigned char nComponentIdendifier_;
    unsigned char vSamplingFactors_;
    unsigned char nQuantizationTableDestinationSelector_;
};

class FrameHeader
{
public:
    enum teEncoding
    {
        BASELINE_DCT,
        EXTENDED_SEQUENTIAL_DCT_HUFFMAN,
        PROGRESSIVE_DCT_HUFFMAN,
        LOSSLESS_HUFFMAN,
        EXTENDED_SEQUENTIAL_DCT_ARITHMETIC,
        PROGRESSIVE_DCT_ARITHMETIC,
        LOSSESS_ARITHMETIC
    };
    
    FrameHeader();
    
    FrameHeader(const FrameHeader & rFrameHeader);
    
    FrameHeader & 
    operator= (const FrameHeader & rFrameHeader);
    
    void
    reset();
    
    void
    setWidth(unsigned short nWidth);
    
    unsigned short
    width()
    const;
    
    void
    setHeight(unsigned short nHeight);
    
    unsigned short
    height()
    const;
    
    unsigned short 
    width(int iComponent)
    const;
    
    unsigned short
    height(int nComponent)
    const;
    
    void
    setSamplePrecision(unsigned char nSamplePrecision);
    
    unsigned char 
    samplePrecision()
    const;
    
    void
    setEncoding(teEncoding eEncoding);
    
    teEncoding
    encoding()
    const;
    
    void
    setComponents(unsigned char nComponents);
    
    int
    components()
    const;
    
    int 
    componentIndexForIdentifier(int nIdentifier) 
    const;
    
    void
    setComponentIdentifier(int iComponent, unsigned char nComponentIdentifier);
    
    unsigned char
    componentIdentifier(int iComponent)
    const;
    
    void
    setSamplingFactors(int iComponent, unsigned char vSamplingFactors);
    
    unsigned char
    samplingFactors(int iComponent)
    const;
    
    void
    setHorizontalSamplingFactor(int iComponent, int nHorizontalSamplingFactor);
    
    int
    horizontalSamplingFactor(int iComponent)
    const;
    
    int
    maximumHorizontalSamplingFactor()
    const;
    
    void
    setVerticalSamplingFactor(int iComponent, int nVerticalSamplingFactor);
    
    int
    verticalSamplingFactor(int iComponent)
    const;
    
    int
    maximumVerticalSamplingFactor()
    const;
    
    void
    setQuantizationTableDestinationSelector(int iComponent, unsigned char nQuantizationTableDestinationSelector);
    
    unsigned char 
    quantizationTableDestinationSelector(int iComponent)
    const;
    
    static
    teEncoding
    GetEncoding(int nMarker);
        
protected:  
    ComponentSpecification & 
    componentSpecification(int iComponent);
    
    const
    ComponentSpecification & 
    componentSpecification(int iComponent)
    const;
    
private:
    teEncoding      eEncoding_;

    unsigned char   nSamplePrecision_;

    unsigned short  nHeight_;
    unsigned short  nWidth_;
    
    unsigned char   nComponents_;
    std::vector<ComponentSpecification> aComponentSpecifications_;
};

class QuantizationTable
{
public:
    typedef enum {
        PRECISION_8_BIT = 0,
        PRECISION_16_BIT = 1,
    } QuantizationTablePrecision;

    QuantizationTablePrecision nPrecision;
    unsigned char nIdentifier;

    union {
        unsigned char lowp[64];
        unsigned short highp[64];
    } aTable;

public:
    QuantizationTable();
    
    QuantizationTable(const QuantizationTable &rQuantizationTable);
    
    QuantizationTable& operator=(const QuantizationTable &rQuantizationTable);
    
    void setQualityLuminance(int quality, bool allow16Bit);
    
    void setQualityChrominance(int quality, bool allow16Bit);

private:
    static const int ZIGZAG_ORDER[];
    static const unsigned char LUMINANCE_BASE_TABLE[];
    static const unsigned char CHROMINANCE_BASE_TABLE[];

    void setQualityQuantizationTable(int quality, const unsigned char* pBaseTable, bool allow16Bit);
};

class CodecJPEGHuffmanTable
{
public:
    CodecJPEGHuffmanTable();
    
    CodecJPEGHuffmanTable(const CodecJPEGHuffmanTable & rHuffmanTable);
    
    CodecJPEGHuffmanTable &
    operator= (const CodecJPEGHuffmanTable & rHuffmanTable);

    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
    
    void setDefaultLuminanceDC();
    void setDefaultLuminanceAC();
    void setDefaultChrominanceDC();
    void setDefaultChrominanceAC();
    void print(int count) const;
    void print() const{
        print(256);
    }

private:
    static const unsigned char DEFAULT_LUMINANCE_AC_TABLE[];
    static const unsigned char DEFAULT_LUMINANCE_DC_TABLE[];
    static const unsigned char DEFAULT_CHROMINANCE_AC_TABLE[];
    static const unsigned char DEFAULT_CHROMINANCE_DC_TABLE[];

};

struct ScanHeader
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
};

class Scan
{    
public:
    //explicit
    Scan();

    void
    copyTablesAndRestartState(const Scan &rScan);
    
    void
    setRestartInterval(int nRestartInterval);

    int
    restartInterval()
    const;
    
    ScanHeader &
    scanHeader();
    
    const
    ScanHeader &
    scanHeader()
    const;
    
    int
    components()
    const;

    CodecJPEGHuffmanTable &
    huffmanTableDC(int iIndex);
    
    const
    CodecJPEGHuffmanTable &
    huffmanTableDC(int iIndex)
    const;

    CodecJPEGHuffmanTable &
    huffmanTableAC(int iIndex);

    const
    CodecJPEGHuffmanTable &
    huffmanTableAC(int iIndex)
    const;

    int
    huffmanTableSelectorDC(int iComponent)
    const;
    
    int 
    huffmanTableSelectorAC(int iComponent)
    const;
    
    const
    CodecJPEGHuffmanTable &
    huffmanTableForComponentDC(int iComponent)
    const;
    
    const
    CodecJPEGHuffmanTable &
    huffmanTableForComponentAC(int iComponent)
    const;
    
    void
    addComment(const std::string & rCommen);
    
    void
    addApplicationData(int iIndex, const std::string & rData);
    
    /// See the comment below.
    void
    setBuffer(unsigned char const * buf, size_t len);
    
    const
    unsigned char *
    bufferData()
    const;
    
    int
    bufferSize()
    const;
    
    
private:
    typedef std::vector<std::string>   taComments;
    
    taComments                  aComments_;
    std::vector<std::string>    aApplicationData_[16];
    
    int                         nRestartInterval_;
    
    /// Pointer to and length of compressed (Huffman encoded) scan data.
    // This pointer has to point to memory in oImageFileBuffer_.
    // The pointee is considered to use nppiJpegDecodeScanDeadzoneSize()
    // usable memory after aBuffer_ + nBufferLength_; 
    unsigned char const *       aBuffer_;
    size_t                      nBufferLength_;
    
    ScanHeader                  oScanHeader_;

public:   
    CodecJPEGHuffmanTable                aHuffmanTables_[8];
    CodecJPEGHuffmanTable              * apHuffmanTables_[8];
};

#endif // CODEC_JPEG_H_
