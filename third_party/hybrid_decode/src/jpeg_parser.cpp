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
#include "jpeg_parser.h"

#include "debug.h"
#include "input_stream_jpeg.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>

#define USE_EXPERIMENTAL_DCT

size_t nppiJpegDecodeGetScanDeadzoneSize() {
    return 2 * 4 * 64 + 8;

    // We are detecting out of bounds condition while scanning only on block boundaries,
    // so we have to be able to safely read memory for one block after scan data ends.
    // Max block size: 64 coeffs, max 32 bits each, times 2 because of stuffing.
    // Additional 8 bytes because of bitstuffing and speculatively unrolled pipelines.
}

JpegParser::JpegParser(ParsedJpeg *jpeg, JpegParserState *state) :
    jpeg_(*jpeg), oFrameHeader_(jpeg->frameHeader),
    nRestartInterval_(jpeg->restartInterval),
    aQuantizationTables_(state->quantizationTables),
    apQuantizationTables_(state->pQuantizationTables),
    apScans_(jpeg->scans), aComments_(jpeg->comments),
    aApplicationData_(jpeg->applicationData) {
    // clear the input ParsedJpeg state
    //
    // TODO(Trevor): Freeing all this memory is not ideal. Is there
    // a way to avoid deallocating all these scans and reuse them?
    for (auto &scan : jpeg->scans) delete scan;
    jpeg->scans.clear();
}

void JpegParser::parse(InputStreamJPEG *inputJpeg) {
    TimeRange _tr("parse");
    
    teParseState eParseState = START_JPEG;
    int nMarker = 0;
    
    std::unique_ptr<Scan> apScan(new Scan());

    // Parsing and Huffman Decoding (on host)
    while (eParseState != END_JPEG && nMarker != -1)
    {
        nMarker = inputJpeg->nextMarker();
        switch (eParseState)
        {
        case START_JPEG:
        {
            if (nMarker != START_0F_IMAGE_JPEG)
                throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                  "Invalid marker for start jpeg: " + std::to_string(nMarker));
            else
                eParseState = FRAME_HEADER_JPEG;
        }
        break;
        case FRAME_HEADER_JPEG:
        {
            switch (nMarker)
            {
                // the following are all optional front matter preceding
                // the actual frame header data
            case DEFINE_QUANTIZATION_TABLE_JPEG:
                inputJpeg->readQuantizationTables(aQuantizationTables_.data(),
                        apQuantizationTables_.data(),
                        sizeof(aQuantizationTables_) /
                        sizeof(aQuantizationTables_[0]));
                break;
            case DEFINE_HUFFMAN_TABLE_JPEG:
                inputJpeg->readHuffmanTables(apScan->aHuffmanTables_, 
                        apScan->apHuffmanTables_,
                        sizeof(apScan->aHuffmanTables_) / sizeof(apScan->aHuffmanTables_[0]));
                break;
            case DEFINE_ARITHMETIC_CODING_CONDITIONING:
                throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                        "Arithmetic Coding Not Supported");
                break;
            case DEFINE_RESTART_INTERVAL_JPEG:
                inputJpeg->readRestartInterval(nRestartInterval_);
                apScan->setRestartInterval(nRestartInterval_);
                break;
            case COMMENT_JPEG:
            {
                std::string sComment;
                inputJpeg->readComment(sComment);    
                aComments_.push_back(sComment);
            }        
            break;
            case APPLICATION_SEGMENT_0_JPEG:
            case APPLICATION_SEGMENT_1_JPEG:
            case APPLICATION_SEGMENT_2_JPEG:
            case APPLICATION_SEGMENT_3_JPEG:
            case APPLICATION_SEGMENT_4_JPEG:
            case APPLICATION_SEGMENT_5_JPEG:
            case APPLICATION_SEGMENT_6_JPEG:
            case APPLICATION_SEGMENT_7_JPEG:
            case APPLICATION_SEGMENT_8_JPEG:
            case APPLICATION_SEGMENT_9_JPEG:
            case APPLICATION_SEGMENT_10_JPEG:
            case APPLICATION_SEGMENT_11_JPEG:
            case APPLICATION_SEGMENT_12_JPEG:
            case APPLICATION_SEGMENT_13_JPEG:
            case APPLICATION_SEGMENT_14_JPEG:
            case APPLICATION_SEGMENT_15_JPEG:
            {
                std::string aData;
                inputJpeg->readApplicationData(aData);
                aApplicationData_[nMarker & 0x0F].push_back(aData);                     
            }
            break;
            // the next two cases deal with the two supported types
            // of JPEG encodings and read their respective frame-header
            // data
            case START_OF_FRAME_HUFFMAN_BASELINE_DCT_JPEG:
            case START_OF_FRAME_HUFFMAN_PROGRESSIVE_DCT_JPEG:
            {
                // set encoding type contained in start-of-frame marker
                oFrameHeader_.setEncoding(FrameHeader::GetEncoding(nMarker));
                // Baseline or Progressive Frame Header
                inputJpeg->readFrameHeader(oFrameHeader_);
                        
                // Frame-header parsing is complete
                eParseState = SCAN_JPEG;
            }
            break;
            default:
                // this is the catch-all for unsupported markers
                throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                        "Unsupported JPEG Format");
            }
        }
        break;
        case SCAN_JPEG:
        {
            switch (nMarker)
            {
                // the following are all optional front matter preceding
                // the actual scan header data
            case DEFINE_QUANTIZATION_TABLE_JPEG:
                inputJpeg->readQuantizationTables(aQuantizationTables_.data(),
                        apQuantizationTables_.data(),
                        sizeof(aQuantizationTables_) /
                        sizeof(aQuantizationTables_[0]));
                break;
            case DEFINE_HUFFMAN_TABLE_JPEG:
                inputJpeg->readHuffmanTables(apScan->aHuffmanTables_, 
                        apScan->apHuffmanTables_,
                        sizeof(apScan->aHuffmanTables_) / sizeof(apScan->aHuffmanTables_[0]));
                break;
            case DEFINE_ARITHMETIC_CODING_CONDITIONING:
                throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                        "Arithmetic Coded JPEG Not Supported");
                break;
            case DEFINE_RESTART_INTERVAL_JPEG:
            {
                int nRestartInterval;
                inputJpeg->readRestartInterval(nRestartInterval);
                apScan->setRestartInterval(nRestartInterval);
            }
            break;
            case COMMENT_JPEG:
            {
                std::string sComment;
                inputJpeg->readComment(sComment);    
                apScan->addComment(sComment);
            }        
            break;
            case APPLICATION_SEGMENT_0_JPEG:
            case APPLICATION_SEGMENT_1_JPEG:
            case APPLICATION_SEGMENT_2_JPEG:
            case APPLICATION_SEGMENT_3_JPEG:
            case APPLICATION_SEGMENT_4_JPEG:
            case APPLICATION_SEGMENT_5_JPEG:
            case APPLICATION_SEGMENT_6_JPEG:
            case APPLICATION_SEGMENT_7_JPEG:
            case APPLICATION_SEGMENT_8_JPEG:
            case APPLICATION_SEGMENT_9_JPEG:
            case APPLICATION_SEGMENT_10_JPEG:
            case APPLICATION_SEGMENT_11_JPEG:
            case APPLICATION_SEGMENT_12_JPEG:
            case APPLICATION_SEGMENT_13_JPEG:
            case APPLICATION_SEGMENT_14_JPEG:
            case APPLICATION_SEGMENT_15_JPEG:
            {
                std::string aData;
                inputJpeg->readApplicationData(aData);
                apScan->addApplicationData(nMarker & 0x0F, aData);
            }
            break;
            case START_OF_SCAN_JPEG:
            {
                // Scan
                inputJpeg->readScanHeader(apScan->scanHeader());

                size_t nStartPosition = inputJpeg->current();
                int nAfterScanMarker = inputJpeg->nextMarkerFar();

                if (apScan->restartInterval() > 0)
                {
                    while (RESTART_0_JPEG <= nAfterScanMarker &&
                            nAfterScanMarker <= RESTART_7_JPEG)
                    {
                        // This is a restart marker, go on
                        nAfterScanMarker = inputJpeg->nextMarkerFar();
                    }
                }

                if (inputJpeg->current() - nStartPosition <= 2)
                {
                    // no data in the scan
                    throw std::runtime_error("Bad JPEG. (case H)");
                }

                size_t nLength = inputJpeg->current() - nStartPosition - 2;
                inputJpeg->seek(nStartPosition + nLength);
                apScan->setBuffer(inputJpeg->getBufferAtOffset(nStartPosition), nLength);

                this->addScan(apScan.release());
                apScan = std::unique_ptr<Scan>(new Scan());
                apScan->copyTablesAndRestartState(this->scan(this->scans()-1));
            }
            break;
            case END_OF_IMAGE_JPEG:
                eParseState = END_JPEG;
                break;
            default:
                // this is the catch-all for unsupported markers
                throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                        "Unsupported JPEG Format");
            }
        }
        break;
        default:
            throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS,
                    "Invalid JPEG File");
        }
    }

    // Store all convenience parameters in the parsed jpeg
    jpeg_.sRatio = getSamplingRatio();
    jpeg_.imgDims = {(int)oFrameHeader_.width(0), (int)oFrameHeader_.height(0)};
    jpeg_.components = oFrameHeader_.components();

    // NOTE: See TODO below
    assert(jpeg_.components < 4);
    
    for (int i = 0; i < jpeg_.components; ++i) {
        // The width & height of the dct coeffs is expressed in 8x8 blocks. We calculate the
        // number of 8x8 blocks, rounded up to the nearest multiple of 8 * samplingFactor
        // for that dim. The line step is in bytes. The NPP iDCT function takes in DCT
        // coeffs in 64x1 blocks (i.e. an 8x8 block layed out linearly in memory).
        // Thus, we express the dims of the dct coeffs in this way.
        int width =
            DivUp(oFrameHeader_.width(i), oFrameHeader_.horizontalSamplingFactor(i) * 8) *
            oFrameHeader_.horizontalSamplingFactor(i) * 8;
        int height =
            DivUp(oFrameHeader_.height(i), oFrameHeader_.verticalSamplingFactor(i) * 8) *
            oFrameHeader_.verticalSamplingFactor(i) * 8;
        int block_width = width / 8;
        int block_height = height / 8;
        int lineStep = block_width * 64 * sizeof(short);
        
        // needed to allocate mem for huffman_decoder to use, and to copy dct coeffs to gpu
        jpeg_.dctSize[i] = lineStep * DivUp(block_height, 2) * 2;
        jpeg_.dctLineStep[i] = lineStep;

        // needed for roi for idct
        jpeg_.yCbCrDims[i] = {width, height};

        // TODO(Trevor): Can there be more than 3 components in the case that we have an
        // alpha channel? Test this to make sure we don't run out of quantization tables
        // here
        //
        // Save the quantization table for each component
        storeQuantTable(&jpeg_.quantTables[i], i);
    }

    // Set the dims for all non-existent components to 0
    for (int i = jpeg_.components; i < 3; ++i) {
      jpeg_.dctSize[i] = 0;
      jpeg_.dctLineStep[i] = 0;
      jpeg_.yCbCrDims[i] = {0, 0};
    }
}

void JpegParser::addScan(Scan * pScan) {
    apScans_.push_back(pScan);
}

unsigned int JpegParser::scans() const {
    return static_cast<unsigned int>(apScans_.size());
}

Scan& JpegParser::scan(unsigned int iScan) {
    if (iScan >= scans())
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Scan Index Out-of-Range");
        
    return *apScans_[iScan];
}

const Scan& JpegParser::scan(unsigned int iScan) const {
    if (iScan >= scans())
        throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Scan Index Out-of-Range");        
    return *apScans_[iScan];
}

QuantizationTable& JpegParser::quantizationTable(int iIndex) {
    return aQuantizationTables_[iIndex];
}

const QuantizationTable& JpegParser::quantizationTable(int iIndex) const {
    return aQuantizationTables_[iIndex];
}

ComponentSampling JpegParser::getSamplingRatio() {
    // NOTE: This function must be called after `parse`
    // so that the jpeg data is available
    int nComponents = oFrameHeader_.components();
    ComponentSampling eComponentSampling = YCbCr_UNKNOWN;

    // JPEG Encoder Parameters
    if (nComponents == 1)
        eComponentSampling = YCbCr_444;
    else {
        int yWidth = oFrameHeader_.width(0);
        int yHeight = oFrameHeader_.height(0);
        int cbWidth = oFrameHeader_.width(1);
        int cbHeight = oFrameHeader_.height(1);
        int crWidth = oFrameHeader_.width(2);
        int crHeight = oFrameHeader_.height(2);
     
        // examine input sampling factor
        //
        // TODO(Trevor): Copied this code from ICE, why does this
        // always check if the different is less than 3?
        if(yWidth == cbWidth) {
            if (yHeight == cbHeight) {
                // cout << "selected 444" << endl;
                eComponentSampling = YCbCr_444;
            } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
                // cout << "selected 440" << endl;
                eComponentSampling = YCbCr_440;
            }
        }
        else if (abs(static_cast<float>(yWidth - 2 * cbWidth)) < 3) { 
            if (yHeight == cbHeight) {
                // cout << "selected 422" << endl;
                eComponentSampling = YCbCr_422;
            } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
                // cout << "selected 420" << endl;
                eComponentSampling = YCbCr_420;
            }
        } 
        else if (abs(static_cast<float>(yWidth - 4 * cbWidth)) < 4) {
            if (yHeight == cbHeight) {
                // cout << "selected 411" << endl;
                eComponentSampling = YCbCr_411;
            } else if (abs(static_cast<float>(yHeight - 2 * cbHeight)) < 3) {
                // cout << "selected 410" << endl;
                eComponentSampling = YCbCr_410;
            }
        }

        // make sure we found a valid sampling ratio
        if (eComponentSampling == YCbCr_UNKNOWN) {
            std::ostringstream os;
            os << "invalid image - Y:" << yWidth << "x" << yHeight <<
                "  Cr:"<< cbWidth << "x" << cbHeight << "  Cr:"<<
                crWidth << "x" << crHeight;
            throw std::runtime_error(os.str());
        }
    }
    return eComponentSampling;
}

void JpegParser::storeQuantTable(QuantizationTable *table, int component) {
    unsigned char idx = oFrameHeader_.quantizationTableDestinationSelector(component);
    QuantizationTable &src = quantizationTable(idx);
#ifdef USE_EXPERIMENTAL_DCT
    // TODO(Trevor): Can we save this info/get it somewhere we can reference it
    // repetedly without having to always make these calls?
    // int device;
    // cudaDeviceProp props;
    // CHECK_CUDA(cudaGetDevice(&device), "Failed to get device id");
    // CHECK_CUDA(cudaGetDeviceProperties(&props, device),
    //         "Failed to get device properties");
    // 
    // TODO(Trevor): We don't have global state for the lib, and repeatedly querying
    // the device props kills performance. For now we only support >=SM3
    bool sm3xOrMore = true; // = props.major >= 3;
    
    if (sm3xOrMore) {
        Npp8u aZigzag[] = {
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        };

        table->nPrecision = src.nPrecision;
        table->nIdentifier = src.nIdentifier;
        switch(src.nPrecision) {
        case QuantizationTable::PRECISION_8_BIT:
            for (int k = 0 ; k < 32 ; ++k) {
                table->aTable.lowp[2*k+0] = src.aTable.lowp[aZigzag[k+ 0]];
                table->aTable.lowp[2*k+1] = src.aTable.lowp[aZigzag[k+32]];
            }
            break;
        case QuantizationTable::PRECISION_16_BIT:            
            for(int k = 0 ; k < 32 ; ++k) {                
                table->aTable.highp[2*k+0] = src.aTable.highp[aZigzag[k+ 0]];
                table->aTable.highp[2*k+1] = src.aTable.highp[aZigzag[k+32]];
            }
            break;
        }
    } else
#endif
    {
        // Just copy the quant table as is
        (*table) = src;
    }
}
