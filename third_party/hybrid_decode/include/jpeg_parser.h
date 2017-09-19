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
#ifndef JPEG_PARSER_H_
#define JPEG_PARSER_H_

#include "codec_jpeg.h"

#include <npp.h>

#include <array>
#include <vector>

using std::array;
using std::vector;

class InputStreamJPEG;

// Returns the size of the deadzone needed after
// the input jpeg string that is to be parsed
//
// TODO(Trevor): Maybe put this somewhere else
// where it makes more sense
size_t nppiJpegDecodeGetScanDeadzoneSize();

// Indicates the sampling ratio for the
// chrominance components of decoded JPEGs.
enum ComponentSampling
{
  YCbCr_444,
  YCbCr_440,
  YCbCr_422,
  YCbCr_420,
  YCbCr_411,
  YCbCr_410,
  YCbCr_UNKNOWN
};

/**
 * Stores all objects from a raw Jpeg string that are 
 * needed to perform the rest of the Jpeg decode
 */
struct ParsedJpeg {
  // Short-cut data for jpeg decoding functions
  ComponentSampling sRatio;
  NppiSize imgDims;
  int components;

  // The size in bytes of memory needed for DCT coeffs & the step in memory
  array<int, 3> dctSize;
  array<int, 3> dctLineStep;
  
  // The dims of the decoded YCbCr image components
  array<NppiSize, 3> yCbCrDims; 
    
  FrameHeader frameHeader;
  int restartInterval;

  // Host-side quantization tables. Stored in zig-zag format for >=SM3
  array<QuantizationTable, 3> quantTables;

  // Scans also contain their Huffman tables
  vector<Scan*> scans;

  // TODO(Trevor): Theses members are not used anywhere
  // in the decoding process. Can we remove them?
  vector<string> comments;
  array<vector<string>, 16> applicationData;

  // Stores raw jpeg string w/ deadzone memory
  // Contains the underlying memory for the Scans
  HostBuffer imageBuffer;
};

/**
 * Contains all state variables for the JpegParser that don't need
 * to be cached for use in the rest of the Jpeg decode
 */
struct JpegParserState {
  // Stores raw jpeg string w/ deadzone memory
  // HostBuffer imageBuffer;

  // Memory for the quant tables before they're
  // processed for use in the iDCT
  array<QuantizationTable, 4> quantizationTables;
  array<QuantizationTable*, 4> pQuantizationTables;
};

/**
 * Object to parse an `InputStreamJPEG` into a `ParsedJpeg` object.
 * Maintains no internal state. All state handed into constructor
 * and stored by reference.
 */
class JpegParser {
public:
  JpegParser(ParsedJpeg *jpeg, JpegParserState *state);
  ~JpegParser() = default;

  void parse(InputStreamJPEG *inputJpeg);
    
  enum teParseState {
    START_JPEG,
    FRAME_HEADER_JPEG,
    SCAN_JPEG,
    SCANE_HEADER_JPEG,
    ENTROPY_CODED_SEGMENT_JPEG,
    END_JPEG
  };
    
private:
  // Helpers to access scans
  void addScan(Scan * pScan);
  unsigned int scans() const;
  Scan& scan(unsigned int iScan);
  const Scan& scan(unsigned int iScan) const;

  // Helpers to access quant tables
  QuantizationTable& quantizationTable(int iIndex);
  const QuantizationTable& quantizationTable(int iIndex) const;
    
  // Helper to find the sampling ratio of the parsed jpeg
  ComponentSampling getSamplingRatio();

  // Helper to prepare the quant tables for use in the iDCT
  void storeQuantTable(QuantizationTable *table, int component);
    
  ParsedJpeg &jpeg_;
    
  FrameHeader &oFrameHeader_;
  int &nRestartInterval_;

  // Host-side quantization tables. Stored in zig-zag format for >=SM3
  array<QuantizationTable, 4> &aQuantizationTables_;
  array<QuantizationTable*, 4> &apQuantizationTables_;

  // Scans also contain their Huffman tables
  vector<Scan*> &apScans_;

  // TODO(Trevor): Theses members are not used anywhere
  // in the decoding process. Can we remove them?
  vector<string> &aComments_;
  array<vector<string>, 16> &aApplicationData_;
};

#endif // JPEG_PARSER_H_
