#include "huffman_decoder.h"

#include "codec_jpeg.h"
#include "debug.h"
#include "jpeg_parser.h"

#include <cassert>

//
/// HuffmanDecoderState
//

HuffmanDecoderState::HuffmanDecoderState() {
  // Allocate host-side Huffman table buffers
  // 
  // TODO(Trevor): This is a hack to get around the fact that NPP does not
  // provide an implementation of the NppiDecodeHuffmanSpec struct externally.
  array<unsigned int, 16> tmp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    NPP_CHECK_NPP(nppiDecodeHuffmanSpecInitAllocHost_JPEG((Npp8u*)tmp.data(),
            nppiDCTable, &apHuffmanDCTable[i]),
        "Failed to allocate host DC huffman table buffer");
    NPP_CHECK_NPP(nppiDecodeHuffmanSpecInitAllocHost_JPEG((Npp8u*)tmp.data(),
            nppiACTable, &apHuffmanACTable[i]),
        "Failed to allocate host AC huffman table buffer");
  }
}

HuffmanDecoderState::~HuffmanDecoderState() {
  for (int i = 0; i < 3; ++i) {
    NPP_CHECK_NPP(nppiDecodeHuffmanSpecFreeHost_JPEG(apHuffmanDCTable[i]),
        "Failed to free DC Huffman table");
    NPP_CHECK_NPP(nppiDecodeHuffmanSpecFreeHost_JPEG(apHuffmanACTable[i]),
        "Failed to free AC Huffman table");
  }
}
        
//
/// HuffmanDecoder
//

HuffmanDecoder::HuffmanDecoder(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
    vector<HostBlocksDCT> *dctCoeffs) : oFrameHeader_(jpeg.frameHeader),
                                        apScans_(jpeg.scans),
                                        aHostBlocksDCT_(*dctCoeffs),
                                        apHuffmanDCTable_(state->apHuffmanDCTable),
                                        apHuffmanACTable_(state->apHuffmanACTable) {
  assert(dctCoeffs->size() == jpeg.components);
}

void HuffmanDecoder::decode() {
  TimeRange _tr("huffman");
    
  {
    TimeRange _tr1("ver");
    // TODO(Trevor): Can we remove this? May be just an ICE thing
    // 
    // Check if JPEG has all necessary initial scans
    // (so it doesn't leak data from previous images).
    // We don't check if there are all necessary refining scans though
    // (It means the file is corrupted, but does not pose a safety risk).
    std::vector<Npp64u> aComponentCoeffsMask(frameHeader().components(), 0);
    for (int iScan = 0; iScan < this->scans(); ++iScan)
    {
      ScanHeader const& header = scan(iScan).scanHeader();
      if (header.nSs > header.nSe || header.nSs >= 64 || header.nSe >= 64)
      {
        NPP_CHECK_NPP(NPP_ERROR, "Invalid Ss and Se values in scan header");
      }
      if ((header.nA >> 4) == 0)
      {
        // Initial scan
        for (int iScanComp = 0; iScanComp < scan(iScan).components(); ++iScanComp)
        {
          int iComp = frameHeader().componentIndexForIdentifier(
              header.aComponentSelector[iScanComp]
              );
          if (iComp == -1)
          {
            NPP_CHECK_NPP(NPP_ERROR, "Invalid component identifier in scan.");
          }
          for (unsigned char shift = header.nSs; shift <= header.nSe; ++shift)
            aComponentCoeffsMask[iComp] |= 1llu << shift;
        }
      }
    }
    for (int iComp = 0; iComp < frameHeader().components(); ++iComp)
    {
      if (aComponentCoeffsMask[iComp] != ~0llu)
      {
        NPP_CHECK_NPP(NPP_ERROR, "Some scans are not present in the file");
      }
    }
  }
    
  // Allocate space for the dct blocks on host
  for (int i = 0; i < frameHeader().components(); ++i)
  {
    TimeRange _tr2("alloc");
    NppiSize oBlocks = {interleavedComponentSize(i).width/8,
                        interleavedComponentSize(i).height/8};
        
    aHostBlocksDCT_[i].resize(oBlocks.width, oBlocks.height);
  }

  {
    TimeRange _tr3("decode");
    for (unsigned int iScan = 0; iScan < this->scans(); ++iScan)
    {
      Scan & rScan = scan(iScan);

      for (int i = 0; i < rScan.components(); ++i)
      {
        TimeRange _tr4("spec_init");
        if (rScan.scanHeader().nSs == 0)
        {
          NPP_CHECK_NPP(nppiDecodeHuffmanSpecInitHost_JPEG(
                  rScan.huffmanTableForComponentDC(i).aCodes, 
                  nppiDCTable, apHuffmanDCTable_[i]),
              "Failed to init host DC huffman table buffer");
        }

        if (rScan.scanHeader().nSe > 0)
        {
          NPP_CHECK_NPP(nppiDecodeHuffmanSpecInitHost_JPEG(
                  rScan.huffmanTableForComponentAC(i).aCodes, 
                  nppiACTable, apHuffmanACTable_[i]),
              "Failed to init host AC huffman table buffer");
        }
      }

      switch (rScan.components()) 
      {
      case 1: 
      {
        int nComponent = frameHeader().componentIndexForIdentifier(
            rScan.scanHeader().aComponentSelector[0]);
        if (nComponent > aHostBlocksDCT_.size()) {
          throw std::runtime_error("Woops, something got broken ripping this out of ice");
        }
        NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R(rScan.bufferData(),
                rScan.bufferSize(), rScan.restartInterval(), 
                rScan.scanHeader().nSs, rScan.scanHeader().nSe, 
                rScan.scanHeader().nA >> 4, rScan.scanHeader().nA & 0x0f,
                aHostBlocksDCT_[nComponent].blockData(),
                aHostBlocksDCT_[nComponent].lineStep(),
                apHuffmanDCTable_[0], apHuffmanACTable_[0], 
                nonInterleavedComponentSize(nComponent)),
            "Failed to decode Huffman on host - 1 component");

        break;
      }
      case 3: 
      {
        TimeRange _tr5("run_decode");
        NppiSize aComponentSizes[3];
        Npp16s * aBlocks[3];
        Npp32s aLineSteps[3];
        for (int iComponent = 0; iComponent < 3; ++iComponent)
        {
          aComponentSizes[iComponent] = interleavedComponentSize(iComponent);
          aBlocks[iComponent] = aHostBlocksDCT_[iComponent].blockData();
          aLineSteps[iComponent] = aHostBlocksDCT_[iComponent].lineStep();
        }

        NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(rScan.bufferData(),
                rScan.bufferSize(), rScan.restartInterval(), 
                rScan.scanHeader().nSs, rScan.scanHeader().nSe, 
                rScan.scanHeader().nA >> 4, rScan.scanHeader().nA & 0x0f,
                aBlocks, aLineSteps, apHuffmanDCTable_.data(),
                apHuffmanACTable_.data(), aComponentSizes),
            "Failed to decode Huffman on host - 3 component");
      }
      break;
      default:
        NPP_CHECK_NPP(NPP_ERROR, "General error in CodecJPEG - 0x1688");
      }
    }
  }
}

unsigned int HuffmanDecoder::scans() const {
  return static_cast<unsigned int>(apScans_.size());
}

Scan& HuffmanDecoder::scan(unsigned int iScan) {
  if (iScan >= scans())
    throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Scan Index Out-of-Range");
        
  return *apScans_[iScan];
}

const Scan& HuffmanDecoder::scan(unsigned int iScan) const {
  if (iScan >= scans())
    throw ICE::ExceptionJPEG(ICE::UNSUPPORTED_JPEG_STATUS, "Scan Index Out-of-Range");        
  return *apScans_[iScan];
}

// FrameHeader& HuffmanDecoder::frameHeader() {
//     return oFrameHeader_;
// }

const FrameHeader& HuffmanDecoder::frameHeader() const {
  return oFrameHeader_;
}

NppiSize HuffmanDecoder::interleavedComponentSize(int nComponent) const {
  NppiSize oResult = {DivUp(oFrameHeader_.width(nComponent),
        oFrameHeader_.horizontalSamplingFactor(nComponent) * 8) *
                      (oFrameHeader_.horizontalSamplingFactor(nComponent) * 8),
                      DivUp(oFrameHeader_.height(nComponent),
                          oFrameHeader_.verticalSamplingFactor(nComponent) * 8) *
                      (oFrameHeader_.verticalSamplingFactor(nComponent) * 8)};
  return oResult;
}

NppiSize HuffmanDecoder::nonInterleavedComponentSize(int nComponent) const {
  NppiSize oResult = {DivUp(oFrameHeader_.width(nComponent),  8) * 8,
                      DivUp(oFrameHeader_.height(nComponent), 8) * 8};
  return oResult;
}
