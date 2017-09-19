#ifndef HUFFMAN_DECODER_H_
#define HUFFMAN_DECODER_H_

#include "host_buffer.h"

#include "common.h"

#include <npp.h>

struct ParsedJpeg;
class FrameHeader;
class Scan;

struct HuffmanDecoderState {
  HuffmanDecoderState();
  ~HuffmanDecoderState();
    
  // Allocated on contruction. Used to avoid re-allocation on host
  array<NppiDecodeHuffmanSpec*, 3> apHuffmanDCTable;
  array<NppiDecodeHuffmanSpec*, 3> apHuffmanACTable;
};


class HuffmanDecoder {
public:
  HuffmanDecoder(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
      vector<HostBlocksDCT> *dctCoeffs);

    
  ~HuffmanDecoder() = default;

  void decode();
    
private:
  // Helpers to access members
  unsigned int scans() const;
  Scan& scan(unsigned int iScan);
  const Scan& scan(unsigned int iScan) const;
  // FrameHeader& frameHeader();
  const FrameHeader& frameHeader() const;

  // Helper to calculate the dims of an image component
  NppiSize interleavedComponentSize(int nComponent) const;
  NppiSize nonInterleavedComponentSize(int nComponent) const;

  const FrameHeader &oFrameHeader_;
  const vector<Scan*> &apScans_;
  vector<HostBlocksDCT> &aHostBlocksDCT_;

  array<NppiDecodeHuffmanSpec*, 3> &apHuffmanDCTable_;
  array<NppiDecodeHuffmanSpec*, 3> &apHuffmanACTable_;
};

    
    
#endif // HUFFMAN_DECODER_H_
