#ifndef HOST_BUFFER_H_
#define HOST_BUFFER_H_

#include <cstdlib> // size_t

class HostBuffer {
public:
  explicit HostBuffer(unsigned int nSize = 0);
    
  ~HostBuffer();
   
  void resize(unsigned int nSize);
    
  unsigned char * data();
    
  const unsigned char * data() const;
    
  unsigned int size() const;

private:
  unsigned char * pData_;
  unsigned int    nSize_;
};

class HostBlocksDCT
{
public:
  HostBlocksDCT(unsigned int nWidth = 0, unsigned int nHeight = 0);
  HostBlocksDCT(unsigned int nWidth, unsigned int nHeight, short *ptr, size_t size);
  
  ~HostBlocksDCT();
    
  void
  resize(unsigned int nWidth, unsigned int nHeight);

  static
  size_t
  get_size(unsigned int nWidth, unsigned int nHeight);
    
  short *
  blockData();
    
  const
  short *
  blockData()
    const;
    
  unsigned int
  lineStep()
    const;

private:
  // Indicates whether or not this object owns the buffer it stores.
  // This enables us to wrap pre-allocated buffers of the correct
  // size and avoid unnescessary memory copies.
  bool owned_;
  
  unsigned int    nWidth_;
  unsigned int    nHeight_;
  short         * pBlocks_;
  size_t          nSize_;
};

#endif // HOST_BUFFER_H_
