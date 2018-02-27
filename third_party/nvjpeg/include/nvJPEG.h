 /* Copyright 2009-2018 NVIDIA Corporation.  All rights reserved.
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

#ifndef NV_JPEG_HEADER
#define NV_JPEG_HEADER

#define NVJPEGAPI

#if defined(__cplusplus)
  extern "C" {
#endif


/**
 * \file nvJPEG.h
 * Type definitions and macros for nvJPEG library.
 */

/**
 * Retrieve the image info, including channel, width and height of each component.
 * If the image is 1-channel, only widthY and heightY are valid. The other two groups
 * are set to 0.
 * If the image is 3-channel, all three groups are valid.
 * The user should call this function to allocate the appropriate buffer
 * before calling the decoder.
 *
 * \param pData 	 	Pointer to the buffer containing the jpeg image to be decoded.
 * \param nLength 	    Length of the jpeg image buffer.
 * \param nComponent    Number of componenets of the image, currently only supports 1-channel or 3-channel.
 * \param nWidthY  	    Width of Y component.
 * \param nHeightY  	Height to Y component.
 * \param nWidthCb  	Width ofCbY component.
 * \param nHeightCb 	Height to Cb component.
 * \param nWidthCr  	Width ofCrY component.
 * \param nHeightCr 	Height to Cr component.
 *
 * \return
 */
NVJPEGAPI
int
nvjpegGetImageInfo(const unsigned char * pData, unsigned int nLength,
				   int * nComponent,
				   int * nWidthY,  int * nHeightY,
				   int * nWidthCb, int * nHeightCb,
				   int * nWidthCr, int * nHeightCr);

/**
 * Decoder path for a single image.
 * Before calling this function, user needs to call the \ref nvjpegGetImageInfo
 * to determine the component and size information of the image and allocate the
 * buffer accordingly.
 *
 * \param pData     Pointer to the buffer containing the jpeg image to be decoded.
 * \param nLength 	Length of the jpeg image buffer.
 * \param pY  		Pointer to Y component.
 * \param nStepY  	Step to Y component.
 * \param pCb  		Pointer to Cb component.
 * \param nStepCb  	Step to Cb component.
 * \param pCr  		Pointer to Cr component.
 * \param nStepCr  	Step to Cr component.
 *
 * \return 0 if successful, non-0 otherwise.
 */
NVJPEGAPI
int
nvjpegDecode(const unsigned char * pData, unsigned int nLength,
	         unsigned char * pY,  int nStepY,
	         unsigned char * pCb, int nStepCb,
	         unsigned char * pCr, int nStepCr);

/*@}*/

#if defined(__cplusplus)
  }
#endif

#endif /* NV_JPEG_HEADER */
