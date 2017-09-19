/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <npp.h>
#include <string>
#include <sstream>
#include <iostream>

#define FOO1(ans) { gpuAssert1((ans), __FILE__, __LINE__); }
inline void gpuAssert1(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
     fprintf(stderr,"CUDA error: %s at %s %d\n", cudaGetErrorString(code), file, line);
     if (abort) exit(code);
   }
}

#define FOO2(ans, str) { gpuAssert2((ans), (str), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *str,
        const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA error: %s at %s %d:%s\n", cudaGetErrorString(code), file, line, str);
        if (abort) exit(code);
    }
}

#define GET_MACRO(_1, _2, NAME,...) NAME
#define CHECK_CUDA(...) GET_MACRO(__VA_ARGS__, FOO2, FOO1)(__VA_ARGS__)


static const char *_cudaGetErrorEnum(NppStatus error)
{
    switch (error)
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "NPP_NOT_SUPPORTED_MODE_ERROR";

        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "NPP_RESIZE_NO_OPERATION_ERROR";

        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_BAD_ARG_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFF_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECT_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUAD_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEM_ALLOC_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_INPUT:
            return "NPP_INVALID_INPUT";

        case NPP_POINTER_ERROR:
            return "NPP_POINTER_ERROR";

        case NPP_WARNING:
            return "NPP_WARNING";

        case NPP_ODD_ROI_WARNING:
            return "NPP_ODD_ROI_WARNING";
#else

            // These are for CUDA 5.5 or higher
        case NPP_BAD_ARGUMENT_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFFICIENT_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECTANGLE_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUADRANGLE_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEMORY_ALLOCATION_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_HOST_POINTER_ERROR:
            return "NPP_INVALID_HOST_POINTER_ERROR";

        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

        case NPP_TEXTURE_BIND_ERROR:
            return "NPP_TEXTURE_BIND_ERROR";

        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "NPP_WRONG_INTERSECTION_ROI_ERROR";

        case NPP_NOT_EVEN_STEP_ERROR:
            return "NPP_NOT_EVEN_STEP_ERROR";

        case NPP_INTERPOLATION_ERROR:
            return "NPP_INTERPOLATION_ERROR";

        case NPP_RESIZE_FACTOR_ERROR:
            return "NPP_RESIZE_FACTOR_ERROR";

        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";


#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_MEMFREE_ERR:
            return "NPP_MEMFREE_ERR";

        case NPP_MEMSET_ERR:
            return "NPP_MEMSET_ERR";

        case NPP_MEMCPY_ERR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERR:
            return "NPP_MIRROR_FLIP_ERR";
#else

        case NPP_MEMFREE_ERROR:
            return "NPP_MEMFREE_ERROR";

        case NPP_MEMSET_ERROR:
            return "NPP_MEMSET_ERROR";

        case NPP_MEMCPY_ERROR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERROR:
            return "NPP_MIRROR_FLIP_ERROR";
#endif

        case NPP_ALIGNMENT_ERROR:
            return "NPP_ALIGNMENT_ERROR";

        case NPP_STEP_ERROR:
            return "NPP_STEP_ERROR";

        case NPP_SIZE_ERROR:
            return "NPP_SIZE_ERROR";

        case NPP_NULL_POINTER_ERROR:
            return "NPP_NULL_POINTER_ERROR";

        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

        case NPP_NOT_IMPLEMENTED_ERROR:
            return "NPP_NOT_IMPLEMENTED_ERROR";

        case NPP_ERROR:
            return "NPP_ERROR";

        case NPP_SUCCESS:
            return "NPP_SUCCESS";

        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "NPP_MISALIGNED_DST_ROI_WARNING";

        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

        case NPP_DOUBLE_SIZE_WARNING:
            return "NPP_DOUBLE_SIZE_WARNING";

        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
        /* These are 6.0 or higher */
        case NPP_LUT_PALETTE_BITSIZE_ERROR:
            return "NPP_LUT_PALETTE_BITSIZE_ERROR";

        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

        case NPP_QUALITY_INDEX_ERROR:
            return "NPP_QUALITY_INDEX_ERROR";

        case NPP_CHANNEL_ORDER_ERROR:
            return "NPP_CHANNEL_ORDER_ERROR";

        case NPP_ZERO_MASK_VALUE_ERROR:
            return "NPP_ZERO_MASK_VALUE_ERROR";

        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "NPP_NUMBER_OF_CHANNELS_ERROR";

        case NPP_COI_ERROR:
            return "NPP_COI_ERROR";

        case NPP_DIVISOR_ERROR:
            return "NPP_DIVISOR_ERROR";

        case NPP_CHANNEL_ERROR:
            return "NPP_CHANNEL_ERROR";

        case NPP_STRIDE_ERROR:
            return "NPP_STRIDE_ERROR";

        case NPP_ANCHOR_ERROR:
            return "NPP_ANCHOR_ERROR";

        case NPP_MASK_SIZE_ERROR:
            return "NPP_MASK_SIZE_ERROR";

        case NPP_MOMENT_00_ZERO_ERROR:
            return "NPP_MOMENT_00_ZERO_ERROR";

        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
            return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

        case NPP_THRESHOLD_ERROR:
            return "NPP_THRESHOLD_ERROR";

        case NPP_CONTEXT_MATCH_ERROR:
            return "NPP_CONTEXT_MATCH_ERROR";

        case NPP_FFT_FLAG_ERROR:
            return "NPP_FFT_FLAG_ERROR";

        case NPP_FFT_ORDER_ERROR:
            return "NPP_FFT_ORDER_ERROR";

        case NPP_SCALE_RANGE_ERROR:
            return "NPP_SCALE_RANGE_ERROR";

        case NPP_DATA_TYPE_ERROR:
            return "NPP_DATA_TYPE_ERROR";

        case NPP_OUT_OFF_RANGE_ERROR:
            return "NPP_OUT_OFF_RANGE_ERROR";

        case NPP_DIVIDE_BY_ZERO_ERROR:
            return "NPP_DIVIDE_BY_ZERO_ERROR";

        case NPP_RANGE_ERROR:
            return "NPP_RANGE_ERROR";

        case NPP_NO_MEMORY_ERROR:
            return "NPP_NO_MEMORY_ERROR";

        case NPP_ERROR_RESERVED:
            return "NPP_ERROR_RESERVED";

        case NPP_NO_OPERATION_WARNING:
            return "NPP_NO_OPERATION_WARNING";

        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
        /* These are 7.0 or higher */
        case NPP_OVERFLOW_ERROR:
            return "NPP_OVERFLOW_ERROR";

        case NPP_CORRUPTED_DATA_ERROR:
            return "NPP_CORRUPTED_DATA_ERROR";
#endif
    }

    return "<unknown>";
}

/// Exception base class.
///     This exception base class will be used for everything C++ throught
/// the NPP project.
///     The exception contains a string message, as well as data fields for a string
/// containing the name of the file as well as the line number where the exception was thrown.
///     The easiest way of throwing exceptions and providing filename and line number is
/// to use one of the ASSERT macros defined for that purpose.
class Exception
{
 public:
  /// Constructor.
  /// \param rMessage A message with information as to why the exception was thrown.
  /// \param rFileName The name of the file where the exception was thrown.
  /// \param nLineNumber Line number in the file where the exception was thrown.
  explicit
    Exception(const std::string &rMessage = "", const std::string &rFileName = "", unsigned int nLineNumber = 0)
    : sMessage_(rMessage), sFileName_(rFileName), nLineNumber_(nLineNumber)
  { };

 Exception(const Exception &rException)
   : sMessage_(rException.sMessage_), sFileName_(rException.sFileName_), nLineNumber_(rException.nLineNumber_)
    { };

  virtual
    ~Exception()
    { };

  /// Get the exception's message.
  const
    std::string &
    message()
    const
  {
    return sMessage_;
  }

  /// Get the exception's file info.
  const
    std::string &
    fileName()
    const
  {
    return sFileName_;
  }

  /// Get the exceptions's line info.
  unsigned int
    lineNumber()
    const
  {
    return nLineNumber_;
  }


  /// Create a clone of this exception.
  ///      This creates a new Exception object on the heap. It is
  /// the responsibility of the user of this function to free this memory
  /// (delete x).
  virtual
    Exception *
    clone()
    const
  {
    return new Exception(*this);
  }

  /// Create a single string with all the exceptions information.
  ///     The virtual toString() method is used by the operator<<()
  /// so that all exceptions derived from this base-class can print
  /// their full information correctly even if a reference to their
  /// exact type is not had at the time of printing (i.e. the basic
  /// operator<<() is used).
  virtual
    std::string
    toString()
    const
  {
    std::ostringstream oOutputString;
    oOutputString << fileName() << ":" << lineNumber() << ": " << message();
    return oOutputString.str();
  }

 private:
  std::string sMessage_;      ///< Message regarding the cause of the exception.
  std::string sFileName_;     ///< Name of the file where the exception was thrown.
  unsigned int nLineNumber_;  ///< Line number in the file where the exception was thrown
};

/// Output stream inserter for Exception.
/// \param rOutputStream The stream the exception information is written to.
/// \param rException The exception that's being written.
/// \return Reference to the output stream being used.
inline std::ostream &
operator << (std::ostream &rOutputStream, const Exception &rException)
{
  rOutputStream << rException.toString();
  return rOutputStream;
}

/// Basic assert macro.
///     This macro should be used to enforce any kind of pre or post conditions.
/// Unlike the C-runtime assert macro, this macro does not abort execution, but throws
/// a C++ exception. The exception is automatically filled with information about the failing
/// condition, the filename and line number where the exception was thrown.
/// \note The macro is written in such a way that omitting a semicolon after its usage
///     causes a compiler error. The correct way to invoke this macro is:
/// NPP_ASSERT(n < MAX);
#define NPP_ASSERT(C) do {if (!(C)) throw Exception(#C " assertion faild!", __FILE__, __LINE__);} while(false)

/// Macro for checking error return code for NPP calls.
#define NPP_CHECK_NPP(S, str) do {NppStatus eStatusNPP;                 \
    eStatusNPP = S;                                                     \
    if (eStatusNPP != NPP_SUCCESS) std::cout << "NPP_CHECK_NPP - eStatusNPP = " << _cudaGetErrorEnum(eStatusNPP) << "("<< eStatusNPP << ") at " << __LINE__ << ": " << str << std::endl; \
    NPP_ASSERT(eStatusNPP == NPP_SUCCESS);} while (false)

namespace ICE
{
    enum StatusJPEG
    {
        NOT_JPEG_STATUS,
        UNSUPPORTED_JPEG_STATUS,
        // INVALID_HORIZONTAL_SAMPLING_FACTOR_JPEG_STATUS,
        // INVALID_VERTICAL_SAMPLING_FACTOR_JPEG_STATUS,
        // COMPONENT_INDEX_OUT_OF_RANGE_JPEG_STATUS,
        // DC_HUFFMAN_TABLE_INDEX_OUT_OF_RANGE_JPEG_STATUS,
        // HUFFMAN_TABLE_NOT_FOUND_JPEG_STATUS,
        // CANNOT_CONVERT_MARKER_TO_ENCODING_JPEG_STATUS,
        // APPLICATION_DATA_INDEX_OUT_OF_RANGE_JPEG_STATUS,
        // SCAN_INDEX_OUT_OF_RANGE_JPEG_STATUS,
        // ARITHMETIC_CODED_JPEGE_STATUS,
        // INVALID_HORIZONTAL_OFFSET_JPEG_STATUS,
        // INVALID_VERTICAL_OFFSET_JPEG_STATUS,
        // INTERNAL_UNSUPPORTED_JPEG_STATUS,
        INTERNAL_SERVER_ERROR_JPEG_STATUS
    };
    
    namespace StringsJPEG
    {
        const std::string sNotJPEG              = "Not JPEG format";
        const std::string sUnsupportedJPEG      = "Unsupported JPEG";
        const std::string sInternalServerError  = "500 Internal Server Error";
    }
    
    class ExceptionJPEG: public std::exception
    {
    public:
        
        ExceptionJPEG(StatusJPEG eStatus, const std::string & rMessage = "") :
            eStatus_(eStatus),
            sMessage_(rMessage) {;}
        
        inline virtual ~ExceptionJPEG() = default;
        
        virtual const char* what() const throw() {
            switch (eStatus_)
            {
            case NOT_JPEG_STATUS:
                return StringsJPEG::sNotJPEG.c_str();
            case UNSUPPORTED_JPEG_STATUS:
                return StringsJPEG::sUnsupportedJPEG.c_str();
            default:
                return StringsJPEG::sInternalServerError.c_str();
            }
        }
        
        StatusJPEG status() const {
            return eStatus_;
        }
        
        std::string message() const {
            return sMessage_;
        }
        
    private:
        StatusJPEG  eStatus_;
        std::string sMessage_;
    };
} //ICE namespace

#endif // DEBUG_H
