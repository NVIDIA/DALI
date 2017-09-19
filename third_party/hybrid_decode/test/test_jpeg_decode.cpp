#include "device_buffer.h"
#include "hybrid_decoder.h"
#include "input_stream_jpeg.h"
#include "jpeg_parser.h"
#include "test_main.h"

#include <gtest/gtest.h>
#include <npp.h>

#include <array>
#include <chrono>
#include <iostream>

using namespace std::chrono;
using std::array;
using std::cout;


class JpegDecodeTest : public TestMain {
public:

protected:
};

TEST_F(JpegDecodeTest, TestDecodeAll) {
    // Memory for dev-side dct coefficients & quant tables
    array<DeviceBuffer, 3> dDct;
    DeviceBuffer dQuant;
    array<DeviceBuffer, 3> dst;

    // scratch space to load quant tables for batched copy
    void *hTmp = ::operator new(64*2*3);
    
    // Buffer for dev-side rgb image
    DeviceBuffer rgb;

    // Allocate dct state
    DctState dState;    
    JpegParserState jpState;
    HuffmanDecoderState hState;
    vector<HostBlocksDCT> hDct(3);
    ParsedJpeg jpeg;
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
    for (int img = 0; img < NUM_IMAGE; ++img) {
        // cout << img << " " << imageFiles_[img] << endl;
        // Parse the raw jpeg string
        parseRawJpegHost(jpegData_[img], jpegLengths_[img], &jpState, &jpeg);

        // Run the huffman decode
        huffmanDecodeHost(jpeg, &hState, &hDct);

        // Batched all quant tables together
        vector<size_t> offset(jpeg.components, 0);
        size_t size = 0;
        {
            TimeRange _tr("quant_batching");
            for (int i = 0; i < jpeg.components; ++i) {
                if (jpeg.quantTables[i].nPrecision == QuantizationTable::PRECISION_8_BIT) {
                    memcpy((Npp8u*)hTmp + size, jpeg.quantTables[i].aTable.lowp, 64);
                    offset[i] = size;
                    size += 64;
                } else {
                    memcpy((Npp8u*)hTmp + size, jpeg.quantTables[i].aTable.highp, 64*2);
                    offset[i] = size;
                    size += 64 * 2;
                }
            }
        }

        // Copy the tables over
        {
            TimeRange _tr1("copy_quant");
            dQuant.resize(size);
            CHECK_CUDA(cudaMemcpyAsync(dQuant, hTmp, size, cudaMemcpyHostToDevice, nppGetStream()));
        }
        
        for (int i = 0; i < jpeg.components; ++i) {
            {
                TimeRange _tr2("copy_dct");
                // Copy the dct coefficients to device
                dDct[i].resize(jpeg.dctSize[i]);
                CHECK_CUDA(cudaMemcpyAsync(dDct[i], hDct[i].blockData(),
                                jpeg.dctSize[i], cudaMemcpyHostToDevice,
                                nppGetStream()));
            }
            
            // Perform the iDCT
            dst[i].resize(jpeg.yCbCrDims[i].width * jpeg.yCbCrDims[i].height);
            int dstStep = jpeg.yCbCrDims[i].width;
            dctQuantInv((Npp16s*)dDct[i].data(), jpeg.dctLineStep[i], dst[i], dstStep,
                    jpeg.yCbCrDims[i], jpeg.quantTables[i].nPrecision,
                    (Npp8u*)dQuant.data() + offset[i], &dState);
        }

        // DEBUG
        // NOTE: CPP doesn't call the casting operator for DeviceBuffer when we the pointers
        // to the DeviceBuffers from a std::array. Here we explicitly make a c-style
        // array with the same pointers to avoid this casting issue
        Npp8u *c_dst[3] = {dst[0].data(), dst[1].data(), dst[2].data()};
    
        // DEBUG
        size_t heights[3] = {(size_t)jpeg.yCbCrDims[0].height,
                             (size_t)jpeg.yCbCrDims[1].height,
                             (size_t)jpeg.yCbCrDims[2].height};
        size_t widths [3] = {(size_t)jpeg.yCbCrDims[0].width,
                             (size_t)jpeg.yCbCrDims[1].width,
                             (size_t)jpeg.yCbCrDims[2].width};
        array<Npp32s, 3> steps = {(Npp32s)jpeg.yCbCrDims[0].width,
                                  (Npp32s)jpeg.yCbCrDims[1].width,
                                  (Npp32s)jpeg.yCbCrDims[2].width};
    
        // this->dump_planar_image(c_dst, steps.data(), heights, widths);

        if (jpeg.components == 3) {
            rgb.resize(jpeg.imgDims.width * jpeg.imgDims.height * 3);
            int rgbStep = jpeg.imgDims.width * 3;

            yCbCrToRgb((const Npp8u**)c_dst, steps.data(), rgb, rgbStep, jpeg.imgDims, jpeg.sRatio);
            
            // ASSERT_TRUE(is_data_equal(rgb.data(), groundTruth_[img],
            //                 jpeg.imgDims.width * jpeg.imgDims.height * 3));
            
            // // DEBUG
            // this->dump_image(rgb, jpeg.imgDims, 3, rgbStep, std::to_string(img));
            // this->dump_rgb_image(rgb, jpeg.imgDims, rgbStep);
        } else {
            // ASSERT_TRUE(is_data_equal(dst[0].data(), groundTruth_[img],
            //                 jpeg.imgDims.width * jpeg.imgDims.height));
            // this->dump_image(dst[0].data(), jpeg.yCbCrDims[0], 1,
            //         jpeg.yCbCrDims[0].width, "test/" + imageFiles_[img]);
        }
    }

    // Get timing results
    CHECK_CUDA(cudaDeviceSynchronize());
    t2 = high_resolution_clock::now();
    cout << "Decoding " << NUM_IMAGE << " images took: " <<
        duration_cast<std::chrono::nanoseconds>(t2-t1).count()
         << " ns" << endl;
    float frames_per_ns = float(NUM_IMAGE) /
        float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}

TEST_F(JpegDecodeTest, TestDecodeAllBatched) {
    // Memory for dev-side dct coefficients & quant tables
    array<DeviceBuffer, 3> dDct;
    DeviceBuffer dQuant;
    array<DeviceBuffer, 3> dst;

    // scratch space to load quant tables for batched copy
    void *hTmp = ::operator new(64*2*3);
    
    // Buffer for dev-side rgb image
    DeviceBuffer rgb;

    // Allocate dct state
    DctState dState;    
    JpegParserState jpState;
    HuffmanDecoderState hState;
    vector<HostBlocksDCT> hDct(3);
    ParsedJpeg jpeg;
    
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
    for (int img = 0; img < NUM_IMAGE; ++img) {
        // Parse the raw jpeg string
        parseRawJpegHost(jpegData_[img], jpegLengths_[img], &jpState, &jpeg);

        // Run the huffman decode
        huffmanDecodeHost(jpeg, &hState, &hDct);

        // Batched all quant tables together
        vector<size_t> offset(jpeg.components, 0);
        size_t size = 0;
        {
            TimeRange _tr("quant_batching");
            for (int i = 0; i < jpeg.components; ++i) {
                if (jpeg.quantTables[i].nPrecision == QuantizationTable::PRECISION_8_BIT) {
                    memcpy((Npp8u*)hTmp + size, jpeg.quantTables[i].aTable.lowp, 64);
                    offset[i] = size;
                    size += 64;
                } else {
                    memcpy((Npp8u*)hTmp + size, jpeg.quantTables[i].aTable.highp, 64*2);
                    offset[i] = size;
                    size += 64 * 2;
                }
            }
        }

        // Copy the tables over
        {
            TimeRange _tr1("copy_quant");
            dQuant.resize(size);
            CHECK_CUDA(cudaMemcpyAsync(dQuant, hTmp, size, cudaMemcpyHostToDevice, nppGetStream()));
        }

        // Copy the dct coefficients to device
        vector<const Npp16s*> bDctCoeff(jpeg.components, nullptr);
        vector<unsigned> bDctStep(jpeg.components, 0);
        vector<Npp8u*> bDst(jpeg.components, nullptr);
        vector<unsigned> bDstStep(jpeg.components, 0);
        vector<NppiSize> bDstSize(jpeg.components);
        vector<const void*> bQuantTable(jpeg.components, nullptr);
        QuantizationTable::QuantizationTablePrecision p = jpeg.quantTables[0].nPrecision;
        for (int i = 0; i < jpeg.components; ++i) {
            TimeRange _tr2("copy_dct");
            dDct[i].resize(jpeg.dctSize[i]);
            CHECK_CUDA(cudaMemcpyAsync(dDct[i], hDct[i].blockData(),
                            jpeg.dctSize[i], cudaMemcpyHostToDevice,
                            nppGetStream()));
            dst[i].resize(jpeg.yCbCrDims[i].width * jpeg.yCbCrDims[i].height);
            int dstStep = jpeg.yCbCrDims[i].width;
            
            // set up params for batched iDCT
            bDctCoeff[i] = (Npp16s*)dDct[i].data();
            bDctStep[i] = jpeg.dctLineStep[i];
            bDst[i] = dst[i];
            bDstStep[i] = dstStep;
            bDstSize[i] = jpeg.yCbCrDims[i];
            bQuantTable[i] = (Npp8u*)dQuant.data() + offset[i];

            // All quant tables must be same precision
            ASSERT_TRUE(p == jpeg.quantTables[i].nPrecision);
        }


        // Perform the iDCT
        BatchedDctParam batchedParam;
        batchedParam.loadToDevice(bDctCoeff, bDctStep, bDst, bDstStep, bDstSize, bQuantTable, p);
        batchedDctQuantInv(&batchedParam);

        // DEBUG
        // NOTE: CPP doesn't call the casting operator for DeviceBuffer when we the pointers
        // to the DeviceBuffers from a std::array. Here we explicitly make a c-style
        // array with the same pointers to avoid this casting issue
        Npp8u *c_dst[3] = {dst[0].data(), dst[1].data(), dst[2].data()};
    
        // DEBUG
        size_t heights[3] = {(size_t)jpeg.yCbCrDims[0].height,
                             (size_t)jpeg.yCbCrDims[1].height,
                             (size_t)jpeg.yCbCrDims[2].height};
        size_t widths [3] = {(size_t)jpeg.yCbCrDims[0].width,
                             (size_t)jpeg.yCbCrDims[1].width,
                             (size_t)jpeg.yCbCrDims[2].width};
        array<Npp32s, 3> steps = {(Npp32s)jpeg.yCbCrDims[0].width,
                                  (Npp32s)jpeg.yCbCrDims[1].width,
                                  (Npp32s)jpeg.yCbCrDims[2].width};
    
        // this->dump_planar_image(c_dst, steps.data(), heights, widths);

        if (jpeg.components == 3) {
            rgb.resize(jpeg.imgDims.width * jpeg.imgDims.height * 3);
            int rgbStep = jpeg.imgDims.width * 3;

            yCbCrToRgb((const Npp8u**)c_dst, steps.data(), rgb, rgbStep, jpeg.imgDims, jpeg.sRatio);

            // DEBUG
            // this->dump_rgb_image(rgb, jpeg.imgDims, rgbStep);
        } else {
            // DEBUG
            // cout << "Grayscale image: processing complete" << endl;
        }
    }

    // Get timing results
    CHECK_CUDA(cudaDeviceSynchronize());
    t2 = high_resolution_clock::now();
    cout << "Decoding " << NUM_IMAGE << " images took: " <<
        duration_cast<std::chrono::nanoseconds>(t2-t1).count()
         << " ns" << endl;
    float frames_per_ns = float(NUM_IMAGE) /
        float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}

TEST_F(JpegDecodeTest, TestDecodeAllBatchedAll) {
    // Scratch space to pack DCT coeffs for copy to device
    HostBuffer packDct;
    vector<int> ptrOffsets;
    int offset = 0;

    // Scratch space to pack quant tables
    HostBuffer packQuant;

    // HACK: resizing these buffer w/o copying current values into new
    // memory loses all of the data we've copied in, but allocating and
    // copying on for every component of every image is way too exspensive.
    // For this demo just allocate big enough buffers up front.
    // This will break if the number of pixels in each image is > 3 * 480 * 3 * 380
    packDct.resize(NUM_IMAGE * 3 * 480 * 3 * 480);
    packQuant.resize(NUM_IMAGE * 64 * 3);
    
    // packed dev-side dct coeffs & quant tables
    DeviceBuffer packDDct, packDQuant;
    
    // Buffer for dev-size ycbcr components
    DeviceBuffer dOut;
    
    // Buffer for dev-side rgb image
    DeviceBuffer rgb;

    // Objects to perform decode
    JpegParserState jpState;
    HuffmanDecoderState hState;
    vector<HostBlocksDCT> hDct(3);
    ParsedJpeg jpeg;
        
    // Objects for idct state
    vector<unsigned> bDctStep;
    vector<unsigned> bDstStep;
    vector<NppiSize> bDstSize;
    vector<int> oPtrOffsets;
    int outSize = 0;

    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
    for (int img = 0; img < NUM_IMAGE; ++img) {
        // Parse the raw jpeg string
        parseRawJpegHost(jpegData_[img], jpegLengths_[img], &jpState, &jpeg);

        // Run the huffman decode
        huffmanDecodeHost(jpeg, &hState, &hDct);

        TimeRange _tr("pack & process");
        // Copy dct coefficients into common buffer
        int tmp = 0;
        for (auto &val : jpeg.dctSize) tmp += val;

        // Copy quant tables into common buffer
        int qOff = 3*64*img;
        {
            TimeRange _tr1("resize host");
            packDct.resize(offset + tmp);
            packQuant.resize(qOff + 3 * 64);
        }
        
        // This example only supports rgb images
        assert(jpeg.components == 3);
        for (int i = 0; i < jpeg.components; ++i) {
            // Pack the dct coefficients
            ptrOffsets.push_back(offset);
            memcpy(packDct.data() + offset, hDct[i].blockData(), jpeg.dctSize[i]);
            offset += jpeg.dctSize[i];

            // This example only supports 8-bit quant tables
            assert(jpeg.quantTables[i].nPrecision == QuantizationTable::PRECISION_8_BIT);
            
            // Pack the quant tables
            memcpy(packQuant.data() + qOff + 64*i, jpeg.quantTables[i].aTable.lowp, 64);
            
            // Save this for later...
            oPtrOffsets.push_back(outSize);
            outSize += jpeg.yCbCrDims[i].width * jpeg.yCbCrDims[i].height;
            
            // idct params
            bDctStep.push_back(jpeg.dctLineStep[i]);
            bDstStep.push_back(jpeg.yCbCrDims[i].width);
            bDstSize.push_back(jpeg.yCbCrDims[i]);
        }
    }

    // Copy the data to device
    packDDct.resize(packDct.size());
    CHECK_CUDA(cudaMemcpyAsync(packDDct.data(), packDct.data(), packDct.size(),
                    cudaMemcpyHostToDevice, nppGetStream()));

    packDQuant.resize(packQuant.size());
    CHECK_CUDA(cudaMemcpyAsync(packDQuant.data(), packQuant.data(), packQuant.size(),
                    cudaMemcpyHostToDevice, nppGetStream()));

    // Allocate memory for the outputs
    dOut.resize(outSize);

    // Build all the batched idct params
    int totalComponents = ptrOffsets.size();
    vector<const Npp16s*> bDctCoeff(totalComponents, nullptr);
    vector<Npp8u*> bDst(totalComponents, nullptr);
    vector<const void*> bQuantTable(totalComponents, nullptr);
    QuantizationTable::QuantizationTablePrecision p = QuantizationTable::PRECISION_8_BIT;
    
    for (int i = 0; i < totalComponents; ++i) {
        bDctCoeff[i] = (Npp16s*)(packDDct.data() + ptrOffsets[i]);
        bDst[i] = dOut.data() + oPtrOffsets[i];
        bQuantTable[i] = packDQuant.data() + 64 * i;
    }

    // Perform the iDCT
    BatchedDctParam batchedParam;
    batchedParam.loadToDevice(bDctCoeff, bDctStep, bDst, bDstStep, bDstSize, bQuantTable, p);
    batchedDctQuantInv(&batchedParam);

    // DEBUG
    // for (int i = 0; i < NUM_IMAGE; ++i) {
    //     Npp8u *c_dst[3] = {bDst[i*3], bDst[i*3 + 1], bDst[i*3 + 2]};
    //     size_t heights[3] = {256, 128, 128};
    //     size_t widths [3] = {256, 128, 128};
    //     array<Npp32s, 3> steps = {256, 128, 128};
        
    //     this->dump_planar_image(c_dst, steps.data(), heights, widths);
    // }

    // Get timing results
    CHECK_CUDA(cudaDeviceSynchronize());
    t2 = high_resolution_clock::now();
    cout << "Decoding " << NUM_IMAGE << " images took: " <<
        duration_cast<std::chrono::nanoseconds>(t2-t1).count()
         << " ns" << endl;
    float frames_per_ns = float(NUM_IMAGE) /
        float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
    cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}
