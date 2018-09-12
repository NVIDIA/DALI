#include "tiff.h"

namespace dali {

namespace {

cv::Mat DecodeTiff(const unsigned char *tiff, int size, DALIImageType image_type) {
    std::vector<char> buf(tiff, tiff + size);
    return cv::imdecode(buf, image_type == DALI_GRAY ? 0 : 1);
}


} // namespace

bool CheckIsTiff(const unsigned char *tiff) {
    assert(tiff);

    std::vector<int> header_intel = {77, 77, 0, 42};
    std::vector<int> header_motorola = {73, 73, 42, 0};

    auto check_header = [&](const std::vector<int> &header) -> bool {
        for (unsigned int i = 0; i < header.size(); i++) {
            if (tiff[i] != header[i]) {
                return false;
            }
        }
        return true;
    };

    return check_header(header_intel) || check_header(header_motorola);
}


DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w) {
    assert(CheckIsTiff(tiff));
    auto tiff_mat = DecodeTiff(tiff, size, DALI_RGB);
    if (tiff_mat.empty()) {
        return DALIError;
    }
    *h = tiff_mat.rows;
    *w = tiff_mat.cols;
    return DALISuccess;
}


DALIError_t DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type, Tensor<CPUBackend> *output) {
    assert(CheckIsTiff(tiff));
    auto tiff_mat = DecodeTiff(tiff, size, image_type);
    auto height = tiff_mat.rows;
    auto width = tiff_mat.cols;
    auto channels = (image_type == DALI_GRAY) ? 1 : 3;

    // if RGB needed, permute from BGR
    if (image_type == DALI_RGB) {
        cv::cvtColor(tiff_mat, tiff_mat, cv::COLOR_BGR2RGB);
    }

    // resize the output tensor
    output->Resize({height, width, channels});
    // force allocation
    output->mutable_data<uint8>();

    std::memcpy(output->raw_mutable_data(), tiff_mat.ptr(), static_cast<size_t>(height) * width * channels);


    return DALISuccess;
}

} // namespace dali