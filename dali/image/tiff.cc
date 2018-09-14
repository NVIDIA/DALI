#include "tiff.h"

namespace dali {

namespace {

const std::vector<int> header_intel = {77, 77, 0, 42};
const std::vector<int> header_motorola = {73, 73, 42, 0};
const int COUNT_SIZE = 2;
const int ENTRY_SIZE = 12;
const int WIDTH_TAG = 256;
const int HEIGHT_TAG = 257;


cv::Mat DecodeTiff(const unsigned char *tiff, int size, DALIImageType image_type) {
    std::vector<char> buf(tiff, tiff + size);
    return cv::imdecode(buf, image_type == DALI_GRAY ? 0 : 1);
}


bool check_header(const unsigned char *tiff, const std::vector<int> &header) {
    for (unsigned int i = 0; i < header.size(); i++) {
        if (tiff[i] != header[i]) {
            return false;
        }
    }
    return true;
}


bool is_little_endian(const unsigned char *tiff) {
    return check_header(tiff, header_intel);
}

} // namespace

bool CheckIsTiff(const unsigned char *tiff) {
    assert(tiff);
    return check_header(tiff, header_intel) || check_header(tiff, header_motorola);
}


DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w) {
    assert(CheckIsTiff(tiff));

    tiff_buffer buffer(std::string(reinterpret_cast<const char *>(tiff), static_cast<size_t>(size)),
                       is_little_endian(tiff));

    auto ifd_offset = buffer.read<unsigned int>(4);
    auto entry_count = buffer.read<unsigned short>(ifd_offset);
    bool width_read = false, height_read = false;

    for (int entry_idx = 0;
         entry_idx < entry_count && !(width_read && height_read);
         entry_idx++) {
        auto entry_offset = ifd_offset + COUNT_SIZE + entry_idx * ENTRY_SIZE;
        auto tag_id = buffer.read<unsigned short>(entry_offset);
        if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG) {
            auto value_type = buffer.read<unsigned short>(entry_offset + 2);
            auto value_count = buffer.read<unsigned int>(entry_offset + 4);
            assert(value_count == 1);

            unsigned long value;
            if (value_type == 3) {
                value = buffer.read<unsigned short>(entry_offset + 8);
            } else if (value_type == 4) {
                value = buffer.read<unsigned long>(entry_offset + 8);
            } else {
                return DALIError;
            }

            if (tag_id == WIDTH_TAG) {
                *w = static_cast<int>(value);
                width_read = true;
            } else {
                *h = static_cast<int>(value);
                height_read = true;
            }
        }
    }
    if (!(width_read && height_read)) {
        return DALIError;
    }

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