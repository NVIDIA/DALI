#include <iostream>

#include "dali/imgcodec/image_source.h"

int main() {
    // Just to see if it links with imgcodec correctly
    auto src = dali::imgcodec::ImageSource::FromFilename("blah blah");

    std::cout << "Hello world!" << std::endl;
}

