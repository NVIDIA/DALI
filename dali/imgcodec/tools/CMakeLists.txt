if (BUILD_TEST)
  add_executable(dali_imgcodec_imagemagick_test "imagemagick_test.cc")
  target_link_libraries(dali_imgcodec_imagemagick_test PRIVATE dali_imgcodec dali)
  set_target_properties(dali_imgcodec_imagemagick_test PROPERTIES 
    OUTPUT_NAME "dali_imgcodec_imagemagick_test.bin")
  set_target_properties(dali_imgcodec_imagemagick_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dali/tools/imgcodec")
endif()
