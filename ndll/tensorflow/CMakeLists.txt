if (BUILD_TENSORFLOW)

  file(GLOB tmp *.cc)
  set(NDLL_TF_SRCS ${NDLL_TF_SRCS} ${tmp} PARENT_SCOPE)

  #g++ -std=c++11 -shared ndll/tensorflow/ndllop.cc -o ndllop.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
endif()
