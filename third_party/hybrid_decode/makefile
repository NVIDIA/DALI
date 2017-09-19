PROJECT=hybrid_decoder
INC_DIR=include
SRC_DIR=src
TEST_DIR=test
LIB_DIR=lib

# Set for your system
CUDA_DIR=/usr/local/cuda
GTEST_DIR=/home/tgale/googletest/googletest

# Outputs
LIB=$(LIB_DIR)/lib$(PROJECT).a
TEST_BIN=run_tests

NVCC=nvcc
CFLAGS=--std=c++11 --compiler-options=-fPIC # --compiler-options=-ggdb # -DENABLE_TIMERANGES
INCLUDE=-I$(INC_DIR)
LDFLAGS=-lnppicom_static -lnppicc_static -lnppc_static -lculibos -lnvToolsExt

TEST_INCLUDE=$(INCLUDE) -I$(GTEST_DIR)/include -I$(TEST_DIR)
TEST_LDFLAGS=$(LDFLAGS) -L$(GTEST_DIR) -L$(LIB_DIR) -lgtest -lpthread -lhybrid_decoder

TEST_FILES=$(shell find $(TEST_DIR) -name "test_*.cpp")
CXX_FILES=$(shell find $(SRC_DIR) -name "*.cpp")
CXX_OBJ=$(CXX_FILES:%.cpp=%.o)
CU_FILES=$(shell find $(SRC_DIR) -name "*.cu")
CU_OBJ=$(CU_FILES:%.cu=%.o)

.PHONY: all test lib clean

all: lib test

%.o:%.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

%.o:%.cpp
	$(NVCC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(LIB): $(CXX_OBJ) $(CU_OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(CXX_OBJ) $(CU_OBJ)

lib: $(LIB)

test: $(LIB)
	$(NVCC) $(CFLAGS) -o $(TEST_BIN) $(TEST_FILES) $(TEST_INCLUDE) $(TEST_LDFLAGS)

clean:
	rm -rf $(TEST_BIN) $(CXX_OBJ) $(CU_OBJ) $(LIB)
