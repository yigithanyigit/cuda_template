CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Default compute capability
ARCH ?= sm_70

# Detect compute capability automatically if possible
COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
ifneq ($(COMPUTE_CAP),)
  ARCH := sm_$(shell echo $(COMPUTE_CAP) | tr -d '.')
endif

# Debug mode flag
DEBUG ?= 0
ifeq ($(DEBUG), 1)
  NVCC_FLAGS := -O0 -g -G
else
  NVCC_FLAGS := -O3
endif

# Common compilation flags
NVCC_FLAGS += -arch=$(ARCH) -std=c++14 --expt-relaxed-constexpr

# Include directories
INCLUDES := -I.

# Find all .cu files
SOURCES := $(wildcard *.cu)
EXECUTABLE := cuda_benchmark

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)
