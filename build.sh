#!/bin/bash

# Set default values
EXECUTABLE="cuda_benchmark"
DEBUG=0
ARCH="sm_80"  # Default architecture, update based on your GPU
VERBOSE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--debug)
      DEBUG=1
      shift
      ;;
    -a|--arch)
      ARCH=$2
      shift 2
      ;;
    -e|--executable)
      EXECUTABLE=$2
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Determine optimization level based on debug flag
if [ $DEBUG -eq 1 ]; then
  OPT_FLAG="-O0 -g"
  NVCC_DEBUG="-G"
  echo "Building in debug mode..."
else
  OPT_FLAG="-O3"
  NVCC_DEBUG=""
  echo "Building in release mode..."
fi

# Common CUDA compiler flags
NVCC_FLAGS="$OPT_FLAG $NVCC_DEBUG -arch=$ARCH -std=c++20 --expt-relaxed-constexpr"

# Source and header files
SRC_FILES=$(find . -name "*.cu" | tr '\n' ' ')
INCLUDE_DIRS="-I."

# Print verbose information if requested
if [ $VERBOSE -eq 1 ]; then
  echo "NVCC_FLAGS: $NVCC_FLAGS"
  echo "SRC_FILES: $SRC_FILES"
  echo "INCLUDE_DIRS: $INCLUDE_DIRS"
fi

# Compile
echo "Compiling with nvcc $NVCC_FLAGS $INCLUDE_DIRS $SRC_FILES -o $EXECUTABLE"
nvcc $NVCC_FLAGS $INCLUDE_DIRS $SRC_FILES -o $EXECUTABLE

# Check if compilation was successful
if [ $? -eq 0 ]; then
  echo "Compilation successful. Executable: $EXECUTABLE"
else
  echo "Compilation failed."
  exit 1
fi

chmod +x $EXECUTABLE
echo "Done"
