#!/bin/bash

mkdir -p build
cd build
rm -rf *

DEMO_NAME=trt_fp32_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

work_path=$(dirname $(readlink -f $0))
LIB_DIR=${work_path}/../../lib/devel
# LIB_DIR=/mydev/code/Paddle/build/paddle_inference_install_dir/
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4/

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
