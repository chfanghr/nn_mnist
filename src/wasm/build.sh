#!/usr/bin/env bash

set -e

if [ -z ${PROJECT_ROOT+x} ]; then
  echo "PROJECT_ROOT is not set"
  exit 1
fi

em++ -s WASM=1 --embed-file "${PROJECT_ROOT}"/data/network.dat@network.dat \
  "${PROJECT_ROOT}"/src/wasm/main.cc "${PROJECT_ROOT}"/src/nn_mnist/network.cc \
  -I "${PROJECT_ROOT}"/include -o wasm_nn_mnist.js --std=c++17

