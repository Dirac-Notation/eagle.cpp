#!/bin/bash

version=$1

wget https://github.com/ggerganov/llama.cpp/archive/refs/tags/${version}.zip
unzip ${version}.zip
rm ${version}.zip
cd llama.cpp-${version}
make -j 32