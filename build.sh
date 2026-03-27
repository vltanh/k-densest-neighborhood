#!/bin/bash

rm -rf build bin

cmake -S . -B build
cmake --build build -j$(nproc)

mkdir -p bin
cp build/solver bin/solver
chmod +x bin/solver