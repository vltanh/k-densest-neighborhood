rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..

mv build/solver bin/solver
chmod +x bin/solver