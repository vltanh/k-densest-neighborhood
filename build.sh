cmake -S . -B build
cmake --build build -j$(nproc)
cp build/solver bin/solver
chmod +x bin/solver