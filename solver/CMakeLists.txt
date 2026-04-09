cmake_minimum_required(VERSION 3.14)
project(KDensestSolver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# 1. External Dependencies
# ==============================================================================
find_package(CURL REQUIRED)

# B. nlohmann/json (Fetched directly as a single header to bypass CMake 4.0+ errors)
set(JSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/external/json/include)
set(JSON_HEADER_FILE ${JSON_INCLUDE_DIR}/nlohmann/json.hpp)
if(NOT EXISTS ${JSON_HEADER_FILE})
    message(STATUS "Downloading nlohmann/json.hpp...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        ${JSON_HEADER_FILE}
        STATUS download_status
    )
endif()

# C. Gurobi
if(NOT DEFINED ENV{GUROBI_HOME})
    message(FATAL_ERROR "GUROBI_HOME environment variable is not set.")
endif()
set(GUROBI_HOME $ENV{GUROBI_HOME})

set(GUROBI_INCLUDE_DIRS ${GUROBI_HOME}/include)
file(GLOB GUROBI_CXX_LIB "${GUROBI_HOME}/lib/libgurobi_c++.*")
file(GLOB GUROBI_C_LIB "${GUROBI_HOME}/lib/libgurobi[0-9]*.so" "${GUROBI_HOME}/lib/libgurobi[0-9]*.dylib" "${GUROBI_HOME}/lib/gurobi[0-9]*.lib")

if(NOT GUROBI_CXX_LIB OR NOT GUROBI_C_LIB)
    message(FATAL_ERROR "Could not locate Gurobi libraries in ${GUROBI_HOME}/lib")
endif()

# ==============================================================================
# 2. Target Definition (Pointing strictly to src/)
# ==============================================================================
add_executable(solver
    src/main.cpp
    src/solver.cpp
    src/simulation_oracle.cpp
    src/openalex_oracle.cpp
)

# ==============================================================================
# 3. Include Directories & Linking
# ==============================================================================
target_include_directories(solver PRIVATE src ${GUROBI_INCLUDE_DIRS} ${JSON_INCLUDE_DIR})

target_link_libraries(solver
    PRIVATE
    CURL::libcurl
    ${GUROBI_CXX_LIB}
    ${GUROBI_C_LIB}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(solver PRIVATE -O3 -march=native -Wall -Wextra)
endif()