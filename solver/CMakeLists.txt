cmake_minimum_required(VERSION 3.14)
project(KDensestSolver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Solver build id: short git sha when available, otherwise "unknown". Wired into
# main.cpp via the SOLVER_BUILD_ID macro and surfaced in every emitted JSON.
execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SOLVER_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SOLVER_GIT_RC
    ERROR_QUIET
)
if(NOT SOLVER_GIT_RC EQUAL 0 OR SOLVER_GIT_SHA STREQUAL "")
    set(SOLVER_GIT_SHA "unknown")
endif()
string(TIMESTAMP SOLVER_BUILD_TIME UTC)
set(SOLVER_BUILD_ID "${SOLVER_GIT_SHA}_${SOLVER_BUILD_TIME}")

# ==============================================================================
# 1. External Dependencies
# ==============================================================================
find_package(CURL REQUIRED)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()
find_package(Boost REQUIRED)

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
    src/subgraph_quality.cpp
    src/average_degree_solver.cpp
    src/bfs_solver.cpp
    src/bfs_expand.cpp
    src/grow_to_k.cpp
)

set_target_properties(solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

# ==============================================================================
# 3. Include Directories & Linking
# ==============================================================================
target_include_directories(solver PRIVATE src ${GUROBI_INCLUDE_DIRS} ${JSON_INCLUDE_DIR})
target_compile_definitions(solver PRIVATE SOLVER_BUILD_ID="${SOLVER_BUILD_ID}")

target_link_libraries(solver
    PRIVATE
    CURL::libcurl
    ${GUROBI_CXX_LIB}
    ${GUROBI_C_LIB}
    Boost::boost
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(solver PRIVATE -O3 -march=native -Wall -Wextra)
endif()
