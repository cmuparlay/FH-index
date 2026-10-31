cmake_minimum_required(VERSION 3.16)
project(pgm_statistics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(PGM_INDEX_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/PGM-index"
    CACHE PATH "Path to the PGM-index checkout")

if(NOT EXISTS "${PGM_INDEX_DIR}/include/pgm/pgm_index.hpp")
    message(FATAL_ERROR
        "PGM-index headers not found at ${PGM_INDEX_DIR}. "
        "Initialize third_party/PGM-index first.")
endif()

add_executable(pgm_benchmark pgm_benchmark.cpp)
target_include_directories(pgm_benchmark SYSTEM PRIVATE "${PGM_INDEX_DIR}/include")
target_compile_options(pgm_benchmark PRIVATE
    $<$<CONFIG:Release>:-O3;-march=native>
    $<$<CONFIG:Debug>:-O0;-g>)
target_compile_definitions(pgm_benchmark PRIVATE
    $<$<CONFIG:Release>:NDEBUG>)

message(STATUS "PGM_INDEX_DIR: ${PGM_INDEX_DIR}")
