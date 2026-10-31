cmake_minimum_required(VERSION 3.16)
project(partition_count LANGUAGES CXX)

set(FH_INDEX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Path to the FH-index source tree")
set(PGM_INDEX_DIR "${FH_INDEX_SOURCE_DIR}/third_party/PGM-index"
    CACHE PATH "Path to the PGM-index source tree")
if(NOT EXISTS "${FH_INDEX_SOURCE_DIR}/include/fh_index/fh_index_rw.h")
    message(FATAL_ERROR "Set FH_INDEX_SOURCE_DIR to this FH-index checkout")
endif()
if(NOT EXISTS "${PGM_INDEX_DIR}/include/pgm/piecewise_linear_model.hpp")
    message(FATAL_ERROR "PGM-index is missing; run git submodule update --init --recursive")
endif()
if(NOT TARGET fh_index::fh_index)
    add_subdirectory("${FH_INDEX_SOURCE_DIR}" fh_index_build)
endif()

add_executable(bench_partition_count bench_partition_count.cpp)
target_link_libraries(bench_partition_count PRIVATE fh_index::fh_index)
target_include_directories(bench_partition_count SYSTEM PRIVATE "${PGM_INDEX_DIR}/include")
target_compile_options(bench_partition_count PRIVATE ${FH_INDEX_CPU_FLAGS})
set_target_properties(bench_partition_count PROPERTIES CXX_EXTENSIONS OFF)
