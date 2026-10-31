cmake_minimum_required(VERSION 3.16)

project(fh_index_ro LANGUAGES CXX)

set(FH_INDEX_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(FH_INDEX_STANDALONE ON)
endif()
# Apply our default only to a standalone build. An add_subdirectory consumer
# owns the parent project's build configuration.
if(FH_INDEX_STANDALONE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
option(FH_INDEX_BUILD_BENCHMARKS "Build benchmark programs" ${FH_INDEX_STANDALONE})
option(FH_INDEX_BUILD_TESTS "Build tests" ${FH_INDEX_STANDALONE})
option(FH_INDEX_BUILD_EXAMPLES "Build usage examples" ${FH_INDEX_STANDALONE})
option(FH_INDEX_BUILD_UTILITIES "Build dataset utilities" ${FH_INDEX_STANDALONE})
option(FH_INDEX_BUILD_EXPERIMENTS "Build paper analysis experiments" OFF)
option(FH_INDEX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(FH_INDEX_CPU native CACHE STRING "CPU target")
set_property(CACHE FH_INDEX_CPU PROPERTY STRINGS baseline native avx2 avx512)

set(FH_INDEX_ALLOCATOR system CACHE STRING "Benchmark allocator")
set_property(CACHE FH_INDEX_ALLOCATOR PROPERTY STRINGS system jemalloc)

set(FH_INDEX_RECLAMATION normal CACHE STRING "Benchmark reclamation")
set_property(CACHE FH_INDEX_RECLAMATION PROPERTY STRINGS normal disabled guards_only record_only)

# native enables the instruction sets supported by the build machine, including
# AVX2/AVX-512 and BMI/LZCNT where available. Explicit modes support other targets.
set(FH_INDEX_CPU_FLAGS "")
if(FH_INDEX_CPU STREQUAL "baseline")
    # Keep the toolchain's default CPU target.
elseif(FH_INDEX_CPU STREQUAL "native")
    set(FH_INDEX_CPU_FLAGS -march=native)
elseif(FH_INDEX_CPU STREQUAL "avx2")
    set(FH_INDEX_CPU_FLAGS -mavx2)
elseif(FH_INDEX_CPU STREQUAL "avx512")
    set(FH_INDEX_CPU_FLAGS -mavx2 -mavx512f)
else()
    message(FATAL_ERROR "Invalid FH_INDEX_CPU: ${FH_INDEX_CPU}")
endif()

if(FH_INDEX_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "FH_INDEX_ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    include(CMakePushCheckState)
    include(CheckCXXSourceCompiles)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined")
    check_cxx_source_compiles("int main() { return 0; }" FH_INDEX_SUPPORTS_SANITIZERS)
    cmake_pop_check_state()
    if(NOT FH_INDEX_SUPPORTS_SANITIZERS)
        message(FATAL_ERROR
            "The selected compiler cannot compile and link ASan/UBSan. "
            "Install its sanitizer runtimes or select another compiler.")
    endif()
endif()

set(FH_INDEX_RECLAMATION_DEFINITION "")
if(FH_INDEX_RECLAMATION STREQUAL "normal")
    # Keep normal epoch guards, retirement and reclamation.
elseif(FH_INDEX_RECLAMATION STREQUAL "disabled")
    set(FH_INDEX_RECLAMATION_DEFINITION FH_INDEX_DISABLE_RECLAMATION=1)
elseif(FH_INDEX_RECLAMATION STREQUAL "guards_only")
    set(FH_INDEX_RECLAMATION_DEFINITION FH_INDEX_DISABLE_RETIREMENT=1)
elseif(FH_INDEX_RECLAMATION STREQUAL "record_only")
    set(FH_INDEX_RECLAMATION_DEFINITION FH_INDEX_RECORD_RETIREMENTS_ONLY=1)
else()
    message(FATAL_ERROR "Invalid FH_INDEX_RECLAMATION: ${FH_INDEX_RECLAMATION}")
endif()
if(FH_INDEX_BUILD_BENCHMARKS AND NOT FH_INDEX_RECLAMATION STREQUAL "normal")
    message(WARNING
        "Benchmark reclamation mode ${FH_INDEX_RECLAMATION} retains replaced objects; "
        "tests keep normal reclamation")
endif()

if(FH_INDEX_ALLOCATOR STREQUAL "system")
    # Use the default allocator without linking an extra library.
elseif(FH_INDEX_ALLOCATOR STREQUAL "jemalloc")
    if(FH_INDEX_BUILD_BENCHMARKS)
        find_library(FH_INDEX_JEMALLOC_LIBRARY NAMES jemalloc)
        if(NOT FH_INDEX_JEMALLOC_LIBRARY)
            message(FATAL_ERROR "jemalloc was selected but its library was not found")
        endif()
    endif()
else()
    message(FATAL_ERROR "Invalid FH_INDEX_ALLOCATOR: ${FH_INDEX_ALLOCATOR}")
endif()

find_package(Threads REQUIRED)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/parlaylib/include/parlay/parallel.h")
    message(FATAL_ERROR
        "ParlayLib is missing. Run git submodule update --init --recursive "
        "from the FH-index checkout before configuring.")
endif()

# Only library requirements propagate to consumers of the header-only target.
add_library(fh_index INTERFACE)
add_library(fh_index::fh_index ALIAS fh_index)
target_compile_features(fh_index INTERFACE cxx_std_20)
target_include_directories(fh_index INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(fh_index SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parlaylib/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parallel-minimum-interval-covering/include")
target_link_libraries(fh_index INTERFACE Threads::Threads)

function(fh_index_add_program NAME SOURCE)
    add_executable(${NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
    set_target_properties(${NAME} PROPERTIES
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    target_link_libraries(${NAME} PRIVATE fh_index::fh_index)
    target_include_directories(${NAME} SYSTEM PRIVATE
        "${PROJECT_SOURCE_DIR}/third_party/timer_tree/include")
    target_compile_options(${NAME} PRIVATE ${FH_INDEX_CPU_FLAGS})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${NAME} PRIVATE -Wall -Wextra)
    endif()
    if(FH_INDEX_ENABLE_SANITIZERS)
        target_compile_options(${NAME} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${NAME} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

if(FH_INDEX_BUILD_TESTS)
    enable_testing()
endif()

if(FH_INDEX_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
if(FH_INDEX_BUILD_TESTS)
    add_subdirectory(test)
endif()
if(FH_INDEX_BUILD_UTILITIES)
    add_subdirectory(util)
endif()
if(FH_INDEX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(FH_INDEX_BUILD_EXPERIMENTS)
    add_subdirectory(experiments/partition_count)
    add_subdirectory(experiments/pgm_statistics)
endif()

message(STATUS "FH-index: C++20; CPU=${FH_INDEX_CPU}; benchmark allocator=${FH_INDEX_ALLOCATOR}")
message(STATUS "Benchmark reclamation=${FH_INDEX_RECLAMATION}; sanitizers=${FH_INDEX_ENABLE_SANITIZERS}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build configurations: ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")
endif()
