#ifndef FH_INDEX_UTIL_IO_UTILITY_H
#define FH_INDEX_UTIL_IO_UTILITY_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_internal/utility/common.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/random.h"

namespace fh_index {
namespace util {
inline std::vector<uint64_t> LoadKeysFromBinary(
    const std::string& file_path,
    const uint64_t size_limit = std::numeric_limits<uint64_t>::max(),
    bool do_sort = true,
    bool check_sorted = true, bool check_duplicated = true) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    } else {
        std::cout << "Open File: " << file_path << std::endl;
    }

    // Get file size and calculate number of keys
    std::streamsize file_size = file.tellg();
    if (file_size < 0 || file_size % sizeof(uint64_t) != 0) {
        throw std::runtime_error("Invalid raw-key file size: " + file_path);
    }
    file.seekg(0, std::ios::beg);

    uint64_t max_size = file_size / sizeof(uint64_t);
    std::cout << "File has " << max_size << " keys" << std::endl;

    max_size = std::min(max_size, size_limit);
    std::cout << "Read " << max_size << " keys" << std::endl;

    std::vector<uint64_t> data(max_size);
    if (max_size != 0) {
        const auto bytes = static_cast<std::streamsize>(max_size * sizeof(uint64_t));
        if (!file.read(reinterpret_cast<char*>(data.data()), bytes)) {
            throw std::runtime_error("Failed to read raw keys: " + file_path);
        }
    }

    if (do_sort && data.size() > 1) {
        auto slice = parlay::make_slice(data.data(), data.data() + max_size);
        parlay::integer_sort_inplace(slice);
    }

    if (check_sorted && data.size() > 1) {
        parlay::parallel_for(1, max_size, [&](size_t i) {
            RUNTIME_ASSERT((data[i] >= data[i - 1]));
        });
    }

    if (check_duplicated && data.size() > 1) {
        parlay::sequence<uint64_t> tmp = parlay::tabulate(max_size, [&](size_t i) {
            return data[i];
        });
        parlay::integer_sort_inplace(tmp);
        parlay::parallel_for(1, max_size, [&](size_t i) {
            RUNTIME_ASSERT((tmp[i] > tmp[i - 1]));
        });
    }

    return data;
}

template <typename R>
void StoreKeysToBinary(const std::string& file_path, const R& data) {
    static_assert(sizeof(*data.data()) == sizeof(uint64_t),
                  "Raw-key files require 64-bit elements");
    if (data.size() > static_cast<size_t>(
            std::numeric_limits<std::streamsize>::max()) / sizeof(uint64_t)) {
        throw std::length_error("Raw-key output is too large: " + file_path);
    }
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Write raw data only (no size header)
    if (data.size() != 0) {
        file.write(reinterpret_cast<const char*>(data.data()),
                   data.size() * sizeof(uint64_t));
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Failed to write raw keys: " + file_path);
    }
}

inline std::vector<uint64_t> UniformRandomKeys(uint64_t size, uint64_t seed, bool sort) {
    parlay::random_generator gen(seed);
    std::uniform_int_distribution<uint64_t> dis;
    std::vector<uint64_t> result(size);
    parlay::parallel_for(0, size, [&](size_t i) {
        auto r = gen[i];
        result[i] = dis(r);
    });
    if (sort) {
        parlay::integer_sort_inplace(result);
    }
    return result;
    // return parlay::tabulate(size, [&](size_t i) {
    //     auto r = gen[i];
    //     return dis(r);
    // });
}

inline std::vector<uint64_t> DeduplicateSortedKeys(const std::vector<uint64_t>& data) {
    std::vector<uint64_t> result;
    if (!data.empty()) {
        result.push_back(data[0]);
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] != data[i - 1]) {
                result.push_back(data[i]);
            }
        }
    }
    return result;
}

}  // namespace util
}  // namespace fh_index

#endif
