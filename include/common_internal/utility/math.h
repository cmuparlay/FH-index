#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include <cstdint>
#include <cassert>
// #include "assert_test.h"

namespace fh_index {
namespace internal {

// Log2 related bit tricks

constexpr inline uint64_t Log2(uint64_t val) {
    assert(val != 0);
    return 63 - __builtin_clzll(val);
}

inline uint64_t Pow2(uint64_t step) {
    assert((step <= 64));
    if (step == 64) {
        // (1ull << 64) may not be 0 in llvm 
        return 0;
    } else {
        return 1ull << step;
    }
}

inline uint64_t Pow2Minus1(uint64_t k) {
    return Pow2(k) - 1;
}

// return the highest bit of val
inline uint64_t HighBit(uint64_t val) {
    return (val == 0) ? 0 : 1ull << Log2(val);
}

// return the lowest bit of val
constexpr inline uint64_t LowBit(uint64_t val) { return val & (-val); }

constexpr inline bool IsPow2Or0(uint64_t val) { return val == LowBit(val); }
constexpr inline bool IsPow2(uint64_t val) { return (val != 0) && IsPow2Or0(val); }
inline bool IsPow2Minus1(uint64_t val) { return IsPow2Or0(val + 1); }

inline uint64_t NextPow2(uint64_t val) {
    assert((val != 0 && val < (1ull << 63)));
    return Pow2(Log2(val) + 1);
}

inline uint64_t RoundDown(uint64_t key, uint64_t to) {
    assert((to == LowBit(to)));
    return key & ~(to - 1);
}
// return key % (1 << log2_step)
inline uint64_t RoundDownLog(uint64_t key, uint64_t log2_step) {
    assert(log2_step <= 63);
    return RoundDown(key, 1ull << log2_step);
}

inline uint64_t RoundUp(uint64_t key, uint64_t to) {
    assert(!(key > (1ull << 63) && to == (1ull << 63)));
    assert((to == LowBit(to)));
    return (key + to - 1) & ~(to - 1);
}

// return key + (1 << log2_step) - 1 - key % (1 << log2_step)
inline uint64_t RoundUpLog(uint64_t key, uint64_t log2_step) {
    assert(log2_step <= 63);
    return RoundUp(key, 1ull << log2_step);
}

// return (a + b) / 2 without overflow
inline uint64_t average(uint64_t a, uint64_t b) {
    return (a >> 1) + (b >> 1) + (a & b & 1);
}

}  // namespace internal
}  // namespace fh_index

#endif
