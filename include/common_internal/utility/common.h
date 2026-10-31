#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H

#include <cstdio>
#include <cstdlib>

// a assert function running even with NDEBUG
#define RUNTIME_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #expr, __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)

// Branch prediction hints
#if defined(__GNUC__) && (__GNUC__ >= 9)
#define UNLIKELY(x) __builtin_expect((x), 0)
#else
#define UNLIKELY(x) (x)
#endif

#ifdef __AVX512F__
  constexpr static bool kHasAvx512 = true;
#else
  constexpr static bool kHasAvx512 = false;
#endif

#ifdef __AVX2__
  constexpr static bool kHasAvx2 = true;
#else
  constexpr static bool kHasAvx2 = false;
#endif

namespace fh_index {
namespace internal {

// Array padding size: all arrays are padded to multiples of this value
// This ensures proper alignment for SIMD operations and consistent boundary handling
constexpr size_t kArrayPaddingSize = 64;

}  // namespace internal
}  // namespace fh_index

#endif
