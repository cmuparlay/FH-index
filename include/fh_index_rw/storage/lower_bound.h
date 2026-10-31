#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fh_index_rw::lower_bound {

template <typename KeyType>
inline size_t Naive(KeyType key, const KeyType* keys, size_t size) {
  return static_cast<size_t>(
      std::lower_bound(keys, keys + size, key) - keys);
}

#ifdef __AVX2__
inline size_t Avx2(uint64_t key, const uint64_t* keys, size_t size) {
  const __m256i key_vector =
      _mm256_set1_epi64x(static_cast<long long>(key));
  const __m256i sign_bit =
      _mm256_set1_epi64x(static_cast<long long>(1ULL << 63));
  const __m256i signed_key = _mm256_xor_si256(key_vector, sign_bit);

  size_t index = 0;
  for (; index + 4 <= size; index += 4) {
    const __m256i values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(keys + index));
    const __m256i signed_values = _mm256_xor_si256(values, sign_bit);
    const __m256i greater =
        _mm256_cmpgt_epi64(signed_values, signed_key);
    const __m256i equal = _mm256_cmpeq_epi64(values, key_vector);
    const unsigned mask = static_cast<unsigned>(_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_or_si256(greater, equal))));
    if (mask != 0) {
      return index + static_cast<size_t>(__builtin_ctz(mask));
    }
  }

  for (; index < size; ++index) {
    if (keys[index] >= key) return index;
  }
  return size;
}
#endif

#ifdef __AVX512F__
inline size_t Avx512(uint64_t key, const uint64_t* keys, size_t size) {
  const __m512i key_vector =
      _mm512_set1_epi64(static_cast<long long>(key));

  size_t index = 0;
  for (; index + 8 <= size; index += 8) {
    const __m512i values = _mm512_loadu_si512(keys + index);
    const unsigned mask = _mm512_cmpge_epu64_mask(values, key_vector);
    if (mask != 0) {
      return index + static_cast<size_t>(__builtin_ctz(mask));
    }
  }

  for (; index < size; ++index) {
    if (keys[index] >= key) return index;
  }
  return size;
}
#endif

// Keep internal and leaf policies independent because their block-size
// distributions differ substantially.
template <typename KeyType>
inline size_t SearchInternal(KeyType key, const KeyType* keys, size_t size) {
  return Naive(key, keys, size);
}

template <typename KeyType>
inline size_t SearchLeaf(KeyType key, const KeyType* keys, size_t size) {
  if constexpr (std::is_same_v<std::remove_cv_t<KeyType>, uint64_t>) {
#ifdef __AVX512F__
    return Avx512(key, keys, size);
#elif defined(__AVX2__)
    return Avx2(key, keys, size);
#else
    return Naive(key, keys, size);
#endif
  } else {
    return Naive(key, keys, size);
  }
}

}  // namespace fh_index_rw::lower_bound
