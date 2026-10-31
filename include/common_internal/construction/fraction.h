#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_FRACTION_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_FRACTION_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace fh_index {
namespace internal {

// An exact non-negative rational number. Arithmetic that converts a fraction
// to an integer states its rounding direction explicitly and uses a 128-bit
// intermediate to avoid overflowing 64-bit products.
class Fraction {
 public:
  constexpr Fraction() = default;

  constexpr Fraction(uint64_t numerator, uint64_t denominator)
      : numerator_(numerator), denominator_(denominator) {
    assert(denominator != 0);
    if (denominator == 0) std::abort();
    const uint64_t divisor = std::gcd(numerator_, denominator_);
    numerator_ /= divisor;
    denominator_ /= divisor;
  }

  constexpr uint64_t numerator() const { return numerator_; }
  constexpr uint64_t denominator() const { return denominator_; }
  constexpr bool IsPositive() const { return numerator_ != 0; }

  size_t FloorMultiply(size_t value) const {
    const Wide product =
        static_cast<Wide>(value) * static_cast<Wide>(numerator_);
    return CheckedSize(product / denominator_);
  }

  size_t CeilMultiply(size_t value) const {
    const Wide product =
        static_cast<Wide>(value) * static_cast<Wide>(numerator_);
    const Wide quotient = product / denominator_;
    return CheckedSize(quotient + (product % denominator_ != 0));
  }

  Fraction Double() const {
    if ((denominator_ & 1) == 0) {
      return Fraction(numerator_, denominator_ / 2);
    }
    if (numerator_ > std::numeric_limits<uint64_t>::max() / 2) {
      std::abort();
    }
    return Fraction(numerator_ * 2, denominator_);
  }

  Fraction Halve() const {
    if ((numerator_ & 1) == 0) {
      return Fraction(numerator_ / 2, denominator_);
    }
    if (denominator_ > std::numeric_limits<uint64_t>::max() / 2) {
      std::abort();
    }
    return Fraction(numerator_, denominator_ * 2);
  }

  friend bool operator==(const Fraction& lhs, const Fraction& rhs) {
    return lhs.numerator_ == rhs.numerator_ &&
           lhs.denominator_ == rhs.denominator_;
  }

  friend bool operator!=(const Fraction& lhs, const Fraction& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const Fraction& lhs, const Fraction& rhs) {
    return static_cast<Wide>(lhs.numerator_) * rhs.denominator_ <
           static_cast<Wide>(rhs.numerator_) * lhs.denominator_;
  }

  friend bool operator>(const Fraction& lhs, const Fraction& rhs) {
    return rhs < lhs;
  }

 private:
  using Wide = __uint128_t;

  static size_t CheckedSize(Wide value) {
    if (value > std::numeric_limits<size_t>::max()) std::abort();
    return static_cast<size_t>(value);
  }

  uint64_t numerator_{0};
  uint64_t denominator_{1};
};

}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_FRACTION_H
