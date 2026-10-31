#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "common_internal/construction/fraction.h"

using fh_index::internal::Fraction;

int main() {
  const Fraction reduced(6, 8);
  assert(reduced.numerator() == 3);
  assert(reduced.denominator() == 4);

  const Fraction density(3, 20);
  assert(density.FloorMultiply(19) == 2);
  assert(density.FloorMultiply(20) == 3);
  assert(density.CeilMultiply(21) == 4);

  assert(density.Halve() == Fraction(3, 40));
  assert(density.Double() == Fraction(3, 10));
  assert(Fraction(2, 3).Halve() == Fraction(1, 3));
  assert(Fraction(3, 4).Double() == Fraction(3, 2));

  assert(Fraction(1, 3) < Fraction(3, 8));
  assert(Fraction(3, 8) > Fraction(1, 3));

  // The 64-bit multiplication would overflow, but the exact result fits.
  const size_t maximum = std::numeric_limits<size_t>::max();
  assert(Fraction(1, 2).FloorMultiply(maximum) == maximum / 2);

  std::cout << "Fraction tests passed" << std::endl;
  return 0;
}
