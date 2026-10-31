#include "fh_index/fh_index_ro.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <utility>

int main() {
  using Key = std::uint64_t;
  using Value = std::uint64_t;

  // BulkLoad requires non-empty, sorted, unique keys.
  const std::array<Key, 5> keys = {10, 20, 30, 40, 50};
  const std::array<Value, 5> values = {100, 200, 300, 400, 500};

  fh_index_ro::FHIndexRO<Key, Value> index(
      /*top_level_threshold=*/256, /*B=*/40, /*F=*/2);
  index.BulkLoad(keys.size(), keys.data(), values.data());

  Value value = 0;
  bool found = false;
  index.Find(30, value, found);
  if (!found) return 1;
  std::cout << "Find(30): " << value << '\n';

  index.Find(35, value, found);
  std::cout << "Find(35): " << (found ? "found" : "not found") << '\n';

  // Scan returns entries starting at the first key >= 20.
  std::array<std::pair<Key, Value>, 3> results;
  const auto count = index.Scan(20, results.size(), results.data());
  std::cout << "Scan from 20:\n";
  for (std::size_t i = 0; i < count; ++i) {
    const auto& [key, result_value] = results[i];
    std::cout << key << " -> " << result_value << '\n';
  }
}
