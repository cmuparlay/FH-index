#include "fh_index/fh_index_rw.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <utility>

int main() {
  using Key = std::uint64_t;
  using Value = std::uint64_t;

  const std::array<Key, 5> keys = {10, 20, 30, 40, 50};
  const std::array<Value, 5> values = {100, 200, 300, 400, 500};

  fh_index_rw::FHIndexRW<Key, Value> index(
      /*top_level_threshold=*/256,
      /*B=*/40, /*B_overflow=*/80, /*F=*/2,
      /*internal_B=*/16, /*internal_B_overflow=*/32, /*internal_F=*/1);
  index.BulkLoad(keys.size(), keys.data(), values.data());

  // Insert fails if the key exists; Update and Delete fail if it is absent.
  if (!index.Insert(25, 250)) return 1;
  if (!index.Update(20, 222)) return 1;
  if (!index.Delete(40)) return 1;

  Value value = 0;
  bool found = false;
  index.Find(20, value, found);
  if (!found) return 1;
  std::cout << "Find(20) after update: " << value << '\n';

  index.Find(40, value, found);
  std::cout << "Find(40) after delete: "
            << (found ? "found" : "not found") << '\n';

  std::array<std::pair<Key, Value>, 4> results;
  const auto count = index.Scan(20, results.size(), results.data());
  std::cout << "Scan from 20:\n";
  for (std::size_t i = 0; i < count; ++i) {
    const auto& [key, result_value] = results[i];
    std::cout << key << " -> " << result_value << '\n';
  }
}
