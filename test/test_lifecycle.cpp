#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "fh_index/fh_index_ro.h"
#include "fh_index/fh_index_rw.h"

using Key = std::uint64_t;
using RO = fh_index_ro::FHIndexRO<Key, Key>;
using RW = fh_index_rw::FHIndexRW<Key, Key>;

static_assert(!std::is_copy_constructible_v<RO>);
static_assert(!std::is_copy_assignable_v<RO>);
static_assert(std::is_nothrow_move_constructible_v<RO>);
static_assert(std::is_nothrow_move_assignable_v<RO>);

void Require(bool condition) {
  if (!condition) throw std::runtime_error("lifecycle regression");
}

template <typename Index>
void CheckValues(Index& index, const std::vector<Key>& keys,
                 const std::vector<Key>& values) {
  for (std::size_t i = 0; i < keys.size(); ++i) {
    Key value = 0;
    bool found = false;
    index.Find(keys[i], value, found);
    Require(found && value == values[i]);
  }
}

void CheckOwnership(std::size_t size) {
  std::vector<Key> keys(size), values(size);
  for (std::size_t i = 0; i < size; ++i) {
    keys[i] = 2 * i + 2;
    values[i] = keys[i] * 10;
  }
  {
    RO source(8, 16, 2);
    source.BulkLoad(size, keys.data(), values.data());
    RO moved(std::move(source));
    CheckValues(moved, keys, values);
    source.BulkLoad(size, keys.data(), values.data());
    CheckValues(source, keys, values);
    RO assigned(8, 16, 2);
    assigned.BulkLoad(size, keys.data(), values.data());
    assigned = std::move(moved);
    CheckValues(assigned, keys, values);
    moved.BulkLoad(size, keys.data(), values.data());
    CheckValues(moved, keys, values);
    // Moving through vector reallocation must transfer ownership as well.
    std::vector<RO> indexes;
    indexes.push_back(std::move(assigned));
    indexes.push_back(std::move(source));
    CheckValues(indexes[0], keys, values);
    CheckValues(indexes[1], keys, values);
  }
  {
    RW index(8, 16, 32, 2, 8, 16, 1);
    for (int round = 0; round < 4; ++round) {
      index.BulkLoad(size, keys.data(), values.data());
      CheckValues(index, keys, values);
      for (std::size_t i = 0; i < size; ++i) {
        Require(index.Insert(keys[i] + 1, values[i] + 1));
      }
      for (std::size_t i = 0; i < size; ++i) {
        Require(index.Delete(keys[i] + 1));
      }
      CheckValues(index, keys, values);
    }
  }
}

struct alignas(16) WideValue {
  Key first;
  Key second;
};

void CheckAlignedValues() {
  using Index = fh_index_rw::FHIndexRW<Key, WideValue>;
  std::array<Key, 5> keys{2, 4, 6, 8, 10};
  std::array<WideValue, 5> values{};
  for (std::size_t i = 0; i < keys.size(); ++i) {
    values[i] = {keys[i] * 10, keys[i] * 100};
  }
  Index index(8, 16, 32, 2, 8, 16, 1);
  index.BulkLoad(keys.size(), keys.data(), values.data());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    WideValue value{};
    bool found = false;
    index.Find(keys[i], value, found);
    Require(found && value.first == values[i].first &&
            value.second == values[i].second);
  }
  Require(index.Insert(7, {70, 700}));
  Require(index.Update(6, {600, 6000}));
  Require(index.Delete(4));
  std::array<std::pair<Key, WideValue>, 6> result{};
  Require(index.Scan(0, result.size(), result.data()) == 5);
  Require(result[1].first == 6 && result[1].second.first == 600);
  Require(result[2].first == 7 && result[2].second.second == 700);
}

int main() {
  CheckOwnership(1);
  CheckOwnership(5);
  CheckOwnership(3000);
  CheckAlignedValues();
  memory::ForceCleanupAll();
}
