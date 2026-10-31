#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "fh_index/fh_index_ro.h"
#include "fh_index/fh_index_rw.h"

using K = uint64_t;
using Pair = std::pair<K, K>;
constexpr K Max = std::numeric_limits<K>::max();
K Value(K k) { return k ^ 0x123456789abcdef0ULL; }

void Check(bool condition, const char* what) {
  if (!condition) throw std::runtime_error(what);
}

// Check observable behavior against an ordered reference, including scans
// whose upper endpoint is the largest representable key.
template <class Index>
void CheckQuery(Index& index, const std::map<K, K>& ref, K key) {
  K value = 0;
  bool found = false;
  index.Find(key, value, found);
  auto it = ref.find(key);
  Check(found == (it != ref.end()), "Find presence");
  if (found) Check(value == it->second, "Find value");
  for (size_t len : {size_t(0), size_t(1), size_t(7), size_t(3000)}) {
    std::vector<Pair> output(len);
    auto pos = ref.lower_bound(key);
    const size_t count = index.Scan(key, len, output.data());
    size_t expected = 0;
    for (; pos != ref.end() && expected < len; ++expected, ++pos) {
      Check(expected < count && output[expected] == Pair(*pos), "Scan content");
    }
    Check(count == expected, "Scan count");
  }
}

int main() {
  std::mt19937_64 rng(42);
  std::vector<K> large{0, 1, Max / 2, Max / 2 + 1, Max - 1, Max};
  for (size_t i = 0; i < 2048; ++i) large.push_back(rng());
  std::sort(large.begin(), large.end());
  large.erase(std::unique(large.begin(), large.end()), large.end());
  std::vector<std::vector<K>> sets{
      {0}, {Max}, {0, 1, 2, Max / 2, Max / 2 + 1, Max - 2, Max - 1, Max}, large};
  for (auto& keys : sets) {
    std::cerr << "SET " << keys.size() << '\n';
    std::vector<K> vals;
    std::map<K, K> ref;
    for (auto k : keys) {
      vals.push_back(Value(k));
      ref.emplace(k, Value(k));
    }
    fh_index_ro::FHIndexRO<K, K> ro(8, 16, 2);
    ro.BulkLoad(keys.size(), keys.data(), vals.data());
    for (auto k : keys) {
      CheckQuery(ro, ref, k);
      if (k) CheckQuery(ro, ref, k - 1);
      if (k < Max) CheckQuery(ro, ref, k + 1);
    }
    fh_index_rw::FHIndexRW<K, K> rw(8, 16, 32, 2, 8, 16, 1);
    rw.BulkLoad(keys.size(), keys.data(), vals.data());
    for (auto k : {K(0), K(1), Max / 2, Max / 2 + 1, Max - 1, Max}) {
      CheckQuery(rw, ref, k);
    }
    for (size_t i = 0; i < 1200; ++i) {
      K key = i % 3 ? keys[rng() % keys.size()] : rng();
      K val = rng();
      switch (i % 3) {
        case 0:
          Check(rw.Insert(key, val) == ref.emplace(key, val).second, "Insert");
          break;
        case 1:
          Check(rw.Delete(key) == (ref.erase(key) != 0), "Delete");
          break;
        case 2: {
          auto it = ref.find(key);
          bool expected = it != ref.end();
          Check(rw.Update(key, val) == expected, "Update");
          if (expected) it->second = val;
          break;
        }
      }
      if (i % 20 == 0) {
        CheckQuery(rw, ref, key);
        CheckQuery(rw, ref, 0);
        CheckQuery(rw, ref, Max);
      }
    }
    std::vector<K> remaining;
    for (auto [key, value] : ref) remaining.push_back(key);
    for (auto key : remaining) Check(rw.Delete(key), "Delete all");
    ref.clear();
    CheckQuery(rw, ref, 0);
    CheckQuery(rw, ref, Max);
    for (auto key : {K(0), Max}) {
      Check(rw.Insert(key, Value(key)), "Insert after empty");
      ref.emplace(key, Value(key));
    }
    CheckQuery(rw, ref, 0);
    CheckQuery(rw, ref, Max);
  }
  memory::ForceCleanupAll();
  std::cout << "BOUNDARIES PASS\n";
}
