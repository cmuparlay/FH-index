#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "fh_index_rw/storage/kv_layer_rw.h"
#include "fh_index_rw/storage/lower_bound.h"

namespace {

using Key = uint64_t;
using Value = uint64_t;
using KVLayer = fh_index_rw::KVLayerRW<Key, Value>;
using Block = KVLayer::BlockType;

class TestContext {
 public:
  explicit TestContext(std::string name) : name_(std::move(name)) {}

  void Expect(bool condition, const char* expression, int line) {
    if (condition) return;
    ++failures_;
    std::cerr << "  line " << line << ": " << expression << " failed\n";
  }

  bool Finish() const {
    std::cout << '[' << (failures_ == 0 ? "PASS" : "FAIL") << "] "
              << name_;
    if (failures_ != 0) std::cout << " (" << failures_ << " failures)";
    std::cout << '\n';
    return failures_ == 0;
  }

 private:
  std::string name_;
  size_t failures_ = 0;
};

#define EXPECT(context, condition) \
  (context).Expect(static_cast<bool>(condition), #condition, __LINE__)

class OwnedBlock {
 public:
  OwnedBlock() = default;
  OwnedBlock(Block* block, size_t capacity)
      : block_(block), capacity_(capacity) {}

  OwnedBlock(const OwnedBlock&) = delete;
  OwnedBlock& operator=(const OwnedBlock&) = delete;

  OwnedBlock(OwnedBlock&& other) noexcept
      : block_(other.block_), capacity_(other.capacity_) {
    other.block_ = nullptr;
    other.capacity_ = 0;
  }

  OwnedBlock& operator=(OwnedBlock&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    block_ = other.block_;
    capacity_ = other.capacity_;
    other.block_ = nullptr;
    other.capacity_ = 0;
    return *this;
  }

  ~OwnedBlock() { Reset(); }

  Block* get() const { return block_; }
  Block* operator->() const { return block_; }
  size_t capacity() const { return capacity_; }

  void Reset() {
    if (block_ != nullptr) Block::Destroy(block_, capacity_);
    block_ = nullptr;
    capacity_ = 0;
  }

 private:
  Block* block_ = nullptr;
  size_t capacity_ = 0;
};

OwnedBlock MakeBlock(const std::vector<Key>& keys,
                     const std::vector<Value>& values,
                     size_t capacity = 0) {
  if (capacity == 0) capacity = keys.size();
  auto [block, actual_capacity] = Block::CreateWithCapacity(capacity);
  for (size_t i = 0; i < keys.size(); ++i) {
    block->keys()[i] = keys[i];
    block->values(actual_capacity)[i] = values[i];
  }
  return OwnedBlock(block, actual_capacity);
}

bool Matches(const Block* block, size_t size, size_t capacity,
             const std::vector<Key>& keys,
             const std::vector<Value>& values) {
  if (block == nullptr || size != keys.size() || size != values.size()) {
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    if (block->keys()[i] != keys[i] ||
        block->values(capacity)[i] != values[i]) {
      return false;
    }
  }
  return true;
}

bool TestCapacityPolicy() {
  TestContext test("capacity policy and block lifecycle");

  EXPECT(test, fh_index_rw::ComputeBlockCapacity(1) == 1);
  EXPECT(test, fh_index_rw::ComputeBlockCapacity(79) == 79);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(1, 0, 80) == 4);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(5, 4, 80) == 6);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(7, 6, 80) == 8);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(9, 8, 80) == 11);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(79, 64, 80) == 80);
  EXPECT(test, fh_index_rw::ComputeMutableBlockCapacity(80, 80, 80) == 80);

  for (const size_t capacity : {1U, 4U, 8U, 40U, 79U}) {
    auto [block, actual_capacity] = Block::CreateWithCapacity(capacity);
    EXPECT(test, block != nullptr);
    EXPECT(test, actual_capacity == capacity);
    EXPECT(test, (reinterpret_cast<uintptr_t>(block) & 15U) == 0);
    EXPECT(test, block->values(actual_capacity) ==
                     reinterpret_cast<Value*>(block->keys() + capacity));
    EXPECT(test, Block::AllocationBytes(capacity) ==
                     capacity * (sizeof(Key) + sizeof(Value)));
    block->keys()[capacity - 1] = capacity;
    block->values(actual_capacity)[capacity - 1] = capacity + 1;
    EXPECT(test, block->keys()[capacity - 1] == capacity);
    EXPECT(test, block->values(actual_capacity)[capacity - 1] == capacity + 1);
    Block::Destroy(block, actual_capacity);
  }

  alignas(16) std::array<std::byte, sizeof(Block)> storage{};
  Block* in_place = Block::ConstructAt(storage.data());
  EXPECT(test, in_place == reinterpret_cast<Block*>(storage.data()));
  Block::DestroyAt(in_place);

  return test.Finish();
}

bool TestLowerBoundImplementations() {
  TestContext test("lower-bound implementations");
  const Key dummy = 0;

  for (size_t size = 0; size <= 79; ++size) {
    for (const bool cross_sign_bit : {false, true}) {
      std::vector<Key> keys(size);
      const Key base = cross_sign_bit
                           ? (1ULL << 63) - 3 * size
                           : 1000 + 7 * size;
      for (size_t i = 0; i < size; ++i) keys[i] = base + 3 * i;
      const Key* data = keys.empty() ? &dummy : keys.data();

      std::vector<Key> queries = {
          0, base, std::numeric_limits<Key>::max()};
      for (const Key key : keys) {
        if (key != 0) queries.push_back(key - 1);
        queries.push_back(key);
        if (key != std::numeric_limits<Key>::max()) queries.push_back(key + 1);
      }

      for (const Key query : queries) {
        const size_t expected = static_cast<size_t>(
            std::lower_bound(data, data + size, query) - data);
        EXPECT(test, fh_index_rw::lower_bound::Naive(
                         query, data, size) == expected);
#ifdef __AVX2__
        EXPECT(test, fh_index_rw::lower_bound::Avx2(
                         query, data, size) == expected);
#endif
#ifdef __AVX512F__
        EXPECT(test, fh_index_rw::lower_bound::Avx512(
                         query, data, size) == expected);
#endif
        EXPECT(test, fh_index_rw::lower_bound::SearchInternal(
                         query, data, size) == expected);
        EXPECT(test, fh_index_rw::lower_bound::SearchLeaf(
                         query, data, size) == expected);
      }

      if (!keys.empty()) {
        std::vector<Value> values(size);
        for (size_t i = 0; i < size; ++i) values[i] = i;
        OwnedBlock block = MakeBlock(keys, values);
        for (const Key query : queries) {
          const size_t expected = static_cast<size_t>(
              std::lower_bound(data, data + size, query) - data);
          EXPECT(test, KVLayer::LowerBoundInInternalBlock(
                           query, block.get(), size) == expected);
          EXPECT(test, KVLayer::LowerBoundInLeafBlock(
                           query, block.get(), size) == expected);
        }
      }
    }
  }

  return test.Finish();
}

bool TestBulkLoad() {
  TestContext test("bulk load");
  const std::vector<Key> keys = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  const std::vector<Value> values = {10, 30, 50, 70, 90,
                                     110, 130, 150, 170, 190};
  const std::array<size_t, 4> breakpoints = {0, 3, 3, 7};

  auto blocks = KVLayer::BulkLoad(keys.data(), values.data(), keys.size(),
                                  breakpoints.data(), breakpoints.size());
  EXPECT(test, blocks.size() == breakpoints.size());

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const size_t begin = breakpoints[block_index];
    const size_t end = block_index + 1 < breakpoints.size()
                           ? breakpoints[block_index + 1]
                           : keys.size();
    const size_t size = end - begin;
    if (size == 0) {
      EXPECT(test, blocks[block_index] == nullptr);
      continue;
    }
    EXPECT(test, blocks[block_index] != nullptr);
    const size_t capacity = fh_index_rw::ComputeBlockCapacity(size);
    for (size_t i = 0; i < size; ++i) {
      EXPECT(test, blocks[block_index]->keys()[i] == keys[begin + i]);
      EXPECT(test, blocks[block_index]->values(capacity)[i] ==
                       values[begin + i]);
    }
    Block::Destroy(blocks[block_index], capacity);
  }

  auto empty = KVLayer::BulkLoad(keys.data(), values.data(), keys.size(),
                                 nullptr, 0);
  EXPECT(test, empty.empty());
  return test.Finish();
}

bool TestLookupAndPositions() {
  TestContext test("lookup and insertion positions");
  OwnedBlock block = MakeBlock({10, 20, 30, 40},
                               {100, 200, 300, 400}, 8);
  constexpr size_t size = 4;

  for (size_t i = 0; i < size; ++i) {
    Key out_key = 0;
    Value out_value = 0;
    EXPECT(test, KVLayer::Get(block->keys()[i], block.get(), size,
                              block.capacity(), out_key, out_value));
    EXPECT(test, out_key == block->keys()[i]);
    EXPECT(test, out_value == block->values(block.capacity())[i]);
  }

  for (const Key missing : {0ULL, 15ULL, 25ULL, 50ULL}) {
    Key out_key = 0;
    Value out_value = 0;
    EXPECT(test, !KVLayer::Get(missing, block.get(), size,
                               block.capacity(), out_key, out_value));
  }

  struct PositionCase {
    Key key;
    size_t position;
    bool found;
  };
  const PositionCase positions[] = {
      {5, 0, false}, {10, 0, true}, {25, 2, false},
      {40, 3, true}, {50, 4, false}};
  for (const auto& position : positions) {
    const auto [actual_position, found] =
        KVLayer::FindKeyOrInsertionPosition(
            block.get(), size, position.key);
    EXPECT(test, actual_position == position.position);
    EXPECT(test, found == position.found);
  }
  const auto [empty_position, empty_found] =
      KVLayer::FindKeyOrInsertionPosition(nullptr, 0, 1);
  EXPECT(test, empty_position == 0);
  EXPECT(test, !empty_found);

  struct LowerBoundCase {
    Key query;
    bool found;
    Key key;
    Value value;
  };
  const LowerBoundCase lower_bounds[] = {
      {5, true, 10, 100}, {10, true, 10, 100},
      {25, true, 30, 300}, {40, true, 40, 400},
      {50, false, 0, 0}};
  for (const auto& lower_bound : lower_bounds) {
    Key out_key = 0;
    Value out_value = 0;
    const bool found = KVLayer::GetLowerBoundKV(
        lower_bound.query, block.get(), size, block.capacity(),
        out_key, out_value);
    EXPECT(test, found == lower_bound.found);
    if (found) {
      EXPECT(test, out_key == lower_bound.key);
      EXPECT(test, out_value == lower_bound.value);
    }
  }

  return test.Finish();
}

bool TestInPlaceMutations() {
  TestContext test("in-place mutations");
  OwnedBlock block = MakeBlock({10, 20, 30, 40},
                               {100, 200, 300, 400}, 8);
  size_t size = 4;

  KVLayer::InsertInPlaceAt(2, 25, 250, block.get(), size, block.capacity());
  ++size;
  EXPECT(test, Matches(block.get(), size, block.capacity(),
                       {10, 20, 25, 30, 40},
                       {100, 200, 250, 300, 400}));

  KVLayer::UpdateInPlaceAt(
      2, 251, block.get(), size, block.capacity());
  EXPECT(test, block->values(block.capacity())[2] == 251);

  KVLayer::DeleteInPlaceAt(0, block.get(), size, block.capacity());
  --size;
  EXPECT(test, Matches(block.get(), size, block.capacity(),
                       {20, 25, 30, 40}, {200, 251, 300, 400}));

  KVLayer::InsertInPlaceAt(size, 50, 500,
                           block.get(), size, block.capacity());
  ++size;
  KVLayer::InsertInPlaceAt(0, 5, 50,
                           block.get(), size, block.capacity());
  ++size;
  EXPECT(test, Matches(block.get(), size, block.capacity(),
                       {5, 20, 25, 30, 40, 50},
                       {50, 200, 251, 300, 400, 500}));

  return test.Finish();
}

bool TestInsertReplacement() {
  TestContext test("insert replacement path");

  {
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithInsertAt(
            nullptr, 0, 0, 0, 10, 100, 80);
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, new_size == 1);
    EXPECT(test, new_capacity == 4);
    EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                         {10}, {100}));
  }

  {
    OwnedBlock old = MakeBlock({10, 20, 30, 40},
                               {100, 200, 300, 400}, 4);
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithInsertAt(
            old.get(), 4, old.capacity(), 2, 25, 250, 80);
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, new_size == 5);
    EXPECT(test, new_capacity == 6);
    EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                         {10, 20, 25, 30, 40},
                         {100, 200, 250, 300, 400}));
    EXPECT(test, Matches(old.get(), 4, old.capacity(),
                         {10, 20, 30, 40}, {100, 200, 300, 400}));
  }

  {
    OwnedBlock old = MakeBlock({10, 20, 30, 40},
                               {100, 200, 300, 400}, 8);
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithInsertAt(
            old.get(), 4, old.capacity(), 4, 50, 500, 80,
            old.capacity());
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, new_size == 5);
    EXPECT(test, new_capacity == 8);
    EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                         {10, 20, 30, 40, 50},
                         {100, 200, 300, 400, 500}));
  }

  {
    OwnedBlock old = MakeBlock({10, 20, 30, 40},
                               {100, 200, 300, 400}, 4);
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithInsertAt(
            old.get(), 4, old.capacity(), 4, 50, 500, 5);
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, new_size == 5);
    EXPECT(test, new_capacity == 5);
  }

  return test.Finish();
}

bool TestUpdateReplacement() {
  TestContext test("update replacement path");
  OwnedBlock old = MakeBlock({10, 20, 30, 40},
                             {100, 200, 300, 400}, 8);

  auto [new_block, new_size, new_capacity] =
      KVLayer::CreateBlockWithUpdateAt(
          old.get(), 4, old.capacity(), 2, 333);
  OwnedBlock replacement(new_block, new_capacity);
  EXPECT(test, new_size == 4);
  EXPECT(test, new_capacity == old.capacity());
  EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                       {10, 20, 30, 40}, {100, 200, 333, 400}));
  EXPECT(test, Matches(old.get(), 4, old.capacity(),
                       {10, 20, 30, 40}, {100, 200, 300, 400}));

  return test.Finish();
}

bool TestDeleteReplacement() {
  TestContext test("delete replacement path");
  const std::vector<Key> original_keys = {10, 20, 30, 40, 50};
  const std::vector<Value> original_values = {100, 200, 300, 400, 500};

  struct DeleteCase {
    size_t position;
    std::vector<Key> keys;
    std::vector<Value> values;
  };
  const std::vector<DeleteCase> cases = {
      {0, {20, 30, 40, 50}, {200, 300, 400, 500}},
      {2, {10, 20, 40, 50}, {100, 200, 400, 500}},
      {4, {10, 20, 30, 40}, {100, 200, 300, 400}}};

  for (const auto& delete_case : cases) {
    OwnedBlock old = MakeBlock(original_keys, original_values, 8);
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithDeleteAt(
            old.get(), original_keys.size(), old.capacity(),
            delete_case.position);
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, new_size == original_keys.size() - 1);
    EXPECT(test, new_capacity == old.capacity());
    EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                         delete_case.keys, delete_case.values));
    EXPECT(test, Matches(old.get(), original_keys.size(), old.capacity(),
                         original_keys, original_values));
  }

  OwnedBlock single = MakeBlock({10}, {100});
  auto [empty_block, empty_size, empty_capacity] =
      KVLayer::CreateBlockWithDeleteAt(
          single.get(), 1, single.capacity(), 0);
  EXPECT(test, empty_block == nullptr);
  EXPECT(test, empty_size == 0);
  EXPECT(test, empty_capacity == 0);

  return test.Finish();
}

bool TestDescriptorSplice() {
  TestContext test("descriptor splice");
  const std::vector<Key> original_keys = {10, 20, 30, 40, 50};
  const std::vector<Value> original_values = {100, 200, 300, 400, 500};
  OwnedBlock old = MakeBlock(original_keys, original_values);

  struct ReplaceCase {
    Key replaced_key;
    std::vector<Key> inserted_keys;
    std::vector<Value> inserted_values;
    std::vector<Key> expected_keys;
    std::vector<Value> expected_values;
  };
  const std::vector<ReplaceCase> cases = {
      {10, {5, 7, 9}, {50, 70, 90},
       {5, 7, 9, 20, 30, 40, 50},
       {50, 70, 90, 200, 300, 400, 500}},
      {30, {25, 28, 32, 35}, {250, 280, 320, 350},
       {10, 20, 25, 28, 32, 35, 40, 50},
       {100, 200, 250, 280, 320, 350, 400, 500}},
      {50, {}, {}, {10, 20, 30, 40}, {100, 200, 300, 400}}};

  for (const auto& replace_case : cases) {
    auto [new_block, new_size, new_capacity] =
        KVLayer::CreateBlockWithReplace(
            old.get(), original_keys.size(), old.capacity(),
            replace_case.replaced_key,
            replace_case.inserted_keys.data(),
            replace_case.inserted_values.data(),
            replace_case.inserted_keys.size());
    OwnedBlock replacement(new_block, new_capacity);
    EXPECT(test, Matches(replacement.get(), new_size, new_capacity,
                         replace_case.expected_keys,
                         replace_case.expected_values));
    EXPECT(test, Matches(old.get(), original_keys.size(), old.capacity(),
                         original_keys, original_values));
  }

  return test.Finish();
}

}  // namespace

int main() {
  size_t failures = 0;
  failures += !TestCapacityPolicy();
  failures += !TestLowerBoundImplementations();
  failures += !TestBulkLoad();
  failures += !TestLookupAndPositions();
  failures += !TestInPlaceMutations();
  failures += !TestInsertReplacement();
  failures += !TestUpdateReplacement();
  failures += !TestDeleteReplacement();
  failures += !TestDescriptorSplice();

  if (failures == 0) {
    std::cout << "ALL KV LAYER RW TESTS PASSED\n";
    return 0;
  }
  std::cerr << failures << " KV LAYER RW TEST GROUPS FAILED\n";
  return 1;
}
