#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common_internal/memory/epoch_reclamation.h"
#include "lower_bound.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"

namespace fh_index_rw {

// Compute block capacity from block size.
// Simply returns the exact size needed.
inline size_t ComputeBlockCapacity(size_t block_size) {
  return block_size;
}

// Grow only a block that actually runs out of room. Bulk-loaded blocks remain
// exact-sized so the many small buckets do not all pay for unused slots.
inline size_t ComputeMutableBlockCapacity(size_t block_size,
                                          size_t current_capacity,
                                          size_t max_capacity) {
  assert(block_size <= max_capacity);
  // Grow by ceil(1.3 * current_capacity). This keeps enough spare capacity
  // for in-place inserts without the persistent slack caused by doubling.
  const size_t grown =
      std::max<size_t>(4, (13 * current_capacity + 9) / 10);
  return std::max(block_size, std::min(grown, max_capacity));
}

// Block structure - contiguous memory for keys and values. Publication state,
// logical size, physical capacity, and mutation version live in the finger.
// Layout: [keys: KeyType * capacity][alignment padding][values: ValueType * capacity]
template <typename KeyType, typename ValueType>
struct Block {
    static_assert(std::is_trivially_copyable_v<KeyType> &&
                  std::is_trivially_copyable_v<ValueType>,
                  "RW blocks require trivially copyable keys and values");
    static_assert(alignof(KeyType) <= alignof(std::max_align_t) &&
                  alignof(ValueType) <= alignof(std::max_align_t),
                  "RW blocks do not support over-aligned keys or values");

    static size_t ValueOffset(size_t capacity) {
      const size_t key_bytes = capacity * sizeof(KeyType);
      return (key_bytes + alignof(ValueType) - 1) & ~(alignof(ValueType) - 1);
    }

    KeyType* keys() {
      return reinterpret_cast<KeyType*>(this);
    }
    const KeyType* keys() const {
      return reinterpret_cast<const KeyType*>(this);
    }

    ValueType* values(size_t capacity) {
      return reinterpret_cast<ValueType*>(
          reinterpret_cast<unsigned char*>(this) + ValueOffset(capacity));
    }
    const ValueType* values(size_t capacity) const {
      return reinterpret_cast<const ValueType*>(
          reinterpret_cast<const unsigned char*>(this) + ValueOffset(capacity));
    }

    static size_t AllocationBytes(size_t capacity) {
      assert(capacity > 0 && "An empty block must be represented by nullptr");
      return ValueOffset(capacity) + capacity * sizeof(ValueType);
    }

    // Allocate an exact-capacity block.
    static std::pair<Block*, size_t> CreateWithCapacity(size_t cap) {
      size_t actual_capacity = ComputeBlockCapacity(cap);
      size_t total_bytes = AllocationBytes(actual_capacity);
      void* mem = memory::Allocate(total_bytes);
      assert(mem != nullptr);
      assert((reinterpret_cast<uintptr_t>(mem) & 15) == 0 &&
             "memory::Allocate must provide 16-byte alignment");
      Block* block = new (mem) Block();
      return {block, actual_capacity};
    }

    // Start a Block object's lifetime in aligned caller-owned storage.
    static Block* ConstructAt(void* storage) {
      return new (storage) Block();
    }

    // End the lifetime of a Block created with ConstructAt. The caller owns
    // and releases the surrounding allocation.
    static void DestroyAt(Block* block) {
      if (block != nullptr) {
        block->~Block();
      }
    }

    // Destroy an exclusively owned block.
    static void Destroy(Block* block, size_t capacity) {
      assert(block != nullptr);
      assert(capacity > 0);
      block->~Block();
      memory::Free(block, AllocationBytes(capacity));
    }

    // Retire a block via epoch-based reclamation.
    // The block will be freed after all concurrent readers finish.
    static void Retire(Block* block, size_t capacity) {
      if (block == nullptr) return;
      memory::Retire(static_cast<void*>(block),
                     AllocationBytes(capacity));
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(Block&&) = delete;

private:
    Block() = default;
    ~Block() = default;
};

// Stateless utility class for KV layer operations.
// All methods are static - no instance state needed.
template <typename KeyType, typename ValueType>
class KVLayerRW {

 public:
  using BlockType = Block<KeyType, ValueType>;

  // BulkLoad creates exact-sized blocks. Slack is added only when a mutable
  // leaf block actually needs to grow.
  // Note: Block size is NOT stored in the block itself. It is stored in the Finger.
  // Blocks are managed via epoch-based reclamation, not owned by KVLayerRW.
  static parlay::sequence<BlockType*> BulkLoad(
      const KeyType* keys, const ValueType* values, size_t size,
      const size_t* breakpoints, size_t num_breakpoints) {

    // Parallel allocation and initialization using parlay::sequence
    parlay::sequence<BlockType*> blocks(num_breakpoints);

    parlay::parallel_for(0, num_breakpoints, [&](size_t i) {
      size_t start = breakpoints[i];
      size_t end = (i + 1 < num_breakpoints) ? breakpoints[i + 1] : size;
      size_t block_elem_count = end - start;

      // Skip empty blocks
      if (block_elem_count == 0) {
        blocks[i] = nullptr;
        return;
      }

      auto [block, capacity] =
          BlockType::CreateWithCapacity(block_elem_count);
      blocks[i] = block;

      // Copy data to block
      KeyType* block_keys = blocks[i]->keys();
      ValueType* block_values = blocks[i]->values(capacity);
      for (size_t j = 0; j < block_elem_count; j++) {
        block_keys[j] = keys[start + j];
        block_values[j] = values[start + j];
      }
    });

    return blocks;
  }

  static bool Get(KeyType key, BlockType* block, size_t block_size, size_t capacity,
                  KeyType& out_key, ValueType& out_value) {
    assert(block != nullptr);

    size_t pos = LowerBoundInLeafBlock(key, block, block_size);

    if (pos < block_size && block->keys()[pos] == key) {
      out_key = block->keys()[pos];
      out_value = block->values(capacity)[pos];
      return true;
    }

    return false;
  }

  static std::pair<size_t, bool> FindKeyOrInsertionPosition(
      BlockType* block, size_t block_size, KeyType key) {
    if (block == nullptr || block_size == 0) return {0, false};
    const size_t pos = LowerBoundInLeafBlock(key, block, block_size);
    return {pos, pos < block_size && block->keys()[pos] == key};
  }

  // Get entry using lower_bound: finds first entry with key >= query key
  // Returns true if found a valid entry (any entry with key >= query key)
  // Only searches within the current block. Cross-block logic is in FHIndexRW.
  static bool GetLowerBoundKV(KeyType key, BlockType* block, size_t block_size, size_t capacity,
                              KeyType& out_key, ValueType& out_value) {
    assert(block != nullptr);

    size_t pos = LowerBoundInInternalBlock(key, block, block_size);

    if (pos < block_size) {
      out_key = block->keys()[pos];
      out_value = block->values(capacity)[pos];
      return true;
    }

    return false;
  }

  // The caller must hold the block's finger writer lock.
  static void InsertInPlaceAt(size_t insert_pos, KeyType key, ValueType value,
                              BlockType* block, size_t block_size,
                              size_t capacity) {
    assert(block != nullptr && block_size < capacity);
    assert(insert_pos <= block_size);
    KeyType* keys = block->keys();
    ValueType* values = block->values(capacity);
    assert(insert_pos == block_size || keys[insert_pos] != key);
    for (size_t i = block_size; i > insert_pos; --i) {
      keys[i] = keys[i - 1];
      values[i] = values[i - 1];
    }
    keys[insert_pos] = key;
    values[insert_pos] = value;
  }

  // The caller must hold the block's finger writer lock.
  static void UpdateInPlaceAt(size_t update_pos, ValueType value,
                              BlockType* block, size_t block_size,
                              size_t capacity) {
    assert(block != nullptr && update_pos < block_size);
    block->values(capacity)[update_pos] = value;
  }

  // The caller must hold the block's finger writer lock.
  static void DeleteInPlaceAt(size_t delete_pos, BlockType* block,
                              size_t block_size, size_t capacity) {
    assert(block != nullptr && delete_pos < block_size);
    KeyType* keys = block->keys();
    ValueType* values = block->values(capacity);
    for (size_t i = delete_pos; i + 1 < block_size; ++i) {
      keys[i] = keys[i + 1];
      values[i] = values[i + 1];
    }
  }

  static size_t LowerBoundInLeafBlock(KeyType key, const BlockType* block,
                                      size_t n) {
    assert(block != nullptr);
    return lower_bound::SearchLeaf(key, block->keys(), n);
  }

  static size_t LowerBoundInInternalBlock(KeyType key, const BlockType* block,
                                          size_t n) {
    assert(block != nullptr);
    return lower_bound::SearchInternal(key, block->keys(), n);
  }

  // Create a new block with an inserted key-value pair.
  // The input block can be nullptr (empty block case).
  // Returns {new_block, new_size, new_capacity}.
  // The caller is responsible for retiring the old block after successful CAS.
  static std::tuple<BlockType*, size_t, size_t> CreateBlockWithInsertAt(
      BlockType* old_block, size_t old_size, size_t old_capacity,
      size_t insert_pos, KeyType key, ValueType value,
      size_t max_capacity = 0, size_t replacement_capacity = 0) {
    assert(insert_pos <= old_size);

    size_t new_size = old_size + 1;
    size_t requested_capacity = replacement_capacity != 0
                                    ? replacement_capacity
                                    : max_capacity == 0
                                          ? new_size
                                          : ComputeMutableBlockCapacity(
                                                new_size, old_capacity,
                                                max_capacity);
    assert(requested_capacity >= new_size);
    auto [new_block, new_capacity] =
        BlockType::CreateWithCapacity(requested_capacity);

    KeyType* new_keys = new_block->keys();
    ValueType* new_values = new_block->values(new_capacity);

    if (old_block == nullptr || old_size == 0) {
      // Empty block case: just insert the new key-value
      new_keys[0] = key;
      new_values[0] = value;
    } else {
      const KeyType* old_keys = old_block->keys();
      const ValueType* old_values = old_block->values(old_capacity);

      // Copy elements before insertion point.
      size_t i = 0;
      for (; i < insert_pos; ++i) {
        new_keys[i] = old_keys[i];
        new_values[i] = old_values[i];
      }

      new_keys[insert_pos] = key;
      new_values[insert_pos] = value;

      // Copy elements after insertion point.
      for (i = insert_pos; i < old_size; ++i) {
        new_keys[i + 1] = old_keys[i];
        new_values[i + 1] = old_values[i];
      }
    }

    return {new_block, new_size, new_capacity};
  }

  // Create a new block with the value at update_pos replaced. The caller has
  // already established that the position exists. Keys and block size are
  // unchanged.
  // The caller publishes the new block atomically and retires the old block.
  static std::tuple<BlockType*, size_t, size_t> CreateBlockWithUpdateAt(
      BlockType* old_block, size_t old_size, size_t old_capacity,
      size_t update_pos, ValueType value) {
    assert(old_block != nullptr && update_pos < old_size);

    const KeyType* old_keys = old_block->keys();
    const ValueType* old_values = old_block->values(old_capacity);
    auto [new_block, new_capacity] =
        BlockType::CreateWithCapacity(old_capacity);
    KeyType* new_keys = new_block->keys();
    ValueType* new_values = new_block->values(new_capacity);

    for (size_t i = 0; i < old_size; ++i) {
      new_keys[i] = old_keys[i];
      new_values[i] = (i == update_pos) ? value : old_values[i];
    }

    return {new_block, old_size, new_capacity};
  }

  // Create a new block with the entry at delete_pos removed. The caller has
  // already established that the position exists. If the resulting block is
  // empty, {nullptr, 0, 0} represents that empty block.
  static std::tuple<BlockType*, size_t, size_t> CreateBlockWithDeleteAt(
      BlockType* old_block, size_t old_size, size_t old_capacity,
      size_t delete_pos) {
    assert(old_block != nullptr && delete_pos < old_size);

    const KeyType* old_keys = old_block->keys();
    const ValueType* old_values = old_block->values(old_capacity);
    size_t new_size = old_size - 1;

    // If block becomes empty, return nullptr
    if (new_size == 0) {
      return {nullptr, 0, 0};
    }

    auto [new_block, new_capacity] =
        BlockType::CreateWithCapacity(old_capacity);

    KeyType* new_keys = new_block->keys();
    ValueType* new_values = new_block->values(new_capacity);

    // Copy elements before deletion point
    size_t i = 0;
    for (; i < delete_pos; i++) {
      new_keys[i] = old_keys[i];
      new_values[i] = old_values[i];
    }

    // Copy elements after deletion point (skip the deleted key)
    for (size_t j = delete_pos + 1; j < old_size; j++, i++) {
      new_keys[i] = old_keys[j];
      new_values[i] = old_values[j];
    }

    return {new_block, new_size, new_capacity};
  }

  // Create a new block with a key replaced by multiple key-value pairs.
  // The replaced key is removed, and the new key-value pairs are inserted in its place.
  // The new key-value pairs must maintain sorted order when inserted.
  // Returns {new_block, new_size, new_capacity}.
  // The caller is responsible for retiring the old block after successful CAS.
  static std::tuple<BlockType*, size_t, size_t> CreateBlockWithReplace(
      BlockType* old_block, size_t old_size, size_t old_capacity,
      KeyType key,
      const KeyType* new_keys, const ValueType* new_values, size_t new_count) {

    assert(old_block != nullptr && old_size > 0);

    const KeyType* old_keys = old_block->keys();
    const ValueType* old_values = old_block->values(old_capacity);

    const size_t replace_pos =
        LowerBoundInInternalBlock(key, old_block, old_size);

    // Key must exist
    assert(replace_pos < old_size && old_keys[replace_pos] == key);

    // Calculate new size: old_size - 1 (remove key) + new_count (insert new pairs)
    size_t new_size = old_size - 1 + new_count;

    auto [new_block, new_capacity] = BlockType::CreateWithCapacity(new_size);

    KeyType* dest_keys = new_block->keys();
    ValueType* dest_values = new_block->values(new_capacity);

    // Copy elements before replacement point
    size_t dest_idx = 0;
    for (size_t i = 0; i < replace_pos; i++, dest_idx++) {
      dest_keys[dest_idx] = old_keys[i];
      dest_values[dest_idx] = old_values[i];
    }

    // Insert new key-value pairs
    for (size_t i = 0; i < new_count; i++, dest_idx++) {
      dest_keys[dest_idx] = new_keys[i];
      dest_values[dest_idx] = new_values[i];
    }

    // Copy elements after replacement point (skip the replaced key)
    for (size_t i = replace_pos + 1; i < old_size; i++, dest_idx++) {
      dest_keys[dest_idx] = old_keys[i];
      dest_values[dest_idx] = old_values[i];
    }

    return {new_block, new_size, new_capacity};
  }

};

}  // namespace fh_index_rw
