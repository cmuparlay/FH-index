#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common_internal/memory/epoch.h"
#include "common_internal/memory/memory_pool.h"

namespace memory {

static_assert(RetireList::kUpdateThreshold > 0,
              "Epoch update threshold must be positive");
static_assert((RetireList::kReclaimInterval &
               (RetireList::kReclaimInterval - 1)) == 0,
              "Reclaim interval must be a power of two");

struct RetiredChunk {
  RetiredChunk* next;
  long retire_epoch;
  size_t count;
  size_t reclaim_index;
  void* entries[RetireList::kEntriesPerChunk];
};

static_assert(sizeof(RetiredChunk) == 4096,
              "RetiredChunk should occupy exactly 4 KiB");

inline void RetireList::Retire(void* ptr, long retire_epoch) {
  if (tail_ == nullptr || tail_->retire_epoch != retire_epoch ||
      tail_->count == kEntriesPerChunk) {
    AppendChunk(retire_epoch);
  }
  tail_->entries[tail_->count++] = ptr;
  ++pending_count_;

#ifdef FH_INDEX_RECORD_RETIREMENTS_ONLY
  return;
#endif

  if (++count_ >= kUpdateThreshold) {
    count_ = 0;
    EpochManager::Get().TryAdvanceEpoch();
  }

  // Reclaim in small bounded batches. Processing 32 objects every eight
  // retirements preserves the drain rate while avoiding allocator work on
  // every mutation.
  if ((count_ & (kReclaimInterval - 1)) == 0) {
    ReclaimSafe(EpochManager::Get().GetCurrentEpoch(), kReclaimBatch);
  }
}

inline void RetireList::ForceCleanup() {
  while (head_ != nullptr) {
    RetiredChunk* next = head_->next;
    for (size_t i = head_->reclaim_index; i < head_->count; ++i) {
      Free(head_->entries[i], 0);
    }
    Free(head_, sizeof(RetiredChunk));
    head_ = next;
  }
  tail_ = nullptr;
  pending_count_ = 0;
  count_ = 0;
}

inline void RetireList::AppendChunk(long epoch) {
  auto* chunk = static_cast<RetiredChunk*>(Allocate(sizeof(RetiredChunk)));
  assert(chunk != nullptr);
  chunk->next = nullptr;
  chunk->retire_epoch = epoch;
  chunk->count = 0;
  chunk->reclaim_index = 0;
  if (tail_ == nullptr) {
    head_ = tail_ = chunk;
  } else {
    tail_->next = chunk;
    tail_ = chunk;
  }
}

inline void RetireList::ReclaimSafe(long current_epoch, size_t budget) {
  while (head_ != nullptr && budget != 0 && current_epoch >= 2 &&
         head_->retire_epoch <= current_epoch - 2) {
    while (head_->reclaim_index < head_->count && budget != 0) {
      void* entry = head_->entries[head_->reclaim_index++];
      Free(entry, 0);
      --pending_count_;
      --budget;
    }
    if (head_->reclaim_index != head_->count) return;

    RetiredChunk* consumed = head_;
    head_ = head_->next;
    if (head_ == nullptr) tail_ = nullptr;
    Free(consumed, sizeof(RetiredChunk));
  }
}

class RetireManager {
 public:
  static RetireManager& Get() {
    static RetireManager instance;
    return instance;
  }

  void Retire(void* ptr, size_t size) {
    (void)size;
    auto& registration = GetThreadEpochRegistration();
    ThreadState* state = registration.slot();
    state->retire_list.Retire(
        ptr, registration.manager()->GetRetireEpoch(state));
  }

  void ForceCleanup() {
    auto& manager = EpochManager::Get();
    for (size_t i = 0; i < manager.GetNumWorkers(); ++i) {
      manager.GetThreadState(i).retire_list.ForceCleanup();
    }
  }

  size_t GetTotalPendingCount() const {
    auto& manager = EpochManager::Get();
    size_t total = 0;
    for (size_t i = 0; i < manager.GetNumWorkers(); ++i) {
      total += manager.GetThreadState(i).retire_list.GetPendingCount();
    }
    return total;
  }
};

// Retire a raw allocation. The size remains in this interface for callers that
// also use size-aware destruction; reclamation stores only the pointer because
// the current allocator backend ultimately uses std::free.
inline void Retire(void* ptr, size_t size) {
#if defined(FH_INDEX_DISABLE_RECLAMATION) || \
    defined(FH_INDEX_DISABLE_RETIREMENT)
  (void)ptr;
  (void)size;
#else
  RetireManager::Get().Retire(ptr, size);
#endif
}

template <typename T>
inline void Retire(T* ptr, size_t size) {
  static_assert(std::is_trivially_destructible_v<T>,
                "Retire<T> only supports trivially destructible types. "
                "For types with destructors, call destructor manually before Retire.");
#if defined(FH_INDEX_DISABLE_RECLAMATION) || \
    defined(FH_INDEX_DISABLE_RETIREMENT)
  (void)ptr;
  (void)size;
#else
  RetireManager::Get().Retire(static_cast<void*>(ptr), size);
#endif
}

inline void ForceCleanupAll() {
#if !defined(FH_INDEX_DISABLE_RECLAMATION) && \
    !defined(FH_INDEX_DISABLE_RETIREMENT)
  RetireManager::Get().ForceCleanup();
#endif
}

}  // namespace memory
