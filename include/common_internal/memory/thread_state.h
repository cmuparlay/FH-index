#pragma once

#include <atomic>
#include <cstddef>
#include <limits>

namespace memory {

#ifndef FH_INDEX_EPOCH_UPDATE_THRESHOLD
#define FH_INDEX_EPOCH_UPDATE_THRESHOLD 4096
#endif

#ifndef FH_INDEX_RECLAIM_INTERVAL
#define FH_INDEX_RECLAIM_INTERVAL 8
#endif

#ifndef FH_INDEX_RECLAIM_BATCH
#define FH_INDEX_RECLAIM_BATCH 32
#endif

struct RetiredChunk;

// Per-thread retirement queue. Method definitions live in
// epoch_reclamation.h, after EpochManager is available.
class alignas(64) RetireList {
 public:
  static constexpr size_t kUpdateThreshold =
      FH_INDEX_EPOCH_UPDATE_THRESHOLD;
  static constexpr size_t kEntriesPerChunk = 508;
  static constexpr size_t kReclaimInterval = FH_INDEX_RECLAIM_INTERVAL;
  static constexpr size_t kReclaimBatch = FH_INDEX_RECLAIM_BATCH;

  RetireList() = default;

  void Retire(void* ptr, long retire_epoch);
  void ForceCleanup();
  size_t GetPendingCount() const { return pending_count_; }

 private:
  void AppendChunk(long epoch);
  void ReclaimSafe(long current_epoch, size_t budget);

  RetiredChunk* head_{nullptr};
  RetiredChunk* tail_{nullptr};
  size_t pending_count_{0};
  size_t count_{0};
};

static_assert(sizeof(RetireList) == 64,
              "RetireList must occupy one cache line");

// A registered thread owns one stable state. Epoch announcements and
// retirement bookkeeping use separate cache lines so readers do not touch the
// retire-list line on entry and exit.
struct alignas(64) ThreadState {
  std::atomic<long> announced_epoch{std::numeric_limits<long>::max()};
  size_t depth{0};
  RetireList retire_list;
};

static_assert(offsetof(ThreadState, retire_list) == 64,
              "RetireList must not share the epoch cache line");
static_assert(sizeof(ThreadState) == 128,
              "ThreadState must occupy exactly two cache lines");

}  // namespace memory
