#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace fh_index_rw {

enum class RebuildKind : uint8_t {
  kSubtree = 0,
  kTopLevelSubtree = 1,
  kRoot = 2,
  kCount = 3,
};

inline constexpr size_t kRebuildKindCount =
    static_cast<size_t>(RebuildKind::kCount);

inline constexpr const char* RebuildKindName(RebuildKind kind) {
  switch (kind) {
    case RebuildKind::kSubtree:
      return "subtree";
    case RebuildKind::kTopLevelSubtree:
      return "top_level_subtree";
    case RebuildKind::kRoot:
      return "root";
    case RebuildKind::kCount:
      break;
  }
  return "unknown";
}

struct RebuildKindStatsSnapshot {
  uint64_t count = 0;
  uint64_t total_ns = 0;
  uint64_t max_ns = 0;
};

struct RetrainStatsSnapshot {
  uint64_t count = 0;
  uint64_t total_ns = 0;
  uint64_t max_ns = 0;
  std::array<RebuildKindStatsSnapshot, kRebuildKindCount> by_kind{};
};

class RetrainStats {
 public:
  void Reset() {
    for (auto& counters : by_kind_) {
      counters.count.store(0, std::memory_order_relaxed);
      counters.total_ns.store(0, std::memory_order_relaxed);
      counters.max_ns.store(0, std::memory_order_relaxed);
    }
  }

  void Record(RebuildKind kind, uint64_t duration_ns) {
    const size_t kind_index = static_cast<size_t>(kind);
    if (kind_index >= kRebuildKindCount) std::abort();
    auto& counters = by_kind_[kind_index];

    counters.count.fetch_add(1, std::memory_order_relaxed);
    counters.total_ns.fetch_add(duration_ns, std::memory_order_relaxed);

    uint64_t current_max = counters.max_ns.load(std::memory_order_relaxed);
    while (current_max < duration_ns &&
           !counters.max_ns.compare_exchange_weak(
               current_max, duration_ns,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {}
  }

  RetrainStatsSnapshot GetSnapshot() const {
    RetrainStatsSnapshot snapshot;
    for (size_t i = 0; i < kRebuildKindCount; ++i) {
      const auto& counters = by_kind_[i];
      auto& kind_snapshot = snapshot.by_kind[i];
      kind_snapshot.count = counters.count.load(std::memory_order_relaxed);
      kind_snapshot.total_ns =
          counters.total_ns.load(std::memory_order_relaxed);
      kind_snapshot.max_ns = counters.max_ns.load(std::memory_order_relaxed);
      snapshot.count += kind_snapshot.count;
      snapshot.total_ns += kind_snapshot.total_ns;
      snapshot.max_ns = std::max(snapshot.max_ns, kind_snapshot.max_ns);
    }
    return snapshot;
  }

 private:
  struct RebuildKindCounters {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> max_ns{0};
  };

  std::array<RebuildKindCounters, kRebuildKindCount> by_kind_{};
};

}  // namespace fh_index_rw
