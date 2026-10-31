#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "fh_index_rw/retrain_stats.h"

using fh_index_rw::RebuildKind;
using fh_index_rw::RetrainStats;
using fh_index_rw::RetrainStatsSnapshot;

void CheckEmpty(const RetrainStatsSnapshot& snapshot) {
  assert(snapshot.count == 0);
  assert(snapshot.total_ns == 0);
  assert(snapshot.max_ns == 0);
  for (const auto& kind : snapshot.by_kind) {
    assert(kind.count == 0);
    assert(kind.total_ns == 0);
    assert(kind.max_ns == 0);
  }
}

void TestCategoriesAndReset() {
  RetrainStats stats;
  CheckEmpty(stats.GetSnapshot());

  stats.Record(RebuildKind::kSubtree, 11);
  stats.Record(RebuildKind::kSubtree, 29);
  stats.Record(RebuildKind::kTopLevelSubtree, 17);
  stats.Record(RebuildKind::kRoot, 41);
  stats.Record(RebuildKind::kRoot, 0);

  const auto snapshot = stats.GetSnapshot();
  assert(snapshot.count == 5);
  assert(snapshot.total_ns == 98);
  assert(snapshot.max_ns == 41);
  const auto& subtree = snapshot.by_kind[static_cast<size_t>(RebuildKind::kSubtree)];
  assert(subtree.count == 2 && subtree.total_ns == 40 && subtree.max_ns == 29);
  const auto& top = snapshot.by_kind[static_cast<size_t>(RebuildKind::kTopLevelSubtree)];
  assert(top.count == 1 && top.total_ns == 17 && top.max_ns == 17);
  const auto& root = snapshot.by_kind[static_cast<size_t>(RebuildKind::kRoot)];
  assert(root.count == 2 && root.total_ns == 41 && root.max_ns == 41);

  stats.Reset();
  CheckEmpty(stats.GetSnapshot());
  stats.Record(RebuildKind::kRoot, 7);
  const auto reused = stats.GetSnapshot();
  assert(reused.count == 1 && reused.total_ns == 7 && reused.max_ns == 7);
  assert(reused.by_kind[static_cast<size_t>(RebuildKind::kRoot)].count == 1);
}

void TestConcurrentRecording() {
  RetrainStats stats;
  constexpr size_t kWorkers = 9;
  constexpr uint64_t kIterations = 5000;
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      const auto kind = static_cast<RebuildKind>(worker % 3);
      for (uint64_t i = 0; i < kIterations; ++i) {
        stats.Record(kind, worker + 1);
      }
    });
  }
  for (auto& worker : workers) worker.join();

  const auto snapshot = stats.GetSnapshot();
  assert(snapshot.count == 9 * kIterations);
  assert(snapshot.total_ns == 45 * kIterations);
  assert(snapshot.max_ns == 9);
  for (size_t kind = 0; kind < 3; ++kind) {
    assert(snapshot.by_kind[kind].count == 3 * kIterations);
    assert(snapshot.by_kind[kind].total_ns == (12 + 3 * kind) * kIterations);
    assert(snapshot.by_kind[kind].max_ns == 7 + kind);
  }
  stats.Reset();
  CheckEmpty(stats.GetSnapshot());
}

int main() {
  TestCategoriesAndReset();
  TestConcurrentRecording();
  std::cout << "Rebuild category, aggregate, reset, and concurrent recording tests passed\n";
}
