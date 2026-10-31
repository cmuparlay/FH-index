// Test suite for epoch-based memory reclamation.
//
// Tests:
// 1. Basic retire and cleanup
// 2. Epoch advancement
// 3. EpochGuard RAII behavior
// 4. Multi-threaded concurrent retire
// 5. Stress test with concurrent operations
// 6. Linearizability check
//
// Note: v2 removes custom deleters for performance. Objects are freed with
// std::free(). Only trivially destructible types are supported.

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "common_internal/memory/epoch.h"
#include "common_internal/memory/epoch_reclamation.h"
#include "parlay/parallel.h"

using namespace std;
using namespace memory;

// Test result structure
struct TestResult {
  bool passed;
  string message;

  TestResult(bool p, const string& msg = "") : passed(p), message(msg) {}
};

void PrintTestResult(const string& test_name, const TestResult& result) {
  cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " << test_name;
  if (!result.message.empty()) {
    cout << ": " << result.message;
  }
  cout << endl;
}

// ============================================================
// Test 1: Basic retire and cleanup
// ============================================================
TestResult TestBasicRetire() {
  EpochManager::Get().Reset();

  const size_t kNumObjects = 100;

  // Allocate and retire objects
  for (size_t i = 0; i < kNumObjects; i++) {
    void* ptr = std::malloc(64);
    Retire(ptr, 64);
  }

  // Check pending count before cleanup
  size_t pending = RetireManager::Get().GetTotalPendingCount();
  if (pending == 0) {
    return TestResult(false, "Expected pending objects before cleanup");
  }

  // Force cleanup
  ForceCleanupAll();

  // After cleanup, pending should be 0
  pending = RetireManager::Get().GetTotalPendingCount();
  if (pending != 0) {
    return TestResult(false, "Expected 0 pending after cleanup, got " + to_string(pending));
  }

  return TestResult(true);
}

// ============================================================
// Test 2: Epoch advancement
// ============================================================
TestResult TestEpochAdvancement() {
  EpochManager::Get().Reset();

  long initial_epoch = EpochManager::Get().GetCurrentEpoch();

  // Without any active guards, epoch should be advanceable
  bool advanced = EpochManager::Get().TryAdvanceEpoch();
  if (!advanced) {
    return TestResult(false, "Failed to advance epoch without active guards");
  }

  long new_epoch = EpochManager::Get().GetCurrentEpoch();
  if (new_epoch != initial_epoch + 1) {
    return TestResult(false, "Epoch did not advance correctly");
  }

  // A guard in the current epoch permits one advance, then prevents a second
  // advance until that operation reaches a quiescent state.
  {
    EpochGuard guard;
    advanced = EpochManager::Get().TryAdvanceEpoch();
    if (!advanced || EpochManager::Get().TryAdvanceEpoch()) {
      return TestResult(false, "Active old-epoch guard did not block advancement");
    }
  }
  if (!EpochManager::Get().TryAdvanceEpoch()) {
    return TestResult(false, "Quiescent guard still blocked advancement");
  }

  return TestResult(true);
}

// ============================================================
// Test 3: EpochGuard RAII behavior
// ============================================================
TestResult TestEpochGuard() {
  EpochManager::Get().Reset();

  auto slot = GetThreadEpochRegistration().slot();
  if (EpochManager::Get().GetAnnouncedEpoch(slot) !=
      EpochManager::kInactiveEpoch) {
    return TestResult(false, "Worker should initially be inactive");
  }

  {
    EpochGuard guard;
    if (EpochManager::Get().GetAnnouncedEpoch(slot) !=
        EpochManager::Get().GetCurrentEpoch()) {
      return TestResult(false, "Guard did not announce the current epoch");
    }
    {
      EpochGuard nested;
      if (EpochManager::Get().GetAnnouncedEpoch(slot) !=
          EpochManager::Get().GetCurrentEpoch()) {
        return TestResult(false, "Nested guard lost the announcement");
      }
    }
  }
  if (EpochManager::Get().GetAnnouncedEpoch(slot) !=
      EpochManager::kInactiveEpoch) {
    return TestResult(false, "Outermost guard did not become quiescent");
  }

  return TestResult(true);
}

// ============================================================
// Test 4: Multi-threaded concurrent retire
// ============================================================
TestResult TestConcurrentRetire() {
  EpochManager::Get().Reset();

  const size_t kNumObjects = 10000;
  const size_t kNumWorkers = parlay::num_workers();

  // Each worker retires objects in parallel
  parlay::parallel_for(0, kNumObjects, [&](size_t i) {
    void* ptr = std::malloc(64);
    Retire(ptr, 64);
  });

  // Check pending count before cleanup
  size_t pending = RetireManager::Get().GetTotalPendingCount();
  if (pending == 0) {
    return TestResult(false, "Expected pending objects before cleanup");
  }

  // Force cleanup
  ForceCleanupAll();

  // After cleanup, pending should be 0
  pending = RetireManager::Get().GetTotalPendingCount();
  if (pending != 0) {
    return TestResult(false, "Expected 0 pending after cleanup, got " + to_string(pending));
  }

  return TestResult(true, "Workers: " + to_string(kNumWorkers));
}

// ============================================================
// Test 5: Stress test with concurrent operations
// ============================================================
TestResult TestStress() {
  EpochManager::Get().Reset();

  const size_t kNumOperations = 100000;
  atomic<size_t> completed_ops{0};

  // Mix of operations: allocate, use with guard, retire
  parlay::parallel_for(0, kNumOperations, [&](size_t i) {
    // Allocate
    void* ptr = std::malloc(64);

    // Use within epoch guard
    {
      EpochGuard guard;
      // Simulate some work
      volatile int* data = static_cast<volatile int*>(ptr);
      *data = static_cast<int>(i);
      int read_back = *data;
      (void)read_back;
    }

    // Retire
    Retire(ptr, 64);
    completed_ops.fetch_add(1, memory_order_relaxed);
  });

  // Verify all operations completed
  if (completed_ops.load() != kNumOperations) {
    return TestResult(false, "Not all operations completed");
  }

  // Force cleanup
  ForceCleanupAll();

  // After cleanup, pending should be 0
  size_t pending = RetireManager::Get().GetTotalPendingCount();
  if (pending != 0) {
    return TestResult(false, "Expected 0 pending after cleanup, got " + to_string(pending));
  }

  return TestResult(true, "Completed " + to_string(kNumOperations) + " ops");
}

// ============================================================
// Test 6: No use-after-free (linearizability)
// ============================================================
TestResult TestNoUseAfterFree() {
  EpochManager::Get().Reset();

  const size_t kNumIterations = 10000;
  atomic<bool> error_detected{false};

  // Pattern: Writer retires, Reader accesses within guard
  // If linearizable, reader should never see freed memory

  atomic<void*> shared_ptr{nullptr};
  atomic<bool> stop{false};

  // Writer thread: allocates, publishes, retires
  auto writer = [&]() {
    for (size_t i = 0; i < kNumIterations && !error_detected.load(); i++) {
      // Allocate new object
      int* ptr = static_cast<int*>(std::malloc(sizeof(int)));
      *ptr = 0xDEADBEEF;

      // Publish
      void* old = shared_ptr.exchange(ptr);

      // Retire old if exists
      if (old != nullptr) {
        Retire(old, sizeof(int));
      }
    }

    // Final cleanup
    stop.store(true);
    void* final_ptr = shared_ptr.exchange(nullptr);
    if (final_ptr) {
      Retire(final_ptr, sizeof(int));
    }
  };

  // Reader thread: reads within epoch guard
  auto reader = [&]() {
    while (!stop.load() && !error_detected.load()) {
      EpochGuard guard;

      void* ptr = shared_ptr.load();
      if (ptr != nullptr) {
        int value = *static_cast<int*>(ptr);
        if (value != static_cast<int>(0xDEADBEEF)) {
          // Corrupted memory (use-after-free or uninitialized)
          error_detected.store(true);
        }
      }
    }
  };

  // Run writer and multiple readers on registered Parlay workers.
  const size_t kNumReaders = 4;
  parlay::parallel_for(0, kNumReaders + 1, [&](size_t i) {
    if (i == 0) {
      writer();
    } else {
      reader();
    }
  }, 1);

  ForceCleanupAll();

  if (error_detected.load()) {
    return TestResult(false, "Use-after-free detected");
  }

  return TestResult(true);
}

// ============================================================
// Test 7: Deferred cleanup timing
// ============================================================
TestResult TestDeferredCleanup() {
  EpochManager::Get().Reset();

  // Retire an object while a guard is active
  void* ptr = std::malloc(64);

  {
    EpochGuard guard;

    // Retire while guard is active
    Retire(ptr, 64);

    // Object should NOT be freed yet (guard is still active)
    size_t pending = RetireManager::Get().GetTotalPendingCount();
    if (pending == 0) {
      return TestResult(false, "Expected pending objects while guard is active");
    }
  }

  // After guard is released and epochs advance, cleanup should happen
  // Force multiple epoch advances
  for (int i = 0; i < 5; i++) {
    EpochManager::Get().TryAdvanceEpoch();
  }

  // Force final cleanup
  ForceCleanupAll();

  // After cleanup, pending should be 0
  size_t pending = RetireManager::Get().GetTotalPendingCount();
  if (pending != 0) {
    return TestResult(false, "Expected 0 pending after cleanup, got " + to_string(pending));
  }

  return TestResult(true);
}

// ============================================================
// Test 8: Chunked retirement and bounded reclamation
// ============================================================
TestResult TestChunkedReclamation() {
  ForceCleanupAll();
  EpochManager::Get().Reset();

  constexpr size_t kOldObjects = RetireList::kUpdateThreshold;
  {
    EpochGuard guard;
    for (size_t i = 0; i < kOldObjects; ++i) {
      Retire(std::malloc(64), 64);
    }
  }

  if (!EpochManager::Get().TryAdvanceEpoch()) {
    return TestResult(false, "Could not make retired chunks reclaimable");
  }

  const size_t before = RetireManager::Get().GetTotalPendingCount();
  constexpr size_t kNewObjects =
      kOldObjects * RetireList::kReclaimInterval /
      RetireList::kReclaimBatch;
  for (size_t i = 0; i < kNewObjects; ++i) {
    Retire(std::malloc(64), 64);
  }

  const size_t after = RetireManager::Get().GetTotalPendingCount();
  if (before != kOldObjects || after != kNewObjects) {
    ForceCleanupAll();
    return TestResult(
        false, "Unexpected pending counts: before=" + to_string(before) +
                   ", after=" + to_string(after));
  }

  ForceCleanupAll();
  return TestResult(true);
}

// ============================================================
// Main
// ============================================================
int main() {
  cout << "========================================" << endl;
  cout << "Epoch Reclamation Tests" << endl;
  cout << "========================================" << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;
  cout << endl;

  vector<pair<string, function<TestResult()>>> tests = {
      {"Basic Retire", TestBasicRetire},
      {"Epoch Advancement", TestEpochAdvancement},
      {"EpochGuard RAII", TestEpochGuard},
      {"Concurrent Retire", TestConcurrentRetire},
      {"Stress Test", TestStress},
      {"No Use-After-Free", TestNoUseAfterFree},
      {"Deferred Cleanup", TestDeferredCleanup},
      {"Chunked Reclamation", TestChunkedReclamation},
  };

  int passed = 0;
  int failed = 0;

  for (const auto& [name, test_fn] : tests) {
    try {
      TestResult result = test_fn();
      PrintTestResult(name, result);
      if (result.passed) {
        passed++;
      } else {
        failed++;
      }
    } catch (const exception& e) {
      PrintTestResult(name, TestResult(false, string("Exception: ") + e.what()));
      failed++;
    }
  }

  cout << endl;
  cout << "========================================" << endl;
  cout << "Results: " << passed << " passed, " << failed << " failed" << endl;
  cout << "========================================" << endl;

  return failed > 0 ? 1 : 0;
}
