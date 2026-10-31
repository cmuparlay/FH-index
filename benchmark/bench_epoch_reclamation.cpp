// Benchmark for epoch-based memory reclamation.
//
// Measures:
// 1. Retire throughput (single-threaded)
// 2. Retire throughput (parallel)
// 3. Epoch advancement overhead
// 4. Comparison with direct free
// 5. EpochGuard overhead
//
// Note: v2 uses malloc/free, no custom deleters, minimal RetiredObject structure

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common_internal/memory/epoch_reclamation.h"
#include "parlay/parallel.h"

using namespace std;
using namespace memory;

// High-resolution timer
class Timer {
 public:
  void start() { start_ = chrono::high_resolution_clock::now(); }
  double elapsed_ms() {
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(end - start_).count() / 1e6;
  }
  double elapsed_ns() {
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::nanoseconds>(end - start_).count();
  }

 private:
  chrono::high_resolution_clock::time_point start_;
};

// ============================================================
// Benchmark 1: Sequential Retire throughput
// ============================================================
void BenchSequentialRetire(size_t num_ops) {
  cout << "\n--- Sequential Retire ---" << endl;
  EpochManager::Get().Reset();

  // Pre-allocate
  vector<void*> ptrs(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    ptrs[i] = malloc(64);
  }

  Timer timer;
  timer.start();
  for (size_t i = 0; i < num_ops; i++) {
    Retire(ptrs[i], 64);
  }
  double retire_time = timer.elapsed_ms();

  cout << "Retire: " << retire_time << " ms, "
       << (num_ops / retire_time / 1000) << " Mops/sec, "
       << (retire_time / num_ops * 1e6) << " ns/op" << endl;

  // Cleanup
  timer.start();
  ForceCleanupAll();
  double cleanup_time = timer.elapsed_ms();
  cout << "Cleanup: " << cleanup_time << " ms" << endl;
}

// ============================================================
// Benchmark 2: Parallel Retire throughput
// ============================================================
void BenchParallelRetire(size_t num_ops) {
  cout << "\n--- Parallel Retire ---" << endl;
  cout << "Workers: " << parlay::num_workers() << endl;
  EpochManager::Get().Reset();

  // Pre-allocate
  vector<void*> ptrs(num_ops);
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    ptrs[i] = malloc(64);
  });

  Timer timer;
  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    Retire(ptrs[i], 64);
  });
  double retire_time = timer.elapsed_ms();

  cout << "Retire: " << retire_time << " ms, "
       << (num_ops / retire_time / 1000) << " Mops/sec, "
       << (retire_time / num_ops * 1e6) << " ns/op" << endl;

  // Cleanup
  timer.start();
  ForceCleanupAll();
  double cleanup_time = timer.elapsed_ms();
  cout << "Cleanup: " << cleanup_time << " ms" << endl;
}

// ============================================================
// Benchmark 3: Epoch advancement overhead
// ============================================================
void BenchEpochAdvancement(size_t num_ops) {
  cout << "\n--- Epoch Advancement ---" << endl;
  EpochManager::Get().Reset();

  // Without any guards (should always succeed)
  Timer timer;
  timer.start();
  size_t success_count = 0;
  for (size_t i = 0; i < num_ops; i++) {
    if (EpochManager::Get().TryAdvanceEpoch()) {
      success_count++;
    }
  }
  double advance_time = timer.elapsed_ms();

  cout << "TryAdvanceEpoch (no guards): " << advance_time << " ms, "
       << (num_ops / advance_time / 1000) << " Mops/sec, "
       << "success rate: " << (100.0 * success_count / num_ops) << "%" << endl;

  // With concurrent guards
  EpochManager::Get().Reset();
  atomic<size_t> parallel_success{0};

  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    EpochGuard guard;
    if (EpochManager::Get().TryAdvanceEpoch()) {
      parallel_success.fetch_add(1, memory_order_relaxed);
    }
  });
  double parallel_time = timer.elapsed_ms();

  cout << "TryAdvanceEpoch (with guards): " << parallel_time << " ms, "
       << (num_ops / parallel_time / 1000) << " Mops/sec, "
       << "success rate: " << (100.0 * parallel_success.load() / num_ops) << "%" << endl;
}

// ============================================================
// Benchmark 4: Comparison with direct free
// ============================================================
void BenchCompareWithDirectFree(size_t num_ops) {
  cout << "\n--- Comparison: Retire vs Direct Free ---" << endl;

  // Direct malloc + free (sequential)
  {
    vector<void*> ptrs(num_ops);
    for (size_t i = 0; i < num_ops; i++) {
      ptrs[i] = malloc(64);
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_ops; i++) {
      free(ptrs[i]);
    }
    double free_time = timer.elapsed_ms();

    cout << "Direct free (seq): " << free_time << " ms, "
         << (num_ops / free_time / 1000) << " Mops/sec" << endl;
  }

  // Direct malloc + free (parallel)
  {
    vector<void*> ptrs(num_ops);
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      ptrs[i] = malloc(64);
    });

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      free(ptrs[i]);
    });
    double free_time = timer.elapsed_ms();

    cout << "Direct free (par): " << free_time << " ms, "
         << (num_ops / free_time / 1000) << " Mops/sec" << endl;
  }

  // Retire + cleanup (sequential)
  EpochManager::Get().Reset();
  {
    vector<void*> ptrs(num_ops);
    for (size_t i = 0; i < num_ops; i++) {
      ptrs[i] = malloc(64);
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_ops; i++) {
      Retire(ptrs[i], 64);
    }
    ForceCleanupAll();
    double retire_time = timer.elapsed_ms();

    cout << "Retire+cleanup (seq): " << retire_time << " ms, "
         << (num_ops / retire_time / 1000) << " Mops/sec" << endl;
  }

  // Retire + cleanup (parallel)
  EpochManager::Get().Reset();
  {
    vector<void*> ptrs(num_ops);
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      ptrs[i] = malloc(64);
    });

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_ops, [&](size_t i) {
      Retire(ptrs[i], 64);
    });
    ForceCleanupAll();
    double retire_time = timer.elapsed_ms();

    cout << "Retire+cleanup (par): " << retire_time << " ms, "
         << (num_ops / retire_time / 1000) << " Mops/sec" << endl;
  }
}

// ============================================================
// Benchmark 5: EpochGuard overhead
// ============================================================
void BenchEpochGuardOverhead(size_t num_ops) {
  cout << "\n--- EpochGuard Overhead ---" << endl;
  EpochManager::Get().Reset();

  // Sequential guard creation/destruction
  Timer timer;
  timer.start();
  for (size_t i = 0; i < num_ops; i++) {
    EpochGuard guard;
    // Empty body - just measure guard overhead
  }
  double seq_time = timer.elapsed_ms();

  cout << "Sequential: " << seq_time << " ms, "
       << (num_ops / seq_time / 1000) << " Mops/sec, "
       << (seq_time / num_ops * 1e6) << " ns/op" << endl;

  // Parallel guard creation/destruction
  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    EpochGuard guard;
    // Empty body
  });
  double par_time = timer.elapsed_ms();

  cout << "Parallel: " << par_time << " ms, "
       << (num_ops / par_time / 1000) << " Mops/sec, "
       << (par_time / num_ops * 1e6) << " ns/op" << endl;
}

// ============================================================
// Benchmark 6: Full cycle (alloc + use + retire)
// ============================================================
void BenchFullCycle(size_t num_ops) {
  cout << "\n--- Full Cycle (alloc + use + retire) ---" << endl;
  EpochManager::Get().Reset();

  // Sequential
  Timer timer;
  timer.start();
  for (size_t i = 0; i < num_ops; i++) {
    void* ptr = malloc(64);
    {
      EpochGuard guard;
      volatile int* data = static_cast<volatile int*>(ptr);
      *data = static_cast<int>(i);
    }
    Retire(ptr, 64);
  }
  ForceCleanupAll();
  double seq_time = timer.elapsed_ms();

  cout << "Sequential: " << seq_time << " ms, "
       << (num_ops / seq_time / 1000) << " Mops/sec" << endl;

  // Parallel
  EpochManager::Get().Reset();
  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    void* ptr = malloc(64);
    {
      EpochGuard guard;
      volatile int* data = static_cast<volatile int*>(ptr);
      *data = static_cast<int>(i);
    }
    Retire(ptr, 64);
  });
  ForceCleanupAll();
  double par_time = timer.elapsed_ms();

  cout << "Parallel: " << par_time << " ms, "
       << (num_ops / par_time / 1000) << " Mops/sec" << endl;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
  size_t num_ops = 100000;
  if (argc >= 2) {
    num_ops = stoull(argv[1]);
  }

  cout << "========================================" << endl;
  cout << "Epoch Reclamation Benchmark" << endl;
  cout << "========================================" << endl;
  cout << "Num ops: " << num_ops << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  BenchSequentialRetire(num_ops);
  BenchParallelRetire(num_ops);
  BenchEpochAdvancement(num_ops);
  BenchCompareWithDirectFree(num_ops);
  BenchEpochGuardOverhead(num_ops);
  BenchFullCycle(num_ops);

  cout << "\n========================================" << endl;
  cout << "Benchmark Complete" << endl;
  cout << "========================================" << endl;

  return 0;
}
