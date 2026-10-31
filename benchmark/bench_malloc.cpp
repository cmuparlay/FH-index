// Benchmark for parallel memory allocation performance
// Tests jemalloc performance under various allocation patterns

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

using namespace std;

// Allocation size classes (similar to block sizes in FH Index)
constexpr size_t kSizeClasses[] = {32, 64, 128, 256, 512, 1024, 2048};
constexpr size_t kNumSizeClasses = sizeof(kSizeClasses) / sizeof(kSizeClasses[0]);

// Timer helper
class Timer {
 public:
  void start() { start_ = chrono::high_resolution_clock::now(); }
  double elapsed_seconds() {
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start_).count();
  }

 private:
  chrono::high_resolution_clock::time_point start_;
};

// Test 1: Sequential allocation and free
void bench_sequential_alloc_free(size_t alloc_size, size_t num_allocs) {
  cout << "\n=== Sequential Alloc/Free ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes, Num allocs: " << num_allocs << endl;

  Timer timer;
  vector<void*> ptrs(num_allocs);

  // Allocation phase
  timer.start();
  for (size_t i = 0; i < num_allocs; i++) {
    ptrs[i] = ::operator new(alloc_size);
  }
  double alloc_time = timer.elapsed_seconds();

  // Free phase
  timer.start();
  for (size_t i = 0; i < num_allocs; i++) {
    ::operator delete(ptrs[i]);
  }
  double free_time = timer.elapsed_seconds();

  double alloc_mops = num_allocs / alloc_time / 1e6;
  double free_mops = num_allocs / free_time / 1e6;
  cout << "Alloc: " << alloc_time << " s, " << alloc_mops << " Mops/sec" << endl;
  cout << "Free:  " << free_time << " s, " << free_mops << " Mops/sec" << endl;
}

// Test 2: Parallel allocation, sequential free
void bench_parallel_alloc_seq_free(size_t alloc_size, size_t num_allocs) {
  cout << "\n=== Parallel Alloc, Sequential Free ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes, Num allocs: " << num_allocs << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  Timer timer;
  vector<void*> ptrs(num_allocs);

  // Parallel allocation
  timer.start();
  parlay::parallel_for(0, num_allocs, [&](size_t i) {
    ptrs[i] = ::operator new(alloc_size);
  });
  double alloc_time = timer.elapsed_seconds();

  // Sequential free
  timer.start();
  for (size_t i = 0; i < num_allocs; i++) {
    ::operator delete(ptrs[i]);
  }
  double free_time = timer.elapsed_seconds();

  double alloc_mops = num_allocs / alloc_time / 1e6;
  double free_mops = num_allocs / free_time / 1e6;
  cout << "Alloc: " << alloc_time << " s, " << alloc_mops << " Mops/sec" << endl;
  cout << "Free:  " << free_time << " s, " << free_mops << " Mops/sec" << endl;
}

// Test 3: Parallel allocation and free
void bench_parallel_alloc_free(size_t alloc_size, size_t num_allocs) {
  cout << "\n=== Parallel Alloc and Free ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes, Num allocs: " << num_allocs << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  Timer timer;
  vector<void*> ptrs(num_allocs);

  // Parallel allocation
  timer.start();
  parlay::parallel_for(0, num_allocs, [&](size_t i) {
    ptrs[i] = ::operator new(alloc_size);
  });
  double alloc_time = timer.elapsed_seconds();

  // Parallel free
  timer.start();
  parlay::parallel_for(0, num_allocs, [&](size_t i) {
    ::operator delete(ptrs[i]);
  });
  double free_time = timer.elapsed_seconds();

  double alloc_mops = num_allocs / alloc_time / 1e6;
  double free_mops = num_allocs / free_time / 1e6;
  cout << "Alloc: " << alloc_time << " s, " << alloc_mops << " Mops/sec" << endl;
  cout << "Free:  " << free_time << " s, " << free_mops << " Mops/sec" << endl;
}

// Test 4: Interleaved alloc-free (simulates insert pattern)
// Each operation: allocate new block, copy data, free old block
void bench_interleaved_alloc_free(size_t alloc_size, size_t num_ops) {
  cout << "\n=== Interleaved Alloc-Free (Insert Pattern) ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes, Num ops: " << num_ops << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  // Pre-allocate initial blocks
  vector<void*> current_blocks(num_ops);
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    current_blocks[i] = ::operator new(alloc_size);
    memset(current_blocks[i], 0, alloc_size);  // Initialize
  });

  Timer timer;

  // Simulate insert: alloc new -> copy -> free old
  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    // Allocate new block (slightly larger)
    void* new_block = ::operator new(alloc_size + 16);

    // Copy data from old to new (simulate insert copy)
    memcpy(new_block, current_blocks[i], alloc_size);

    // Free old block
    ::operator delete(current_blocks[i]);

    current_blocks[i] = new_block;
  });
  double op_time = timer.elapsed_seconds();

  double mops = num_ops / op_time / 1e6;
  cout << "Time: " << op_time << " s, " << mops << " Mops/sec" << endl;
  cout << "Per-op latency: " << (op_time / num_ops * 1e9) << " ns" << endl;

  // Cleanup
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    ::operator delete(current_blocks[i]);
  });
}

// Test 5: Interleaved alloc-free WITHOUT free (leak old blocks like current Insert)
void bench_interleaved_alloc_only(size_t alloc_size, size_t num_ops) {
  cout << "\n=== Interleaved Alloc-Only (Leak Pattern) ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes, Num ops: " << num_ops << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  // Pre-allocate initial blocks
  vector<void*> current_blocks(num_ops);
  vector<void*> leaked_blocks(num_ops);  // Track leaked blocks for cleanup
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    current_blocks[i] = ::operator new(alloc_size);
    memset(current_blocks[i], 0, alloc_size);
  });

  Timer timer;

  // Simulate insert: alloc new -> copy -> DON'T free old (leak)
  timer.start();
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    // Allocate new block
    void* new_block = ::operator new(alloc_size + 16);

    // Copy data from old to new
    memcpy(new_block, current_blocks[i], alloc_size);

    // DON'T free old block - leak it
    leaked_blocks[i] = current_blocks[i];
    current_blocks[i] = new_block;
  });
  double op_time = timer.elapsed_seconds();

  double mops = num_ops / op_time / 1e6;
  cout << "Time: " << op_time << " s, " << mops << " Mops/sec" << endl;
  cout << "Per-op latency: " << (op_time / num_ops * 1e9) << " ns" << endl;

  // Cleanup both leaked and current blocks
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    ::operator delete(current_blocks[i]);
    ::operator delete(leaked_blocks[i]);
  });
}

// Test 6: Various size classes (like FH Index block sizes)
void bench_size_classes(size_t num_allocs_per_class) {
  cout << "\n=== Size Class Allocation Test ===" << endl;
  cout << "Num allocs per class: " << num_allocs_per_class << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  for (size_t size : kSizeClasses) {
    vector<void*> ptrs(num_allocs_per_class);

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_allocs_per_class, [&](size_t i) {
      ptrs[i] = ::operator new(size);
    });
    double alloc_time = timer.elapsed_seconds();

    timer.start();
    parlay::parallel_for(0, num_allocs_per_class, [&](size_t i) {
      ::operator delete(ptrs[i]);
    });
    double free_time = timer.elapsed_seconds();

    double alloc_mops = num_allocs_per_class / alloc_time / 1e6;
    cout << "Size " << size << " bytes: "
         << alloc_mops << " Mops/sec (alloc), "
         << (alloc_time / num_allocs_per_class * 1e9) << " ns/alloc" << endl;
  }
}

// Test 7: Contention test - all threads allocate same size
void bench_contention(size_t alloc_size, size_t num_allocs, size_t num_rounds) {
  cout << "\n=== Contention Test ===" << endl;
  cout << "Alloc size: " << alloc_size << " bytes" << endl;
  cout << "Num allocs per round: " << num_allocs << endl;
  cout << "Num rounds: " << num_rounds << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;

  double total_time = 0;
  for (size_t round = 0; round < num_rounds; round++) {
    vector<void*> ptrs(num_allocs);

    Timer timer;
    timer.start();
    parlay::parallel_for(0, num_allocs, [&](size_t i) {
      ptrs[i] = ::operator new(alloc_size);
      // Touch memory to ensure it's actually allocated
      *static_cast<char*>(ptrs[i]) = static_cast<char>(i);
    });
    double alloc_time = timer.elapsed_seconds();
    total_time += alloc_time;

    // Free
    parlay::parallel_for(0, num_allocs, [&](size_t i) {
      ::operator delete(ptrs[i]);
    });
  }

  double avg_time = total_time / num_rounds;
  double mops = num_allocs / avg_time / 1e6;
  cout << "Avg time per round: " << avg_time << " s" << endl;
  cout << "Throughput: " << mops << " Mops/sec" << endl;
  cout << "Per-alloc latency: " << (avg_time / num_allocs * 1e9) << " ns" << endl;
}

void print_usage(const char* prog) {
  cout << "Usage: " << prog << " <num_allocs> [alloc_size]" << endl;
  cout << "  num_allocs: Number of allocations per test (default: 100000)" << endl;
  cout << "  alloc_size: Size of each allocation in bytes (default: 128)" << endl;
}

int main(int argc, char** argv) {
  size_t num_allocs = 100000;
  size_t alloc_size = 128;

  if (argc >= 2) {
    num_allocs = stoull(argv[1]);
  }
  if (argc >= 3) {
    alloc_size = stoull(argv[2]);
  }

  cout << "========================================" << endl;
  cout << "Memory Allocation Benchmark" << endl;
  cout << "========================================" << endl;
  cout << "Num workers: " << parlay::num_workers() << endl;
  cout << "Default alloc size: " << alloc_size << " bytes" << endl;
  cout << "Num allocations: " << num_allocs << endl;

#ifdef __APPLE__
  cout << "Platform: macOS" << endl;
#else
  cout << "Platform: Linux" << endl;
#endif

  // Check if jemalloc is being used
#ifdef FH_INDEX_USE_JEMALLOC
  cout << "Allocator: jemalloc" << endl;
#else
  cout << "Allocator: system default (probably glibc malloc)" << endl;
#endif

  // Run benchmarks
  bench_sequential_alloc_free(alloc_size, num_allocs);
  bench_parallel_alloc_seq_free(alloc_size, num_allocs);
  bench_parallel_alloc_free(alloc_size, num_allocs);
  bench_interleaved_alloc_free(alloc_size, num_allocs);
  bench_interleaved_alloc_only(alloc_size, num_allocs);
  bench_size_classes(num_allocs);
  bench_contention(alloc_size, num_allocs, 5);

  cout << "\n========================================" << endl;
  cout << "Benchmark Complete" << endl;
  cout << "========================================" << endl;

  return 0;
}
