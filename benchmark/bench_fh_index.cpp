#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "timer.hpp"
#include "util/concurrent_timer.h"
#include "util/zipfian.h"
#include "fh_index/fh_index_ro.h"
#include "fh_index/fh_index_rw.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "util/io_utility.h"

using namespace std;

using KeyType = uint64_t;
using ValueType = uint64_t;
using EntryType = std::pair<KeyType, ValueType>;

#define PRINT_DETAIL 1

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef FINGER_SPARSITY
#define FINGER_SPARSITY 8
#endif

// FHIndexRO
constexpr size_t kBlockSize = BLOCK_SIZE;
constexpr size_t kFingerSparsity = FINGER_SPARSITY;
constexpr size_t kInternalBlockSize = 16;
constexpr size_t kInternalFingerSparsity = 2;

// Common
constexpr size_t kTopLevelThreshold = 256;

// template <typename IndexType>
// void BulkLoad(size_t n, IndexType& index, KeyType* keys, ValueType* values) {
//   cout << "BulkLoad start" << endl;
//   index.BulkLoad(n, keys, values);
//   cout << "BulkLoad done" << endl;
// }

template <typename IndexType>
void test(size_t n, IndexType& index, KeyType* keys, ValueType* values, size_t num_round,
          size_t num_queries, size_t granularity, double zipf_alpha,
          size_t scramble) {
  time_nested("Sort", [&]() {
    auto entries = parlay::tabulate(n, [&](size_t i) -> EntryType {
      return {keys[i], values[i]};
    });
    parlay::integer_sort_inplace(
        entries, [](EntryType kv) { return kv.first; });
    parlay::parallel_for(0, n, [&](size_t i) {
      keys[i] = entries[i].first;
      values[i] = entries[i].second;
    });
  });

  for (size_t round = 0; round < 1; round++) {
    cout << "BulkLoad start" << endl;
    time_nested("BulkLoad", [&] {
      index.BulkLoad(n, keys, values);
    });
    cout << "BulkLoad done" << endl;
  }

  index.Print();

  (void)granularity;

  cout << "Query count: " << num_queries << endl;

  KeyType* q_keys = new KeyType[num_queries];
  ValueType* q_values = new ValueType[num_queries];

  Zipfian zipfian(zipf_alpha, n);

#ifdef LATENCY
  size_t tot_num_batches = 0;
  double tot_round_latency_sum = 0;
#endif

  double tot_round_time_s = 0;

  for (size_t round = 0; round < num_round; round++) {
    auto random = parlay::random(round);
    parlay::parallel_for(0, num_queries, [&](size_t i) {
      double possibility = (double)random.ith_rand(i) / (double)UINT64_MAX;
      size_t idx = zipfian.GetIdx(possibility);
      if (scramble) {
        idx = parlay::hash64(idx) % n;
      }
      q_keys[i] = keys[idx];
      q_values[i] = values[idx];
    });

#ifdef LATENCY
    HighResolutionTimer latency_timer(parlay::num_workers());
#endif
    HighResolutionTimer throughput_timer(1);

    auto work = [&](size_t i) {
#ifdef LATENCY
      latency_timer.start_timing(parlay::worker_id());
#endif
      uint64_t value;
      bool found;
      index.Find(q_keys[i], value, found);
      RUNTIME_ASSERT(found && (value == q_values[i]));

#ifdef LATENCY
      latency_timer.end_timing(parlay::worker_id());
#endif
    };

    throughput_timer.start_timing(0);
    parlay::parallel_for(0, num_queries, work);
    throughput_timer.end_timing(0);

#ifdef LATENCY
    size_t num_batches = latency_timer.get_call_count();
    tot_num_batches += num_batches;
    double round_latency_sum = latency_timer.get_total_duration_ns() / 1e9;
    tot_round_latency_sum += round_latency_sum;
#endif

    double round_time_s = throughput_timer.get_total_duration_ns() / 1e9;
    tot_round_time_s += round_time_s;

#ifdef PRINT_DETAIL
    double throughput_m = num_queries / round_time_s / 1e6;
    std::cout << "Round " << round << " Time taken: " << round_time_s
              << " second" << std::endl;
    std::cout << "Mops/sec: " << throughput_m << endl;

#ifdef LATENCY
    double average_latency_ns = round_latency_sum / num_batches * 1e9;
    std::cout << "NumBatch: " << num_batches
              << " Total Time taken: " << round_latency_sum << " second"
              << std::endl;
    std::cout << "Average Latency: " << average_latency_ns << " ns"
              << std::endl;
#endif
#endif
  }

  std::cout << "Total Time taken: " << tot_round_time_s << " second"
            << std::endl;
  std::cout << "Mops/sec: " << num_round * num_queries / tot_round_time_s / 1e6
            << std::endl;

#ifdef LATENCY
  std::cout << "Total NumBatch: " << tot_num_batches << std::endl;
  std::cout << "Total Time taken: " << tot_round_latency_sum << " second"
            << std::endl;
  std::cout << "Average Latency: "
            << tot_round_latency_sum / tot_num_batches * 1e9 << " ns"
            << std::endl;
#endif

  cout << "Predict Count: " << predict_count << " Predict Sum: " << predict_sum
       << std::endl;
  cout << "Predict Count / Query: " << (double)predict_count / num_queries
       << std::endl;
  cout << "Predict Sum / Query: " << (double)predict_sum / num_queries
       << std::endl;
  cout << "Average Predict Distance" << (double)predict_sum / predict_count
       << std::endl;
      
  delete[] q_keys;
  delete[] q_values;
}

#ifdef VERSION_RW
// Benchmark Insert/Delete operations for Read-Write index
template <typename IndexType>
void test_insert_delete(size_t n, IndexType& index, KeyType* keys,
                        ValueType* values, size_t num_round, size_t num_ops,
                        double insert_ratio) {
  // Sort keys first
  time_nested("Sort", [&]() {
    auto entries = parlay::tabulate(n, [&](size_t i) -> EntryType {
      return {keys[i], values[i]};
    });
    parlay::integer_sort_inplace(entries,
                                 [](EntryType kv) { return kv.first; });
    parlay::parallel_for(0, n, [&](size_t i) {
      keys[i] = entries[i].first;
      values[i] = entries[i].second;
    });
  });

  // Split data using INTERLEAVED pattern for uniform distribution:
  // - Bulk load uses EVEN indices (0, 2, 4, ...)
  // - Insert/Delete uses ODD indices (1, 3, 5, ...)
  // This ensures insert keys are uniformly distributed across the index.
  size_t load_size = n / 2;
  size_t op_pool_size = n - load_size;

  // Allocate separate arrays for load and operation keys
  KeyType* load_keys = new KeyType[load_size];
  ValueType* load_values = new ValueType[load_size];
  KeyType* op_keys = new KeyType[op_pool_size];
  ValueType* op_values = new ValueType[op_pool_size];

  // Interleaved split: even indices for load, odd indices for ops
  parlay::parallel_for(0, load_size, [&](size_t i) {
    load_keys[i] = keys[i * 2];
    load_values[i] = values[i * 2];
  });
  parlay::parallel_for(0, op_pool_size, [&](size_t i) {
    size_t src_idx = i * 2 + 1;
    if (src_idx < n) {
      op_keys[i] = keys[src_idx];
      op_values[i] = values[src_idx];
    }
  });

  // Bulk load first half
  cout << "BulkLoad start (size: " << load_size << ")" << endl;
  time_nested("BulkLoad", [&] { index.BulkLoad(load_size, load_keys, load_values); });
  cout << "BulkLoad done" << endl;

  index.Print();

  // Prepare operation keys (sample from op_pool with optional duplicates)
  if (num_ops > op_pool_size) {
    num_ops = op_pool_size;
    cout << "Adjusted num_ops to " << num_ops << " (limited by available keys)"
         << endl;
  }

  KeyType* insert_keys = new KeyType[num_ops];
  ValueType* insert_values = new ValueType[num_ops];

  // Use keys from op_pool for insert
  parlay::parallel_for(0, num_ops, [&](size_t i) {
    insert_keys[i] = op_keys[i];
    insert_values[i] = op_values[i];
  });

  size_t num_inserts = static_cast<size_t>(num_ops * insert_ratio);
  size_t num_deletes = num_ops - num_inserts;

  cout << "=== Insert/Delete Benchmark ===" << endl;
  cout << "Num inserts: " << num_inserts << ", Num deletes: " << num_deletes
       << endl;

  // ===== INSERT BENCHMARK =====
  if (num_inserts > 0) {
    cout << "\n--- Insert Benchmark ---" << endl;

#ifdef LATENCY
    size_t tot_num_batches = 0;
    double tot_round_latency_sum = 0;
#endif
    double tot_round_time_s = 0;

    for (size_t round = 0; round < num_round; round++) {
      // Reset: delete all inserted keys before each round (except first)
      if (round > 0) {
        parlay::parallel_for(0, num_inserts,
                             [&](size_t i) { index.Delete(insert_keys[i]); });
      }

#ifdef LATENCY
      HighResolutionTimer latency_timer(parlay::num_workers());
#endif
      HighResolutionTimer throughput_timer(1);

      auto work = [&](size_t i) {
#ifdef LATENCY
        latency_timer.start_timing(parlay::worker_id());
#endif
        bool success = index.Insert(insert_keys[i], insert_values[i]);
        (void)success;  // May fail if key exists
#ifdef LATENCY
        latency_timer.end_timing(parlay::worker_id());
#endif
      };

      throughput_timer.start_timing(0);
      parlay::parallel_for(0, num_inserts, work);
      throughput_timer.end_timing(0);

#ifdef LATENCY
      size_t num_batches = latency_timer.get_call_count();
      tot_num_batches += num_batches;
      double round_latency_sum = latency_timer.get_total_duration_ns() / 1e9;
      tot_round_latency_sum += round_latency_sum;
#endif

      double round_time_s = throughput_timer.get_total_duration_ns() / 1e9;
      tot_round_time_s += round_time_s;

#ifdef PRINT_DETAIL
      double throughput_m = num_inserts / round_time_s / 1e6;
      std::cout << "Insert Round " << round << " Time: " << round_time_s
                << " s, Mops/sec: " << throughput_m << endl;
#ifdef LATENCY
      double average_latency_ns = round_latency_sum / num_batches * 1e9;
      std::cout << "  Avg Latency: " << average_latency_ns << " ns" << endl;
#endif
#endif
    }

    std::cout << "Insert Total Time: " << tot_round_time_s << " s" << endl;
    std::cout << "Insert Mops/sec: "
              << num_round * num_inserts / tot_round_time_s / 1e6 << endl;
#ifdef LATENCY
    std::cout << "Insert Avg Latency: "
              << tot_round_latency_sum / tot_num_batches * 1e9 << " ns" << endl;
#endif
  }

  // ===== DELETE BENCHMARK =====
  if (num_deletes > 0) {
    cout << "\n--- Delete Benchmark ---" << endl;

    // Use separate keys for delete (from the end of insert_keys array)
    // This allows mixed mode to have independent insert and delete operations
    KeyType* delete_keys = insert_keys + num_inserts;
    ValueType* delete_values = insert_values + num_inserts;

    // For delete benchmark, we need to pre-insert the keys first
    cout << "Pre-inserting " << num_deletes << " keys for delete benchmark..."
         << endl;
    parlay::parallel_for(0, num_deletes, [&](size_t i) {
      index.Insert(delete_keys[i], delete_values[i]);
    });

#ifdef LATENCY
    size_t tot_num_batches = 0;
    double tot_round_latency_sum = 0;
#endif
    double tot_round_time_s = 0;

    for (size_t round = 0; round < num_round; round++) {
      // Reset: re-insert deleted keys before each round (except first)
      if (round > 0) {
        parlay::parallel_for(0, num_deletes, [&](size_t i) {
          index.Insert(delete_keys[i], delete_values[i]);
        });
      }

#ifdef LATENCY
      HighResolutionTimer latency_timer(parlay::num_workers());
#endif
      HighResolutionTimer throughput_timer(1);

      auto work = [&](size_t i) {
#ifdef LATENCY
        latency_timer.start_timing(parlay::worker_id());
#endif
        bool success = index.Delete(delete_keys[i]);
        (void)success;  // May fail if key not found
#ifdef LATENCY
        latency_timer.end_timing(parlay::worker_id());
#endif
      };

      throughput_timer.start_timing(0);
      parlay::parallel_for(0, num_deletes, work);
      throughput_timer.end_timing(0);

#ifdef LATENCY
      size_t num_batches = latency_timer.get_call_count();
      tot_num_batches += num_batches;
      double round_latency_sum = latency_timer.get_total_duration_ns() / 1e9;
      tot_round_latency_sum += round_latency_sum;
#endif

      double round_time_s = throughput_timer.get_total_duration_ns() / 1e9;
      tot_round_time_s += round_time_s;

#ifdef PRINT_DETAIL
      double throughput_m = num_deletes / round_time_s / 1e6;
      std::cout << "Delete Round " << round << " Time: " << round_time_s
                << " s, Mops/sec: " << throughput_m << endl;
#ifdef LATENCY
      double average_latency_ns = round_latency_sum / num_batches * 1e9;
      std::cout << "  Avg Latency: " << average_latency_ns << " ns" << endl;
#endif
#endif
    }

    std::cout << "Delete Total Time: " << tot_round_time_s << " s" << endl;
    std::cout << "Delete Mops/sec: "
              << num_round * num_deletes / tot_round_time_s / 1e6 << endl;
#ifdef LATENCY
    std::cout << "Delete Avg Latency: "
              << tot_round_latency_sum / tot_num_batches * 1e9 << " ns" << endl;
#endif
  }

  delete[] insert_keys;
  delete[] insert_values;

  // Free interleaved split arrays
  delete[] load_keys;
  delete[] load_values;
  delete[] op_keys;
  delete[] op_values;
}
#endif  // VERSION_RW

int main(int argc, char** argv) {
  init_root_timer();
  timer::active = true;
  timer::default_detail = true;
  timer::print_when_time = true;

  string dataset;
  size_t load_limit;

  size_t num_round;
  size_t num_queries;
  size_t granularity;
  double zipf_alpha;
  size_t scramble;
  size_t seed = 0;

  // omp_set_num_threads(24);

  string bench_mode = "find";  // Default benchmark mode

  if (argc < 8) {
    cout << "Usage: " << argv[0]
         << " <dataset> <load_limit> <num_round> <num_queries> "
            "<granularity> <zipf_alpha> <scramble> [seed] [bench_mode]"
         << endl;
    cout << "  bench_mode: find (default), insert, delete" << endl;
    cout << "    - find: benchmark Find operations only" << endl;
    cout << "    - insert: benchmark Insert operations (RW only)" << endl;
    cout << "    - delete: benchmark Delete operations (RW only)" << endl;
    cout << "Version: "
#ifdef VERSION_RO
         << "RO (Read-Only)"
#elif defined(VERSION_RW)
         << "RW (Read-Write)"
#else
         << "UNKNOWN"
#endif
         << " (compile-time)" << endl;
    return 1;
  }
  dataset = argv[1];
  load_limit = std::stoull(argv[2]);
  num_round = std::stoull(argv[3]);
  num_queries = std::stoull(argv[4]);
  granularity = std::stoull(argv[5]);
  zipf_alpha = std::stod(argv[6]);
  scramble = std::stoull(argv[7]);
  if (argc >= 9) {
    seed = std::stoull(argv[8]);
  }
  if (argc >= 10) {
    bench_mode = argv[9];
  }

  std::vector<KeyType> key_data;

  if (dataset == "GEN") {
    cout << "Uniform Random Keys" << endl;
    key_data = fh_index::util::UniformRandomKeys(load_limit, seed, false);
  } else {
    cout << "Load Keys from " << dataset << endl;
    key_data = fh_index::util::LoadKeysFromBinary(dataset, load_limit, false, false, false);
  }

  size_t n = key_data.size();

  KeyType* keys = new KeyType[n];
  ValueType* values = new ValueType[n];

  parlay::parallel_for(0, n, [&](size_t i) {
    keys[i] = key_data[i];
    values[i] = parlay::hash64(keys[i]);
    // entries[i] = std::make_pair(keys[i], parlay::hash64(keys[i]));
  });

  for (size_t i = 0; i < num_round; i++) {
    cout << "Key: " << keys[i] << " Value: " << values[i] << endl;
  }

#ifdef VERSION_RO
  // Read-Only version
  cout << "Using FHIndexRO (Read-Only)" << endl;
  if (bench_mode != "find") {
    cerr << "Error: RO version only supports 'find' benchmark mode" << endl;
    return 1;
  }
  fh_index_ro::FHIndexRO<uint64_t, uint64_t> index(kTopLevelThreshold,
                                                   kBlockSize, kFingerSparsity);
  test(n, index, keys, values, num_round, num_queries, granularity, zipf_alpha,
       scramble);
#elif defined(VERSION_RW)
  // Read-Write version
  cout << "Using FHIndexRW (Read-Write)" << endl;
  cout << "Benchmark mode: " << bench_mode << endl;
  fh_index_rw::FHIndexRW<uint64_t, uint64_t> index(kTopLevelThreshold,
                                                   kBlockSize, kBlockSize * 2, kFingerSparsity,
                                                   kInternalBlockSize, kInternalBlockSize * 2, kInternalFingerSparsity);
  if (bench_mode == "find") {
    test(n, index, keys, values, num_round, num_queries, granularity, zipf_alpha,
         scramble);
  } else if (bench_mode == "insert") {
    // Insert-only benchmark: insert_ratio = 1.0
    test_insert_delete(n, index, keys, values, num_round, num_queries, 1.0);
  } else if (bench_mode == "delete") {
    // Delete-only benchmark: insert_ratio = 0.0 (all ops are deletes)
    // Note: For delete-only, we first need to insert the keys
    test_insert_delete(n, index, keys, values, num_round, num_queries, 0.0);
  } else {
    cerr << "Error: Unknown benchmark mode '" << bench_mode << "'" << endl;
    return 1;
  }
#else
  #error "Must define either VERSION_RO or VERSION_RW"
#endif

  delete[] keys;
  delete[] values;

  timer::active = false;
  print_all_timers(print_type::pt_full);
  return 0;
}
