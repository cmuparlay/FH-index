// Benchmark for segmenting.h
// Usage: ./bench_segmenting <dataset|n> <B> <F>
// - dataset: path to binary file, or integer n for uniform random data
// - B: block size
// - F: keys per finger (finger density = 1/F)
// Thread count controlled via PARLAY_NUM_THREADS environment variable

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/single_level_segmentor.h"
#include "common_internal/utility/math.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "util/io_utility.h"

using namespace std;
using KeyType = uint64_t;
using namespace fh_index::internal::segmenting;
using fh_index::internal::Fraction;

// Benchmark result structure
struct BenchResult {
  double total_ms = 0;
  double build_maximal_ms = 0;
  double segment_selection_ms = 0;
  double segment_adjustment_ms = 0;
  double generate_fgds_ms = 0;
  double generate_finger_keys_ms = 0;
  double compute_finger_positions_ms = 0;
};

// Simple timer helper
class SimpleTimer {
 public:
  void start() { start_time_ = chrono::high_resolution_clock::now(); }

  double stop_ms() {
    auto end_time = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::microseconds>(end_time - start_time_);
    return duration.count() / 1000.0;
  }

 private:
  chrono::high_resolution_clock::time_point start_time_;
};

// Run benchmark for a single configuration
BenchResult RunBenchmark(size_t n, KeyType* keys, size_t B, size_t F) {
  BenchResult result;
  SimpleTimer timer;

  // Create segmentor with FingerSparsityConstraintDescriptor
  Fraction covering_finger_density(1, F);
  Segmentor<KeyType> segmentor(n, keys, B, covering_finger_density);

  // Total time
  timer.start();

  // Stage 1: BuildMaximalCovering
  SimpleTimer stage_timer;
  stage_timer.start();
  segmentor.BuildMaximalCovering();
  result.build_maximal_ms = stage_timer.stop_ms();

  // Stage 2: SegmentSelection
  stage_timer.start();
  segmentor.SegmentSelection();
  result.segment_selection_ms = stage_timer.stop_ms();

  // Stage 3: SegmentAdjustment
  stage_timer.start();
  segmentor.SegmentAdjustment();
  result.segment_adjustment_ms = stage_timer.stop_ms();

  // Stage 4: GenerateFGD
  stage_timer.start();
  segmentor.GenerateFGD();
  result.generate_fgds_ms = stage_timer.stop_ms();

  // Stage 5: GenerateFingerKeys
  stage_timer.start();
  segmentor.GenerateFingerKeys();
  result.generate_finger_keys_ms = stage_timer.stop_ms();

  // Stage 6: ComputeFingerPositions
  stage_timer.start();
  segmentor.ComputeFingerPositions();
  result.compute_finger_positions_ms = stage_timer.stop_ms();

  result.total_ms = timer.stop_ms();

  return result;
}

// Average multiple results
BenchResult AverageResults(const vector<BenchResult>& results) {
  BenchResult avg;
  size_t count = results.size();
  if (count == 0) return avg;

  for (const auto& r : results) {
    avg.total_ms += r.total_ms;
    avg.build_maximal_ms += r.build_maximal_ms;
    avg.segment_selection_ms += r.segment_selection_ms;
    avg.segment_adjustment_ms += r.segment_adjustment_ms;
    avg.generate_fgds_ms += r.generate_fgds_ms;
    avg.generate_finger_keys_ms += r.generate_finger_keys_ms;
    avg.compute_finger_positions_ms += r.compute_finger_positions_ms;
  }

  avg.total_ms /= count;
  avg.build_maximal_ms /= count;
  avg.segment_selection_ms /= count;
  avg.segment_adjustment_ms /= count;
  avg.generate_fgds_ms /= count;
  avg.generate_finger_keys_ms /= count;
  avg.compute_finger_positions_ms /= count;

  return avg;
}

// Check if string is a number
bool IsNumber(const string& s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (!isdigit(c)) return false;
  }
  return true;
}

// Extract dataset name from path
string ExtractDatasetName(const string& dataset) {
  if (IsNumber(dataset)) {
    return "uniform_" + dataset;
  }
  // Extract filename without path and extension
  size_t last_slash = dataset.find_last_of("/\\");
  string filename =
      (last_slash == string::npos) ? dataset : dataset.substr(last_slash + 1);
  size_t last_dot = filename.find_last_of('.');
  if (last_dot != string::npos) {
    filename = filename.substr(0, last_dot);
  }
  return filename;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " <dataset|n> <B> <F>" << endl;
    cerr << "  dataset: path to binary file, or integer n for uniform random"
         << endl;
    cerr << "  B: block size" << endl;
    cerr << "  F: keys per finger (finger density = 1/F)" << endl;
    cerr << "Thread count: set PARLAY_NUM_THREADS environment variable" << endl;
    return 1;
  }

  string dataset = argv[1];
  size_t B = stoull(argv[2]);
  size_t F = stoull(argv[3]);

  // Get thread count
  size_t num_threads = parlay::num_workers();

  // Load or generate data
  vector<KeyType> key_data;
  size_t n;

  if (IsNumber(dataset)) {
    n = stoull(dataset);
    cerr << "Generating " << n << " uniform random keys..." << endl;
    key_data = fh_index::util::UniformRandomKeys(n, 42, true);  // sorted
  } else {
    cerr << "Loading keys from " << dataset << "..." << endl;
    key_data = fh_index::util::LoadKeysFromBinary(dataset);
  }

  // Deduplicate
  key_data = fh_index::util::DeduplicateSortedKeys(key_data);
  n = key_data.size();

  cerr << "Data loaded: n=" << n << ", B=" << B << ", F=" << F
       << ", threads=" << num_threads << endl;

  // Allocate aligned array
  KeyType* keys = (KeyType*)aligned_alloc(64, sizeof(KeyType) * n);
  parlay::parallel_for(0, n, [&](size_t i) { keys[i] = key_data[i]; });

  // Warmup run
  cerr << "Warmup run..." << endl;
  RunBenchmark(n, keys, B, F);

  // Benchmark runs
  constexpr size_t NUM_RUNS = 3;
  vector<BenchResult> results;
  results.reserve(NUM_RUNS);

  for (size_t i = 0; i < NUM_RUNS; i++) {
    cerr << "Run " << (i + 1) << "/" << NUM_RUNS << "..." << endl;
    results.push_back(RunBenchmark(n, keys, B, F));
  }

  BenchResult avg = AverageResults(results);

  // Output CSV line (to stdout)
  // Format: dataset,n,B,F,threads,total_ms,build_maximal_ms,segment_selection_ms,
  //         segment_adjustment_ms,generate_fgds_ms,generate_finger_keys_ms,compute_finger_positions_ms
  string dataset_name = ExtractDatasetName(dataset);
  cout << fixed << setprecision(3);
  cout << dataset_name << "," << n << "," << B << "," << F << ","
       << num_threads << "," << avg.total_ms << "," << avg.build_maximal_ms
       << "," << avg.segment_selection_ms << "," << avg.segment_adjustment_ms
       << "," << avg.generate_fgds_ms << "," << avg.generate_finger_keys_ms
       << "," << avg.compute_finger_positions_ms << endl;

  free(keys);
  return 0;
}
