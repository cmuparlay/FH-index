// Benchmark for segmenting.h effect (compression ratio)
// Usage: ./bench_segmenting_effect <dataset|n> <B> <F>
// - dataset: path to binary file, or integer n for uniform random data
// - B: block size
// - F: keys per finger (finger density = 1/F)
// Recursively calls segmenting until fgd_size == 1, outputs stats per level.

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

// Level statistics
struct LevelStats {
  size_t level;
  size_t input_n;
  size_t fgd_size;
  size_t finger_size;
  double compression_ratio;
};

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
  size_t last_slash = dataset.find_last_of("/\\");
  string filename =
      (last_slash == string::npos) ? dataset : dataset.substr(last_slash + 1);
  size_t last_dot = filename.find_last_of('.');
  if (last_dot != string::npos) {
    filename = filename.substr(0, last_dot);
  }
  return filename;
}

// Run segmenting on given keys, return stats and next level keys
pair<LevelStats, vector<KeyType>> RunSegmentingLevel(
    size_t level, size_t n, KeyType* keys, size_t B, size_t F) {
  LevelStats stats;
  stats.level = level;
  stats.input_n = n;

  // Create and run segmentor
  Fraction covering_finger_density(1, F);
  Segmentor<KeyType> segmentor(n, keys, B, covering_finger_density);
  segmentor.Run();

  stats.fgd_size = segmentor.fgd_size;
  stats.finger_size = segmentor.finger_keys.size();
  stats.compression_ratio = static_cast<double>(stats.fgd_size) / n;

  // Extract fgd_keys for next level
  vector<KeyType> next_keys;
  next_keys.reserve(stats.fgd_size + 1);  // +1 for sentinel
  for (size_t i = 0; i < stats.fgd_size; i++) {
    next_keys.push_back(segmentor.fgd_keys[i]);
  }
  // Add sentinel
  next_keys.push_back(numeric_limits<KeyType>::max());

  return {stats, next_keys};
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " <dataset|n> <B> <F>" << endl;
    cerr << "  dataset: path to binary file, or integer n for uniform random"
         << endl;
    cerr << "  B: block size" << endl;
    cerr << "  F: keys per finger (finger density = 1/F)" << endl;
    return 1;
  }

  string dataset = argv[1];
  size_t B = stoull(argv[2]);
  size_t F = stoull(argv[3]);

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

  // Add sentinel
  key_data.push_back(numeric_limits<KeyType>::max());
  n = key_data.size();

  string dataset_name = ExtractDatasetName(dataset);

  cerr << "Data loaded: n=" << n << ", B=" << B << ", F=" << F << endl;

  // Output CSV header
  cout << "dataset,B,F,level,input_n,fgd_size,finger_size,compression_ratio"
       << endl;

  // Allocate aligned array for first level
  KeyType* keys = (KeyType*)aligned_alloc(64, sizeof(KeyType) * n);
  parlay::parallel_for(0, n, [&](size_t i) { keys[i] = key_data[i]; });

  vector<LevelStats> all_stats;
  size_t level = 0;

  while (n > 1) {
    cerr << "Level " << level << ": input_n=" << n << endl;

    auto [stats, next_keys] = RunSegmentingLevel(level, n, keys, B, F);
    all_stats.push_back(stats);

    // Output CSV line
    cout << fixed << setprecision(6);
    cout << dataset_name << "," << B << "," << F << "," << stats.level << ","
         << stats.input_n << "," << stats.fgd_size << "," << stats.finger_size
         << "," << stats.compression_ratio << endl;

    cerr << "  fgd_size=" << stats.fgd_size
         << ", finger_size=" << stats.finger_size
         << ", compression=" << stats.compression_ratio << endl;

    // Check termination
    if (stats.fgd_size == 1) {
      break;
    }

    // Prepare next level
    free(keys);
    n = next_keys.size();  // includes sentinel
    keys = (KeyType*)aligned_alloc(64, sizeof(KeyType) * n);
    parlay::parallel_for(0, n, [&](size_t i) { keys[i] = next_keys[i]; });

    level++;
  }

  free(keys);

  cerr << "Done. Total levels: " << all_stats.size() << endl;
  return 0;
}
