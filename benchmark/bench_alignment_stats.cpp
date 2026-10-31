#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_internal/construction/fraction.h"
#include "common_internal/construction/multi_level_segmentor.h"

using KeyType = uint64_t;
using Clock = std::chrono::steady_clock;
using fh_index::internal::Fraction;
using fh_index::internal::segmenting::LevelConfig;
using fh_index::internal::segmenting::MultiLevelSegmentor;

struct AlignmentLevelStats {
  size_t level;
  size_t fhbs_before;
  size_t fhbs_after;
  size_t fingers_before;
  size_t fingers_after;
};

struct AlignmentRunStats {
  double initialize_seconds;
  double propagation_seconds;
  double finalize_seconds;
  std::vector<AlignmentLevelStats> levels;
};

std::vector<KeyType> LoadDataset(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Failed to open " + path);
  const auto bytes = input.tellg();
  if (bytes < static_cast<std::streamoff>(sizeof(KeyType)) ||
      bytes % static_cast<std::streamoff>(sizeof(KeyType)) != 0) {
    throw std::runtime_error("Invalid uint64 dataset size: " + path);
  }

  input.seekg(0);
  KeyType first = 0;
  input.read(reinterpret_cast<char*>(&first), sizeof(first));
  const size_t raw_words = static_cast<size_t>(bytes) / sizeof(KeyType);
  const bool has_count_header = first == raw_words - 1;
  const size_t key_count = has_count_header ? static_cast<size_t>(first)
                                            : raw_words;
  if (!has_count_header) input.seekg(0);

  std::vector<KeyType> keys(key_count);
  input.read(reinterpret_cast<char*>(keys.data()),
             static_cast<std::streamsize>(key_count * sizeof(KeyType)));
  if (!input) throw std::runtime_error("Failed to read " + path);
  return keys;
}

// MultiLevelSegmentor names this class as a friend so tests and diagnostic
// benchmarks can exercise the individual construction phases without adding
// instrumentation to the production path.
class MultiLevelSegmentorTest {
 public:
  template <typename T>
  static AlignmentRunStats Measure(MultiLevelSegmentor<T>& segmentor) {
    const auto initialize_start = Clock::now();
    segmentor.InitializeLevels();
    const auto initialize_end = Clock::now();

    AlignmentRunStats stats{};
    stats.levels.resize(segmentor.levels_.size());
    for (size_t level = 0; level < segmentor.levels_.size(); ++level) {
      const auto& data = segmentor.levels_[level];
      stats.levels[level] = AlignmentLevelStats{
          level, data.size(), 0, CountFingers(data), 0};
    }

    const auto propagation_start = Clock::now();
    segmentor.AugmentLevels();
    const auto propagation_end = Clock::now();

    for (size_t level = 0; level < segmentor.levels_.size(); ++level) {
      const auto& data = segmentor.levels_[level];
      stats.levels[level].fhbs_after = data.size();
      stats.levels[level].fingers_after = CountFingers(data);
    }

    const auto finalize_start = Clock::now();
    segmentor.FinalizeLevels();
    const auto finalize_end = Clock::now();

    stats.initialize_seconds =
        std::chrono::duration<double>(initialize_end - initialize_start).count();
    stats.propagation_seconds =
        std::chrono::duration<double>(propagation_end - propagation_start).count();
    stats.finalize_seconds =
        std::chrono::duration<double>(finalize_end - finalize_start).count();
    return stats;
  }

 private:
  template <typename LevelData>
  static size_t CountFingers(const LevelData& data) {
    size_t count = 0;
    for (const auto& descriptor : data.fgd_values) {
      count += descriptor.GetAPSize() + 1;
    }
    return count;
  }
};

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <dataset> <B> <rho-numerator> <rho-denominator>\n";
    return 1;
  }

  const std::string dataset_path = argv[1];
  const size_t leaf_B = std::stoull(argv[2]);
  const uint64_t rho_numerator = std::stoull(argv[3]);
  const uint64_t rho_denominator = std::stoull(argv[4]);
  const Fraction leaf_density(rho_numerator, rho_denominator);

  auto keys = LoadDataset(dataset_path);
  if (keys.empty()) {
    std::cerr << "Dataset contains no keys\n";
    return 1;
  }

  MultiLevelSegmentor<KeyType> segmentor(
      keys.size(), keys.data(), LevelConfig(leaf_B, leaf_density),
      LevelConfig(16, Fraction(1, 1)), 0, 256,
      std::numeric_limits<KeyType>::max());
  const auto stats = MultiLevelSegmentorTest::Measure(segmentor);

  const double logical_seconds = stats.initialize_seconds +
                                 stats.propagation_seconds +
                                 stats.finalize_seconds;
  const double propagation_fraction =
      logical_seconds == 0 ? 0 : stats.propagation_seconds / logical_seconds;

  std::cout << "dataset,B,rho_numerator,rho_denominator,keys,level,"
               "fhbs_before,fhbs_after,fhbs_added,fingers_before,fingers_after,"
               "fingers_added,propagation_triggered,initialize_seconds,"
               "propagation_seconds,finalize_seconds,logical_seconds,"
               "propagation_fraction\n";
  std::cout << std::fixed << std::setprecision(9);
  for (const auto& level : stats.levels) {
    const size_t fhbs_added = level.fhbs_after - level.fhbs_before;
    const size_t fingers_added = level.fingers_after - level.fingers_before;
    std::cout << dataset_path << ',' << leaf_B << ',' << rho_numerator << ','
              << rho_denominator << ',' << keys.size() << ',' << level.level
              << ',' << level.fhbs_before << ',' << level.fhbs_after << ','
              << fhbs_added << ',' << level.fingers_before << ','
              << level.fingers_after << ',' << fingers_added << ','
              << (fhbs_added != 0 ? 1 : 0) << ',' << stats.initialize_seconds
              << ',' << stats.propagation_seconds << ','
              << stats.finalize_seconds << ',' << logical_seconds << ','
              << propagation_fraction << '\n';
  }
}
