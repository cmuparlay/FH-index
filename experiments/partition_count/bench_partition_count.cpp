/**
 * Benchmark: Compare partition counts between PGM and FH-index
 *
 * PGM: Position-based modeling (CDF approximation)
 * FH:  Finger-hinted modeling
 *
 * Under the same last-mile search span of 128:
 * - PGM uses epsilon = 64 (an error interval of width 2*epsilon)
 * - FH uses B = 128, F = 4
 *
 * Usage:
 *   ./bench_partition_count <dataset> <load_limit> <epsilon> <B> <F>
 *
 * Dataset:
 *   - "GEN": Generate uniform random keys
 *   - <path>: Load keys from binary file (SOSD format: uint64_t count + uint64_t[] keys)
 *
 * Output:
 *   Results are appended to results/result.csv
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/stat.h>

// PGM-index (use internal segmentation directly for runtime epsilon)
#include "pgm/piecewise_linear_model.hpp"

// FH-index segmenting
#include "common_internal/construction/fraction.h"
#include "common_internal/construction/single_level_segmentor.h"
#include "util/io_utility.h"

using namespace std;

// Parameters (set from command line)
size_t PGM_EPSILON = 64;
size_t FH_B = 128;
size_t FH_F = 4;

// Output file path
const string RESULTS_DIR = "results";
const string RESULTS_FILE = RESULTS_DIR + "/result.csv";

//==============================================================================
// PGM partition count
//==============================================================================

size_t GetPGMPartitionCount(const vector<uint64_t>& keys) {
    size_t n = keys.size();
    using Segment = typename pgm::internal::OptimalPiecewiseLinearModel<uint64_t, size_t>::CanonicalSegment;
    size_t count = pgm::internal::make_segmentation(
        n, PGM_EPSILON,
        [&](size_t i) { return keys[i]; },
        [](Segment) {}  // discard segments, only count
    );
    return count;
}

//==============================================================================
// FH-index partition count (FGD count)
// FH-RW style: Directly use keys array without FILTER_STRIDE
//==============================================================================

size_t GetFHRWPartitionCount(const vector<uint64_t>& keys) {
    size_t n = keys.size();
    if (n == 0) return 0;

    // The cover-and-trim construction uses provisional density rho/2.
    // The command-line parameter F denotes 1/rho, so the provisional
    // density is 1/(2F).
    auto covering_finger_density =
        fh_index::internal::Fraction(1, 2 * FH_F);
    fh_index::internal::segmenting::Segmentor<uint64_t> segmentor(
        n,
        keys.data(),
        FH_B,  // B parameter for segmentor
        covering_finger_density
    );
    segmentor.Run();

    return segmentor.fgd_size;
}

//==============================================================================
// Main benchmark
//==============================================================================

void PrintUsage(const char* prog) {
    cerr << "Usage: " << prog << " <dataset> <load_limit> <epsilon> <B> <F>" << endl;
    cerr << endl;
    cerr << "Dataset:" << endl;
    cerr << "  GEN         Generate uniform random keys" << endl;
    cerr << "  <path>      Load keys from binary file (SOSD format)" << endl;
    cerr << endl;
    cerr << "Output:" << endl;
    cerr << "  Results are appended to " << RESULTS_FILE << endl;
    cerr << endl;
    cerr << "Example:" << endl;
    cerr << "  " << prog << " GEN 200000000 64 128 4" << endl;
}

std::vector<uint64_t> LoadSosdKeys(const std::string& path, size_t limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open dataset: " + path);
    const auto bytes = input.tellg();
    if (bytes < static_cast<std::streamoff>(sizeof(uint64_t)))
        throw std::runtime_error("Missing SOSD count header: " + path);
    input.seekg(0);
    uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    const auto payload_bytes = static_cast<uint64_t>(bytes) - sizeof(uint64_t);
    if (payload_bytes % sizeof(uint64_t) != 0 ||
        count != payload_bytes / sizeof(uint64_t))
        throw std::runtime_error("SOSD header does not match file size: " + path);
    const auto loaded = static_cast<size_t>(std::min<uint64_t>(count, limit));
    std::vector<uint64_t> keys(loaded);
    input.read(reinterpret_cast<char*>(keys.data()), loaded * sizeof(uint64_t));
    if (!input) throw std::runtime_error("Cannot read dataset: " + path);
    std::sort(keys.begin(), keys.end());
    return keys;
}

int Run(int argc, char* argv[]) {
    if (argc < 6) {
        PrintUsage(argv[0]);
        return 1;
    }

    string dataset = argv[1];
    size_t load_limit = stoull(argv[2]);
    PGM_EPSILON = stoull(argv[3]);
    FH_B = stoull(argv[4]);
    FH_F = stoull(argv[5]);
    if (load_limit == 0 || PGM_EPSILON == 0 || FH_B == 0 ||
        FH_F == 0 || FH_F > numeric_limits<size_t>::max() / 2) {
        cerr << "load_limit, epsilon, B and F must be positive; 2*F must fit size_t" << endl;
        return 1;
    }

    cerr << "=== Partition Count Comparison ===" << endl;
    cerr << "Last-mile search span: 2*epsilon = " << (2 * PGM_EPSILON) << endl;
    cerr << "PGM: epsilon = " << PGM_EPSILON << endl;
    cerr << "FH:  B = " << FH_B << ", F = " << FH_F << endl;
    cerr << endl;

    // Load or generate keys
    vector<uint64_t> keys;

    if (dataset == "GEN") {
        cerr << "Generating " << load_limit << " uniform random keys..." << endl;
        keys = fh_index::util::UniformRandomKeys(load_limit, 42, true);
    } else {
        cerr << "Loading keys from " << dataset << "..." << endl;
        keys = LoadSosdKeys(dataset, load_limit);
    }

    // Remove duplicates
    keys.erase(unique(keys.begin(), keys.end()), keys.end());

    // Remove Maximum key if exists (to avoid overflow in PGM)
    if (!keys.empty() && keys.back() == numeric_limits<uint64_t>::max()) {
        keys.pop_back();
    }

    size_t n = keys.size();
    if (n == 0) throw std::runtime_error("Dataset contains no indexable keys");
    cerr << "Loaded " << n << " unique keys" << endl;

    // Get partition counts
    cerr << "Computing PGM partition count..." << endl;
    size_t pgm_count = GetPGMPartitionCount(keys);

    cerr << "Computing FH partition count..." << endl;
    size_t fh_count = GetFHRWPartitionCount(keys);

    // Print results to stderr
    cerr << endl;
    cerr << "=== Results ===" << endl;
    cerr << "Dataset: " << dataset << endl;
    cerr << "Keys: " << n << endl;
    cerr << "PGM partitions: " << pgm_count << endl;
    cerr << "FH partitions:  " << fh_count << endl;
    if (fh_count > 0) {
        cerr << "Ratio (PGM/FH): " << (double)pgm_count / fh_count << "x" << endl;
    }

    // Create results directory if not exists
    mkdir(RESULTS_DIR.c_str(), 0755);

    // Check if file exists to determine if we need to write header
    bool write_header = false;
    {
        ifstream check(RESULTS_FILE);
        write_header = !check.good();
    }

    // Write CSV output to file
    ofstream outfile(RESULTS_FILE, ios::app);
    if (!outfile) {
        cerr << "Error: Cannot open " << RESULTS_FILE << " for writing" << endl;
        return 1;
    }

    if (write_header) {
        outfile << "dataset,n,epsilon,B,F,pgm_partitions,fh_partitions" << endl;
    }
    outfile << std::quoted(dataset, '"', '"') << "," << n << "," << PGM_EPSILON << "," << FH_B << "," << FH_F << "," << pgm_count << "," << fh_count << endl;
    outfile.close();

    cerr << endl;
    cerr << "Results appended to: " << RESULTS_FILE << endl;

    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
