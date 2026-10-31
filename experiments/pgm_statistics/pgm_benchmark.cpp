/**
 * PGM-index benchmark backend for real SOSD-format uint64 datasets.
 *
 * The public interface lives in pgm_statistics.py. This binary memory-maps one
 * dataset, builds each requested PGM-index over a half-open rank range, samples
 * successful queries uniformly by rank from the indexed keys, and emits raw
 * counterbalanced timing pairs as CSV.
 */

#include "pgm/pgm_index.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset_path;
    uint64_t analysis_start{};
    uint64_t analysis_end{};
    uint64_t num_queries{};
    uint64_t runs{};
    uint64_t seed{};
    uint64_t warmup_queries{};
    std::vector<size_t> epsilons;
};

class MappedDataset {
public:
    explicit MappedDataset(const std::string &path) : path_(path) {
        const uint16_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
            throw std::runtime_error("only little-endian hosts are supported");
        }

        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open dataset: " + path);
        }

        struct stat file_stat {};
        if (fstat(fd_, &file_stat) != 0) {
            throw std::runtime_error("cannot stat dataset: " + path);
        }
        if (file_stat.st_size < static_cast<off_t>(sizeof(uint64_t))) {
            throw std::runtime_error("dataset is too small to contain a uint64 header: " + path);
        }
        file_size_ = static_cast<size_t>(file_stat.st_size);

        mapping_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("cannot mmap dataset: " + path);
        }

        std::memcpy(&count_, mapping_, sizeof(count_));
        if (count_ > (std::numeric_limits<size_t>::max() - sizeof(uint64_t)) /
                         sizeof(uint64_t)) {
            throw std::runtime_error("dataset key count overflows address space: " + path);
        }
        const size_t expected_size =
            sizeof(uint64_t) + static_cast<size_t>(count_) * sizeof(uint64_t);
        if (file_size_ != expected_size) {
            throw std::runtime_error(
                "dataset size does not match its uint64 header: " + path +
                " (expected " + std::to_string(expected_size) + ", found " +
                std::to_string(file_size_) + ")");
        }

        keys_ = reinterpret_cast<const uint64_t *>(
            static_cast<const unsigned char *>(mapping_) + sizeof(uint64_t));
    }

    MappedDataset(const MappedDataset &) = delete;
    MappedDataset &operator=(const MappedDataset &) = delete;

    ~MappedDataset() {
        if (mapping_ != nullptr) {
            munmap(mapping_, file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    uint64_t count() const { return count_; }
    const uint64_t *keys() const { return keys_; }

private:
    std::string path_;
    int fd_ = -1;
    void *mapping_ = nullptr;
    size_t file_size_ = 0;
    uint64_t count_ = 0;
    const uint64_t *keys_ = nullptr;
};

uint64_t parse_u64(const char *text, const char *name) {
    try {
        size_t consumed = 0;
        const std::string value(text);
        const uint64_t parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
}

std::vector<size_t> parse_epsilons(const std::string &text) {
    std::vector<size_t> epsilons;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string token = text.substr(begin, comma - begin);
        if (token.empty()) {
            throw std::runtime_error("invalid empty epsilon");
        }
        const uint64_t epsilon = parse_u64(token.c_str(), "epsilon");
        epsilons.push_back(static_cast<size_t>(epsilon));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (epsilons.empty()) {
        throw std::runtime_error("at least one epsilon is required");
    }
    return epsilons;
}

Options parse_options(int argc, char **argv) {
    if (argc != 9) {
        throw std::runtime_error(
            std::string("usage: ") + argv[0] +
            " DATASET START END NUM_QUERIES RUNS SEED WARMUP_QUERIES "
            "EPSILON_CSV");
    }
    Options options;
    options.dataset_path = argv[1];
    options.analysis_start = parse_u64(argv[2], "analysis start");
    options.analysis_end = parse_u64(argv[3], "analysis end");
    options.num_queries = parse_u64(argv[4], "number of queries");
    options.runs = parse_u64(argv[5], "runs");
    options.seed = parse_u64(argv[6], "seed");
    options.warmup_queries = parse_u64(argv[7], "warmup queries");
    options.epsilons = parse_epsilons(argv[8]);
    if (options.analysis_start >= options.analysis_end) {
        throw std::runtime_error("analysis range must satisfy START < END");
    }
    if (options.num_queries == 0) {
        throw std::runtime_error("number of queries must be positive");
    }
    if (options.runs == 0) {
        throw std::runtime_error("runs must be positive");
    }
    return options;
}

template<typename Index>
uint64_t find_partition_batch(
    const Index &index,
    const std::vector<uint64_t> &queries,
    size_t count
) {
    uint64_t accumulator = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto range = index.search(queries[i]);
        accumulator += static_cast<uint64_t>(range.pos);
    }
    return accumulator;
}

template<typename Index>
uint64_t total_lookup_batch(
    const Index &index,
    const uint64_t *keys,
    const std::vector<uint64_t> &queries,
    size_t count
) {
    uint64_t accumulator = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto range = index.search(queries[i]);
        const auto lo = keys + range.lo;
        const auto hi = keys + range.hi;
        const auto it = std::lower_bound(lo, hi, queries[i]);
        accumulator += it == hi ? static_cast<uint64_t>(range.hi) : *it;
    }
    return accumulator;
}

template<typename Function>
double measure_ns_per_query(Function &&function, size_t count, volatile uint64_t &sink) {
    const auto start = Clock::now();
    sink ^= function();
    const auto end = Clock::now();
    const double elapsed_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    return elapsed_ns / static_cast<double>(count);
}

template<size_t Epsilon>
void run_for_epsilon(
    const uint64_t *keys,
    size_t indexed_keys,
    const std::vector<uint64_t> &queries,
    size_t runs,
    size_t warmup_queries,
    std::ostream &out
) {
    using Index = pgm::PGMIndex<uint64_t, Epsilon>;
    std::cerr << "[epsilon=" << Epsilon << "] building index over "
              << indexed_keys << " keys\n";
    Index index(keys, keys + indexed_keys);

    const size_t verification_queries = std::min<size_t>(queries.size(), 10'000);
    for (size_t i = 0; i < verification_queries; ++i) {
        const auto range = index.search(queries[i]);
        const auto hi = keys + range.hi;
        const auto it = std::lower_bound(keys + range.lo, hi, queries[i]);
        if (it == hi || *it != queries[i]) {
            throw std::runtime_error(
                "PGM search range failed to contain a sampled indexed key");
        }
    }

    volatile uint64_t sink = 0;
    const size_t warmup_count = std::min(warmup_queries, queries.size());
    if (warmup_count > 0) {
        // Warm both paths in both orders: partition -> total -> total -> partition.
        sink ^= find_partition_batch(index, queries, warmup_count);
        sink ^= total_lookup_batch(index, keys, queries, warmup_count);
        sink ^= total_lookup_batch(index, keys, queries, warmup_count);
        sink ^= find_partition_batch(index, queries, warmup_count);
    }

    std::cerr << "[epsilon=" << Epsilon << "] segments="
              << index.segments_count() << " height=" << index.height()
              << " memory=" << index.size_in_bytes() << "B\n";

    for (size_t group = 1; group <= runs; ++group) {
        // Forward order: partition routing first, then total lookup.
        const double forward_partition_ns = measure_ns_per_query(
            [&] { return find_partition_batch(index, queries, queries.size()); },
            queries.size(), sink);
        const double forward_total_ns = measure_ns_per_query(
            [&] { return total_lookup_batch(index, keys, queries, queries.size()); },
            queries.size(), sink);

        // Reverse order: total lookup first, then partition routing.
        const double reverse_total_ns = measure_ns_per_query(
            [&] { return total_lookup_batch(index, keys, queries, queries.size()); },
            queries.size(), sink);
        const double reverse_partition_ns = measure_ns_per_query(
            [&] { return find_partition_batch(index, queries, queries.size()); },
            queries.size(), sink);

        out << Epsilon << ',' << group << ','
            << std::fixed << std::setprecision(9)
            << forward_partition_ns << ',' << forward_total_ns << ','
            << reverse_total_ns << ',' << reverse_partition_ns << ','
            << index.segments_count() << ',' << index.height() << ','
            << index.size_in_bytes() << '\n';
        out.flush();

        std::cerr << "[epsilon=" << Epsilon << "][group=" << group
                  << "] forward=(partition " << forward_partition_ns
                  << " ns, total " << forward_total_ns
                  << " ns), reverse=(total " << reverse_total_ns
                  << " ns, partition " << reverse_partition_ns << " ns)\n";
    }
    std::cerr << "[epsilon=" << Epsilon << "] sink=" << sink << "\n";
}

void dispatch_epsilon(
    size_t epsilon,
    const uint64_t *keys,
    size_t indexed_keys,
    const std::vector<uint64_t> &queries,
    size_t runs,
    size_t warmup_queries,
    std::ostream &out
) {
#define RUN_EPSILON(value)                                                        \
    case value:                                                                  \
        run_for_epsilon<value>(                                                   \
            keys, indexed_keys, queries, runs, warmup_queries, out);              \
        return
    switch (epsilon) {
        RUN_EPSILON(2);
        RUN_EPSILON(4);
        RUN_EPSILON(8);
        RUN_EPSILON(16);
        RUN_EPSILON(32);
        RUN_EPSILON(64);
        RUN_EPSILON(128);
        RUN_EPSILON(256);
        RUN_EPSILON(512);
        default:
            throw std::runtime_error("unsupported epsilon: " + std::to_string(epsilon));
    }
#undef RUN_EPSILON
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        MappedDataset dataset(options.dataset_path);
        if (options.analysis_end > dataset.count()) {
            throw std::runtime_error(
                "analysis end " + std::to_string(options.analysis_end) +
                " exceeds dataset key count " + std::to_string(dataset.count()));
        }

        const uint64_t indexed_keys_u64 =
            options.analysis_end - options.analysis_start;
        if (indexed_keys_u64 > std::numeric_limits<size_t>::max() ||
            options.num_queries > std::numeric_limits<size_t>::max() ||
            options.runs > std::numeric_limits<size_t>::max() ||
            options.warmup_queries > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("requested size exceeds this host's address space");
        }
        const size_t indexed_keys = static_cast<size_t>(indexed_keys_u64);
        const size_t num_queries = static_cast<size_t>(options.num_queries);
        const uint64_t *keys = dataset.keys() + options.analysis_start;

        // PGM-index reserves numeric_limits<K>::max() as an internal sentinel.
        // Its upstream tests remove this value before construction. Preserve the
        // requested logical range, but exclude any trailing reserved values from
        // the PGM-modeled prefix and from successful-query sampling.
        size_t pgm_indexed_keys = indexed_keys;
        while (pgm_indexed_keys > 0 &&
               keys[pgm_indexed_keys - 1] == std::numeric_limits<uint64_t>::max()) {
            --pgm_indexed_keys;
        }
        const size_t excluded_sentinel_keys = indexed_keys - pgm_indexed_keys;
        if (pgm_indexed_keys == 0) {
            throw std::runtime_error(
                "analysis range contains only PGM's reserved UINT64_MAX sentinel");
        }

        std::cerr << "dataset=" << options.dataset_path
                  << " source_keys=" << dataset.count()
                  << " analysis=[" << options.analysis_start << ','
                  << options.analysis_end << ") indexed_keys=" << indexed_keys
                  << " pgm_indexed_keys=" << pgm_indexed_keys
                  << " excluded_sentinel_keys=" << excluded_sentinel_keys
                  << " queries=" << num_queries << " runs=" << options.runs
                  << " seed=" << options.seed << '\n';
        std::cerr << "sampling successful queries uniformly by rank from "
                     "PGM-indexed keys\n";

        std::mt19937_64 generator(options.seed);
        std::uniform_int_distribution<uint64_t> rank_distribution(
            0, static_cast<uint64_t>(pgm_indexed_keys - 1));
        std::vector<uint64_t> queries(num_queries);
        for (auto &query : queries) {
            query = keys[rank_distribution(generator)];
        }

        std::cout
            << "epsilon,group,forward_partition_ns,forward_total_ns,"
               "reverse_total_ns,reverse_partition_ns,num_segments,height,"
               "memory_bytes\n";
        for (const size_t epsilon : options.epsilons) {
            dispatch_epsilon(
                epsilon,
                keys,
                pgm_indexed_keys,
                queries,
                static_cast<size_t>(options.runs),
                static_cast<size_t>(options.warmup_queries),
                std::cout);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
