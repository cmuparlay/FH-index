#include <iostream>
#include <fstream>
#include <random>
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "util/io_utility.h"

int main(int argc, char* argv[]) {
    if (!(argc >= 4 && argc <= 5)) {
        std::cerr << "Usage: " << argv[0] << " <filename> <numIntegers> <sorted> <?seed>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    uint64_t numIntegers = std::stoull(argv[2]);
    bool sorted = std::stoi(argv[3]);
    uint64_t seed = (argc == 5) ? std::stoull(argv[4]) : 0;

    auto keys = fh_index::util::UniformRandomKeys(numIntegers, seed, sorted);
    if (sorted) {
        parlay::parallel_for(1, keys.size(), [&](size_t i) {
            RUNTIME_ASSERT(keys[i] > keys[i - 1]);
        });
    }

    fh_index::util::StoreKeysToBinary(filename, keys);

    std::cout << "Binary file generated successfully." << std::endl;

    return 0;
}