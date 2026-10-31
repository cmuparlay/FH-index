#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>
#include <climits>
#include "util/io_utility.h"

using namespace std;

std::vector<uint64_t> load_binary_data(
    const std::string& file_path,
    const uint64_t size_limit = std::numeric_limits<uint64_t>::max()) {
    std::vector<uint64_t> keys =
        fh_index::util::LoadKeysFromBinary(file_path, size_limit,
                                          false, false, false);
    
    std::sort(keys.begin(), keys.end());

    return keys;
}

std::vector<uint64_t> deduplicateSortedVector(
    const std::vector<uint64_t>& data) {
    std::vector<uint64_t> result;
    if (!data.empty()) {
        result.push_back(data[0]);
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] != data[i - 1]) {
                result.push_back(data[i]);
            }
        }
    }
    return result;
}

uint64_t average_distance(const std::vector<uint64_t>& numbers) {
    if (numbers.size() < 2) return 0;
    uint64_t sum = 0;
    for (size_t i = 1; i < numbers.size(); ++i) {
        sum += numbers[i] - numbers[i - 1];
        // if (i % (numbers.size() / 100) == 0) {
        //     cout << i << ' ' << numbers[i] << endl;
        // }
    }
    return sum / (numbers.size() - 1);
}

uint64_t calculateMinimumDistance(const std::vector<uint64_t>& numbers) {
    if (numbers.size() < 2) return 0;
    uint64_t minDistance = std::numeric_limits<uint64_t>::max();

    for (size_t i = 1; i < numbers.size(); ++i) {
        uint64_t distance = numbers[i] - numbers[i - 1];
        minDistance = std::min(minDistance, distance);
    }

    return minDistance;
}

int main(int argc, char* argv[]) try {
    string src_dataset;
    string dst_dataset;
    bool deduplicate = false;
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0]
             << " <src_dataset> <dst_dataset> [deduplicate]" << endl;
        return 1;
    } else {
        src_dataset = argv[1];
        dst_dataset = argv[2];
        deduplicate = argc == 4 && argv[3] == string("deduplicate");
    }

    cout << "Dataset: " << src_dataset << endl;
    cout << "Output: " << dst_dataset << endl;
    cout << "Deduplicate: " << deduplicate << endl;

    std::vector<uint64_t> numbers = load_binary_data(src_dataset);
    if (deduplicate) {
        numbers = deduplicateSortedVector(numbers);
    }

    cout << "Dataset: " << src_dataset << endl;
    cout << "Size: " << numbers.size() << endl;
    if (numbers.size() < 2) {
        cout << "Average Distance: n/a (fewer than two keys)" << endl;
        cout << "Minimum Distance: n/a (fewer than two keys)" << endl;
    } else {
        cout << "Average Distance: " << average_distance(numbers) << endl;
        cout << "Minimum Distance: " << calculateMinimumDistance(numbers) << endl;
    }

    fh_index::util::StoreKeysToBinary(dst_dataset, numbers);

    return 0;
} catch (const std::exception& error) {
    cerr << "sort_dataset: " << error.what() << endl;
    return 1;
}
