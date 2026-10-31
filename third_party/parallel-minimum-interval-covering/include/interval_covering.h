#ifndef INTERVAL_COVERING_H
#define INTERVAL_COVERING_H

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

#include "macro.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

template<typename GetL, typename GetR>
class IntervalCovering {
 public:
  // Deduce endpoint type T from GetL's return type
  using T = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<GetL>()(size_t(0)))>>;

  // Deduce GetR's return type for comparison
  using T_GetR = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<GetR>()(size_t(0)))>>;

  // Static assertion to ensure GetL and GetR return the same type
  static_assert(std::is_same<T, T_GetR>::value,
                "GetL and GetR must return the same type");

  // Constructor
  IntervalCovering(size_t n, GetL L, GetR R)
      : n(n), L(L), R(R), furthest_id(n) {}

  void BuildFurthestSerial(size_t ll, size_t lr, size_t rl, size_t rr) {
    size_t rid = rl;
    for (size_t i = ll; i <= lr; i ++) {
      T r_of_i = R(i);
      while (rid <= rr && L(rid) <= r_of_i) {
        rid++;
      }
      furthest_id[i] = rid - 1;
    }
  }

  // merge L[ll, lr] with R[rl, rr] to find furthest
  void BuildFurthestParallelCore(size_t ll, size_t lr, size_t rl, size_t rr) {
    if (lr - ll + 1 + rr - rl + 1 <= parallel_merge_size) {
      BuildFurthestSerial(ll, lr, rl, rr);
      return;
    }

    size_t lmid = (ll + lr) >> 1;
    T r_of_lmid = R(lmid);
    
    // find furthest[lmid] in [l, r)
    size_t l = std::max(lmid, rl);
    size_t r = rr + 1;
    while (l + 1 < r) {
      size_t mid = (l + r) >> 1;
      if (L(mid) <= r_of_lmid) {
        l = mid;
      } else {
        r = mid;
      }
    }
    furthest_id[lmid] = l;

    // recurse on left and right halves
    parlay::par_do(
      [&]() {
        if (ll < lmid) {
          BuildFurthestParallelCore(ll, lmid - 1, rl, l);
        }
      },
      [&]() {
        if (lmid < lr) {
          BuildFurthestParallelCore(lmid + 1, lr, l, rr);
        }
      }
    );

  }

  // merge L with R to find furthest
  void BuildFurthestParallel() {
    BuildFurthestParallelCore(0, n - 1, 0, n - 1);
  }

  void BuildFurthest() {
    BuildFurthestParallel();

    DEBUG_ONLY {
      // Save parallel results
      parlay::sequence<size_t> furthest_id_parallel(n);
      parlay::parallel_for(
          0, n, [&](size_t i) { furthest_id_parallel[i] = furthest_id[i]; });

      // Run serial version on a copy to verify
      parlay::sequence<size_t> furthest_id_serial(n);
      std::swap(furthest_id, furthest_id_serial);
      BuildFurthestSerial(0, n - 1, 0, n - 1);

      // Verify results match
      for (size_t i = 0; i < n; i++) {
        assert(furthest_id[i] == furthest_id_parallel[i]);
      }

      // Restore parallel results
      furthest_id = furthest_id_parallel;
    }
  }


  void PrintIntervals() {
    for (size_t i = 0; i < n; i++) {
      std::cout << "Interval " << i << ": [" << L(i) << ", " << R(i) << "]\n";
    }
  }


  void BuildIntervalSample() {
    size_t sample_rate = parallel_block_size;
    parlay::random rnd(0);
    sampled = parlay::tabulate(n, [&](size_t i) -> bool {
      return rnd.ith_rand(i) % sample_rate == 0;
    });
    sampled[0] = sampled[n - 1] = true;

    sampled_id = parlay::pack_index(sampled);
  }

  // Build connections between sampled intervals
  // For each sampled interval, find the next sampled interval on the optimal path
  void BuildConnectionBetweenSamples() {
    sampled_id_nxt_initial = parlay::sequence<size_t>(n, 0);
    parlay::parallel_for(0, sampled_id.size(), [&](size_t i) {
      size_t start_id = sampled_id[i];
      size_t id = furthest_id[start_id];
      while (!sampled[id]) {
        id = furthest_id[id];
      }
      sampled_id_nxt_initial[start_id] = id;
    });
  }

  void ScanSamples() {
    valid_sampled_node = std::vector<size_t>();
    size_t id = 0;
    while (id < n - 1) {
      valid[id] = true;
      valid_sampled_node.push_back(id);
      id = sampled_id_nxt_initial[id];
    }
    // no need to push n-1 into valid_sampled_node
    valid[id] = true;
  }

  void ScanNonsampleNodes() {
    parlay::parallel_for(0, valid_sampled_node.size(), [&](size_t i) {
      size_t start_id = valid_sampled_node[i];
      size_t end_id = sampled_id_nxt_initial[start_id];

      size_t id = furthest_id[start_id];
      while (id != end_id) {
        assert(id < end_id);
        valid[id] = true;
        id = furthest_id[id];
      }
    });
  }

  void KernelParallelFast() {
    BuildFurthest();
    BuildIntervalSample();
    BuildConnectionBetweenSamples();
    ScanSamples();
    ScanNonsampleNodes();
  }

  void KernelParallel() {
    KernelParallelFast();
  }

  void KernelSerial() {
    valid[0] = 1;
    valid[n - 1] = 1;

    size_t id = 0;
    for (size_t i = 1; i < n - 1; i++) {
      if (L(i + 1) > R(id)) {
        valid[i] = 1;
        id = i;
      } else {
        valid[i] = 0;
      }
    }
  }

  void Kernel() {
    if (n <= 2) {
      valid[0] = 1;
      valid[n - 1] = 1;
      return;
    }

    KernelParallel();

    DEBUG_ONLY {
      // Save parallel results
      parlay::sequence<bool> valid_parallel(n, false);
      parlay::parallel_for(0, n, [&](size_t i) { valid_parallel[i] = valid[i]; });

      // Reset valid
      memset(valid.data(), 0, n * sizeof(bool));

      // Run serial version
      KernelSerial();

      // Verify results match
      for (size_t i = 0; i < n; i++) {
        if (valid[i] != valid_parallel[i]) {
          printf("Kernel mismatch at %ld: serial=%d, parallel=%d\n", i,
                 valid[i], valid_parallel[i]);
          assert(false);
        }
      }
    }
  }

  void Run() {
    if (n == 0) {
      return;
    }

    valid = parlay::sequence<bool>(n, 0);
    
    DEBUG_ONLY {
      // L(i) <= L(i+1) and R(i) <= R(i+1) (monotonically non-decreasing)
      parlay::parallel_for(0, n - 1, [&](size_t i) {
        assert(L(i) <= L(i + 1) && R(i) <= R(i + 1));
      });

      // L(i) < R(i)
      parlay::parallel_for(0, n, [&](size_t i) {
        assert(L(i) < R(i));
      });

      // L(i + 1) <= R(i)
      parlay::parallel_for(0, n - 1, [&](size_t i) {
        assert(L(i + 1) <= R(i));
      });
    }

    Kernel();
  }

  static constexpr size_t parallel_block_size = parlay::internal::_block_size;
  static constexpr size_t parallel_merge_size = 2000;

  size_t n;
  GetL L;
  GetR R;

  std::vector<size_t> valid_sampled_node;
  parlay::sequence<bool> valid, sampled;
  parlay::sequence<size_t> furthest_id;
  parlay::sequence<size_t> sampled_id, sampled_id_nxt_initial;
};

#endif