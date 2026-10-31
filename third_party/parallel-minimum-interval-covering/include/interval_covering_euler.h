#ifndef INTERVAL_COVERING_EULER_H
#define INTERVAL_COVERING_EULER_H

// Legacy Euler Tour based implementation
// This file preserves the original LinkList-based algorithm for reference
// The current implementation uses a more efficient sampling-based approach

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
class IntervalCoveringEuler {
 public:
  // Deduce endpoint type T from GetL's return type
  using T = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<GetL>()(size_t(0)))>>;

  // Deduce GetR's return type for comparison
  using T_GetR = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<GetR>()(size_t(0)))>>;

  // Static assertion to ensure GetL and GetR return the same type
  static_assert(std::is_same<T, T_GetR>::value,
                "GetL and GetR must return the same type");

  // Constructor
  IntervalCoveringEuler(size_t n, GetL L, GetR R)
      : n(n), L(L), R(R), furthest_id(n) {}

  class LinkListNode {
    public:
    LinkListNode(bool valid = false, size_t nxt = kNullPtr, bool sampled = false)
        : nxt(nxt), sampled(sampled), valid(valid) {}

    bool get_valid() const { return valid; }
    void set_valid(bool v) { valid = v; }

    bool get_sampled() const { return sampled; }
    void set_sampled(bool s) { sampled = s; }

    size_t get_nxt() const { return nxt; }
    void set_nxt(size_t n) {
      assert(n <= kNullPtr);
      nxt = n;
    }

  private:
    size_t nxt;      // Next pointer (62 bits)
    bool sampled;   // Sampled flag (1 bit)
    bool valid;     // Valid flag (1 bit)
  };

  static constexpr size_t kNullPtr = (1ULL << 62) - 1;

  size_t l_nodeid(size_t i) const { return i * 2; }
  size_t r_nodeid(size_t i) const { return i * 2 + 1; }
  LinkListNode& l_node(size_t i) { return link_list[l_nodeid(i)]; }
  LinkListNode& r_node(size_t i) { return link_list[r_nodeid(i)]; }
  void link(size_t from, size_t to) { link_list[from].set_nxt(to); }
  std::string node_str(size_t nodeid) {
    if (nodeid == kNullPtr) {
      return "nullptr";
    } else {
      size_t i = nodeid;
      bool is_l = (i % 2 == 0);
      size_t id = i / 2;
      return (is_l ? "l" : "r") + std::to_string(id);
    }
  }

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

  void BuildLinkListFast() {
    link_list = parlay::sequence<LinkListNode>(n * 2, LinkListNode());

    assert(n >= 2);

    // furthest_id[n - 2] == furthest_id[n - 1] == n - 1, can cause loop
    link(l_nodeid(n - 2), r_nodeid(n - 2));
    link(r_nodeid(n - 2), r_nodeid(n - 1));
    parlay::parallel_for(0, n - 2, [&](size_t i) {
      link(l_nodeid(i), r_nodeid(i));
      if (furthest_id[i] == furthest_id[i + 1]) {
        link(r_nodeid(i), l_nodeid(i + 1));
      } else {
        link(r_nodeid(i), r_nodeid(furthest_id[i]));
      }
    }, parallel_block_size);

    // furthest_id[0] != 0, must link
    // furthest_id[n - 1] == n - 1, no need to link
    link(l_nodeid(furthest_id[0]), l_nodeid(0));
    parlay::parallel_for(1, n - 1, [&](size_t i) {
      if (furthest_id[i - 1] != furthest_id[i]) {
        link(l_nodeid(furthest_id[i]), l_nodeid(i));
      }
    }, parallel_block_size);

    r_node(0).set_valid(true);
  }

  // Build the link list for the euler tour
  void BuildLinkList() {
    BuildLinkListFast();
  }

  void ScanLinkListSerial() {
    // Serial version: start from l_nodeid(n-1) and follow the link list
    size_t node_id = l_nodeid(n - 1);
    bool valid = false;
    while (node_id != kNullPtr) {
      valid = valid || link_list[node_id].get_valid();
      link_list[node_id].set_valid(valid);
      node_id = link_list[node_id].get_nxt();
    }
  }

  void BuildSampleId() {
    // sample n * 2 / parallel_block_size nodes from the link list
    size_t nn = n * 2;
    parlay::random rnd(0);

    size_t total_sampled_max = 1 + (nn + parallel_block_size - 1) / parallel_block_size;
    sampled_id.resize(total_sampled_max);
    size_t actual_sampled = 0;

    auto sample = [&](size_t node_id) {
      if (link_list[node_id].get_sampled()) return;
      assert(actual_sampled < total_sampled_max);
      link_list[node_id].set_sampled(true);
      sampled_id[actual_sampled++] = node_id;
    };

    sample(l_nodeid(n - 1));  // always sample the start node
    for (size_t i = 1; i < total_sampled_max; i ++) {
      size_t node_id = rnd.ith_rand(i) % nn;
      sample(node_id);
    }
    sampled_id.resize(actual_sampled);
  }

  void ScanLinkListParallel() {
    BuildSampleId();

    // scan from each sampled node until next sampled node
    sampled_id_nxt_initial.resize(sampled_id.size());
    parlay::parallel_for(0, sampled_id.size(), [&](size_t i) {
      // save the initial nxt for sampled nodes
      sampled_id_nxt_initial[i] = link_list[sampled_id[i]].get_nxt();

      size_t start_id = sampled_id[i];
      bool valid = link_list[start_id].get_valid();
      size_t node_id = link_list[start_id].get_nxt();
      while (node_id != kNullPtr) {
        valid = valid || link_list[node_id].get_valid();
        link_list[node_id].set_valid(valid);
        if (link_list[node_id].get_sampled()) {
          break;
        }
        node_id = link_list[node_id].get_nxt();
      }

      // link sampled nodes together
      link(sampled_id[i], node_id);
    }, 1);

    // scan over sampled sketch
    {
      size_t node_id = sampled_id[0];
      bool valid = false;
      while (node_id != kNullPtr) {
        valid = valid || link_list[node_id].get_valid();
        link_list[node_id].set_valid(valid);
        node_id = link_list[node_id].get_nxt();
      }
    }


    // FIX: Separate restore and scan into two phases to avoid data race

    // Phase 1: Restore all links first
    // parlay::parallel_for(0, sampled_id.size(), [&](size_t i) {
    //   link(sampled_id[i], sampled_id_nxt_initial[i]);
    // }, 1);

    // Phase 2: Then scan from each sampled node
    parlay::parallel_for(0, sampled_id.size(), [&](size_t i) {
      size_t start_id = sampled_id[i];
      bool valid = link_list[start_id].get_valid();
      link(start_id, sampled_id_nxt_initial[i]);  // Restore link

      size_t node_id = sampled_id_nxt_initial[i];
      while (node_id != kNullPtr) {
        valid = valid || link_list[node_id].get_valid();
        link_list[node_id].set_valid(valid);
        if (link_list[node_id].get_sampled()) {
          break;
        }
        node_id = link_list[node_id].get_nxt();
      }
    }, 1);
  }

  void PrintIntervals() {
    for (size_t i = 0; i < n; i++) {
      std::cout << "Interval " << i << ": [" << L(i) << ", " << R(i) << "]\n";
    }
  }

  void PrintLinkListNodes() {
    for (size_t i = 0; i < link_list.size(); i++) {
      std::cout << node_str(i) << ": nxt=" << node_str(link_list[i].get_nxt())
                << ", valid=" << link_list[i].get_valid()
                << ", sampled=" << link_list[i].get_sampled() << "\n";
    }
  }

  void PrintLinkList() {
    size_t nodeid = l_nodeid(n - 1);
    while (nodeid != kNullPtr) {
      std::cout << node_str(nodeid) << " (valid=" << link_list[nodeid].get_valid() << ")\n";
      nodeid = link_list[nodeid].get_nxt();
    }
  }

   void ScanLinkList() {
    DEBUG_ONLY {
      size_t nn = n * 2;

      // Save the original link_list valid flags
      parlay::sequence<bool> saved_valid(nn);
      parlay::parallel_for(
          0, nn, [&](size_t i) { saved_valid[i] = link_list[i].get_valid(); });

      // Run parallel version
      ScanLinkListParallel();

      // Save parallel results
      parlay::sequence<bool> parallel_valid(nn);
      parlay::parallel_for(
          0, nn, [&](size_t i) { parallel_valid[i] = link_list[i].get_valid(); });

      // Restore original state
      parlay::parallel_for(
          0, nn, [&](size_t i) { link_list[i].set_valid(saved_valid[i]); });

      // Run serial version
      ScanLinkListSerial();

      // Verify results match
      parlay::parallel_for(0, nn, [&](size_t i) {
        if (link_list[i].get_valid() != parallel_valid[i]) {
          printf("ScanLinkList mismatch at %ld: serial=%d, parallel=%d\n", i,
                 link_list[i].get_valid(), parallel_valid[i]);
          PrintLinkList();
          PrintIntervals();
          assert(false);
        }
      });

      return;
    }

    // Non-debug mode: just run parallel version
    ScanLinkListParallel();
  }

  void KernelParallel() {
    BuildFurthest();
    BuildLinkList();
    ScanLinkList();
    parlay::parallel_for(0, n, [&](size_t i) {
      valid[i] = (l_node(i).get_valid() != r_node(i).get_valid()) ? 1 : 0;
    });
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
  // kNullPtr must fit in 62 bits (LinkListNode.nxt is 62 bits)

  size_t n;
  GetL L;
  GetR R;

  parlay::sequence<LinkListNode> link_list;
  parlay::sequence<bool> valid;
  parlay::sequence<size_t> furthest_id;
  parlay::sequence<size_t> sampled_id, sampled_id_nxt_initial;
};

#endif
