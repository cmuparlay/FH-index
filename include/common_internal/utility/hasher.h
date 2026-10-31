#ifndef COMMON_HASHER_H
#define COMMON_HASHER_H

#include <cstdint>
#include <cstddef>
#include "parlay/utilities.h"

namespace fh_index {
namespace internal {

struct hash {
  size_t operator()(const uint64_t p) const {
    return parlay::hash64(p);
  }
};

}  // namespace internal
}  // namespace fh_index

#endif