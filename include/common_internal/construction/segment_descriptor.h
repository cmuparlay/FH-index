#ifndef FH_INDEX_INTERNAL_CONSTRUCTION_SEGMENT_DESCRIPTOR_H
#define FH_INDEX_INTERNAL_CONSTRUCTION_SEGMENT_DESCRIPTOR_H

#include <cstddef>
#include <cstdint>

namespace fh_index {
namespace internal {
namespace segmenting {

/**
 * SegmentDescriptor - Describes a segment in the segmentation algorithm
 *
 * A segment represents a contiguous range of keys [l, r) that will be
 * represented by a single Finger Group Descriptor (FGD).
 *
 * Fields:
 * - l: Left boundary (inclusive) of the segment in the key array
 * - r: Right boundary (exclusive) of the segment in the key array
 */
struct SegmentDescriptor {
  size_t l;  // Left boundary (inclusive)
  size_t r;  // Right boundary (exclusive)
};

}  // namespace segmenting
}  // namespace internal
}  // namespace fh_index

#endif  // FH_INDEX_INTERNAL_CONSTRUCTION_SEGMENT_DESCRIPTOR_H
