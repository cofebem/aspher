#pragma once

#include <cstddef>
#include <vector>

namespace hmc {

// A box of the uniform quad-tree. The grid is Ns x Ns elements; a box at
// `level` (0 = root) covers the element range [ix0, ix0+side) x [iy0, iy0+side)
// with side = Ns >> level. Only index ranges are stored (no per-box index
// lists): the global flat index of element (ix, iy) is iy*Ns + ix.
struct H2Box {
    int level;          // 0 = root
    int bx, by;         // box coordinates within the level, in [0, 2^level)
    int ix0, iy0, side; // element range origin and side length
    int parent;         // box id, -1 for root
    int child[4];       // child box ids by quadrant c = cx + 2*cy; -1 if none
    bool leaf;
};

// Balanced (complete) quad-tree over an Ns x Ns grid down to square leaves of
// `leaf_side` elements. Ns and leaf_side are powers of two with leaf_side | Ns.
class UniformQuadTree {
public:
    UniformQuadTree(int Ns, int leaf_side);

    int Ns() const { return Ns_; }
    int leaf_side() const { return leaf_side_; }
    int nlevels() const { return nlevels_; }
    int leaf_level() const { return nlevels_ - 1; }

    const std::vector<H2Box>& boxes() const { return boxes_; }

    // A07: storage of the tree itself. `logical` counts the elements in use,
    // `bytes` counts what the containers actually hold (capacity), which is
    // what the process paid for.
    std::size_t logical_bytes() const {
        return boxes_.size() * sizeof(H2Box) +
               level_begin_.size() * sizeof(int);
    }
    std::size_t bytes() const {
        return boxes_.capacity() * sizeof(H2Box) +
               level_begin_.capacity() * sizeof(int);
    }

    int level_begin(int level) const { return level_begin_[level]; }
    int boxes_per_side(int level) const { return 1 << level; }
    int box_id(int level, int bx, int by) const {
        return level_begin_[level] + by * boxes_per_side(level) + bx;
    }

    // Same-level boxes with Chebyshev box-distance <= r (includes self), clipped
    // to grid bounds.
    std::vector<int> neighbors(int box, int r) const;

    // FMM interaction list: children of parent's near-neighbors, minus this
    // box's own near-neighbors.
    std::vector<int> interaction_list(int box, int near_radius) const;

private:
    int Ns_, leaf_side_, nlevels_;
    std::vector<int> level_begin_;
    std::vector<H2Box> boxes_;
};

} // namespace hmc
