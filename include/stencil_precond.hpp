#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <vector>

namespace hmc {

// Block layout for the compressed (active-set) stencil apply. Mirrors the
// H2Mask slot convention: occupied blocks of ls x ls elements get consecutive
// slot ids, and a compressed vector has nslots*ls^2 entries with slot s
// holding its block's elements in row-major local order
// (entry ly*ls + lx <-> grid element (by*ls + ly)*Ns + bx*ls + lx).
struct StencilBlockLayout {
    int Ns = 0;
    int ls = 0;                        // block side; nl = Ns/ls blocks per side
    int nslots = 0;
    std::vector<int> slot_bx, slot_by; // block coords of slot s (size nslots)
    std::vector<int> block_slot;       // nl*nl, at by*nl + bx: slot id or -1
    int nl() const { return Ns / ls; }
};

// Sparse real-space preconditioner for the projected CG, replacing the
// full-grid FFT application of the |k| symbol.
//
// The Boussinesq operator has symbol S(k) ~ 1/|k|, so the preconditioner uses
// m(k) = |k|. That is a POSITIVE-ORDER symbol, so its real-space kernel is
// short-ranged: 74.6% of sum|w| lies within radius 1 and 83.9% within radius
// 2, and truncating to the radius-2 disc reproduces the untruncated
// preconditioner's iteration count across every regime measured (spec §3.3).
// Because the residual is zero off the contact set, the truncated convolution
// evaluated on the contact set is IDENTICAL to the full-grid convolution
// sampled there; the only approximation anywhere is truncating w itself.
//
// Cost is O(taps * N_c) with no grid-sized allocation at all -- against the
// FFT path's two Ns^2 transforms and four full-grid passes per apply, which at
// Ns=16384 was 77-81% of the whole solve (spec §1).
//
// Weights are normalised to w(0,0) = 1: the overall scale of M^-1 cancels in
// CG, and the solver is scale-invariant by construction (A02).
class StencilPreconditioner {
public:
    // radius is the l2 disc radius in cells; 2 (13 taps) is the measured
    // default. Requires 2*radius + 1 <= Ns.
    explicit StencilPreconditioner(int Ns, int radius = 2);

    int radius() const { return R_; }
    int ntaps() const { return static_cast<int>(w_.size()); }
    const std::vector<int>& tap_dx() const { return dx_; }
    const std::vector<int>& tap_dy() const { return dy_; }
    const std::vector<double>& tap_w() const { return w_; }

    // z = M^-1 g, masked to {i : contact[i] != 0} and mean-zeroed over it.
    // Same contract as FourierPreconditioner::apply_into.
    void apply_into(const Eigen::VectorXd& g,
                    const std::vector<std::uint8_t>& contact,
                    Eigen::VectorXd& z) const;
    void apply_single_into(const Eigen::VectorXf& g,
                           const std::vector<std::uint8_t>& contact,
                           Eigen::VectorXf& z) const;

    // Compressed (slot-blocked) apply for the O(N_c) active-set solve.
    // Requires radius() <= layout.ls. Tap sums are bit-for-bit equal to
    // apply_into's on the same data; the final z differs only through the
    // contact-mean reduction order (as FourierPreconditioner already documents
    // for its indexed variant).
    void apply_into_blocked(const Eigen::VectorXd& gc,
                            const std::vector<std::uint8_t>& contact_c,
                            const StencilBlockLayout& layout,
                            Eigen::VectorXd& zc) const;
    void apply_single_into_blocked(const Eigen::VectorXf& gc,
                                   const std::vector<std::uint8_t>& contact_c,
                                   const StencilBlockLayout& layout,
                                   Eigen::VectorXf& zc) const;

    // API parity with FourierPreconditioner: there is no grid scratch to free.
    void release_scratch() const {}

private:
    int Ns_, R_;
    std::vector<int> dx_, dy_;   // tap offsets, |d|^2 <= R^2
    std::vector<double> w_;      // tap weights, w[0] == 1 at (0,0)
};

} // namespace hmc
