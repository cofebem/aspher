#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace hmc {

// Spectral preconditioner for the projected CG. The Boussinesq operator S has
// Fourier symbol Ŝ(q) ∝ 1/|q|, so M⁻¹ with symbol ∝ |q| collapses κ(S) ∼ Ns to
// ≈ O(1). Applied per iteration by FFT to the residual masked to the contact
// set; the result is kept on the contact set and its contact-mean removed (the
// total load / mean is fixed by the constraint, not by CG). The overall scale
// of M⁻¹ is irrelevant (it cancels in CG), so the bare wavenumber |k| is used
// and the q=0 mode is zeroed.
//
// The field is real, so the x-direction transforms run in half-spectrum r2c /
// c2r form: only kx ∈ [0, Ns/2] is stored and transformed, halving both the
// FFT work and the complex buffer. All grid-sized scratch is owned by the
// object and reused across calls (apply is called every CG iteration), so no
// large temporaries are allocated per iteration. Consequently a single
// FourierPreconditioner must not be applied from two threads concurrently.
class FourierPreconditioner {
public:
    explicit FourierPreconditioner(int Ns);

    // z = M⁻¹ g, masked to {i : contact[i] != 0} and mean-zeroed over it.
    Eigen::VectorXd apply(const Eigen::VectorXd& g,
                          const std::vector<std::uint8_t>& contact) const;

    // Allocation-free variant: writes into z (resized if needed), so the CG
    // loop reuses one buffer across iterations.
    void apply_into(const Eigen::VectorXd& g,
                    const std::vector<std::uint8_t>& contact,
                    Eigen::VectorXd& z) const;

    // Single-precision variants (FFT done in float; symbol applied as float).
    Eigen::VectorXf apply_single(const Eigen::VectorXf& g,
                                 const std::vector<std::uint8_t>& contact) const;
    void apply_single_into(const Eigen::VectorXf& g,
                           const std::vector<std::uint8_t>& contact,
                           Eigen::VectorXf& z) const;

private:
    int Ns_, nh_; // nh_ = Ns/2 + 1 stored kx modes
    Eigen::MatrixXf wh_; // (nh x Ns) wavenumber symbol |k| at (kx, ky), wh_(0,0)=0

    // reusable scratch, sized lazily per precision on first use:
    // G (Ns x Ns real, laid out G(ix, iy)) and C (nh x Ns complex half spectrum)
    mutable Eigen::MatrixXd Gd_;
    mutable Eigen::MatrixXcd Cd_;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;
};

} // namespace hmc
