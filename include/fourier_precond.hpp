#pragma once

#include <Eigen/Dense>
#include <fftw3.h>

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
// The 2-D transforms run through FFTW's real 2-D r2c/c2r plans (SIMD kernels,
// internal multithreading when built with the fftw3*_omp libraries): only the
// kx ∈ [0, Ns/2] half spectrum is stored, and FFTW's Ns·Ns round-trip scale is
// folded into the symbol. Plans and all grid-sized scratch are owned by the
// object, created lazily per precision on first use, and reused across calls
// (apply runs every CG iteration), so no large temporaries are allocated per
// iteration. Consequently a single FourierPreconditioner must not be applied
// (or first-used) from two threads concurrently, and instances are
// non-copyable.
class FourierPreconditioner {
public:
    explicit FourierPreconditioner(int Ns);
    ~FourierPreconditioner();
    FourierPreconditioner(const FourierPreconditioner&) = delete;
    FourierPreconditioner& operator=(const FourierPreconditioner&) = delete;

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
    // (nh x Ns) symbol |k|/Ns² at (kx, ky): wavenumber magnitude with FFTW's
    // unnormalised r2c+c2r round-trip scale folded in; wh_(0,0) = 0
    Eigen::MatrixXf wh_;

    // reusable scratch + FFTW plans, created lazily per precision on first
    // use (the plans bind to the scratch pointers, so the scratch is never
    // reallocated): G (Ns x Ns real, laid out G(ix, iy)) and C (nh x Ns
    // complex half spectrum, laid out C(kx, ky))
    mutable Eigen::MatrixXd Gd_;
    mutable Eigen::MatrixXcd Cd_;
    mutable fftw_plan fwd_d_ = nullptr, inv_d_ = nullptr;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;
    mutable fftwf_plan fwd_f_ = nullptr, inv_f_ = nullptr;
};

} // namespace hmc
