#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <memory>
#include <vector>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }

// Spectral preconditioner for the projected CG. The Boussinesq operator S has
// Fourier symbol Ŝ(q) ∝ 1/|q|, so M⁻¹ with symbol ∝ |q| collapses κ(S) ∼ Ns to
// ≈ O(1). Applied per iteration by FFT to the residual masked to the contact
// set; the result is kept on the contact set and its contact-mean removed (the
// total load / mean is fixed by the constraint, not by CG). The overall scale
// of M⁻¹ is irrelevant (it cancels in CG), so the bare wavenumber |k| is used
// and the q=0 mode is zeroed.
//
// The 2-D transforms run in real half-spectrum form (only kx ∈ [0, Ns/2] is
// stored; the unnormalised r2c+c2r round-trip scale Ns² is folded into the
// symbol) through the shared engine in src/fft_engine.hpp
// (hmc::fft::SquareR2C), selected at build time:
//   * default: the bundled pocketfft (BSD-3-Clause, header-only, SIMD,
//     multithreaded) — keeps binaries permissively licensed;
//   * -DASPHER_USE_FFTW=ON: FFTW3 plans (slightly faster; FFTW is GPL, so
//     distributed binaries then carry GPL terms).
// Grid-sized scratch is owned by the object, sized lazily per precision, and
// reused across calls, so no large temporaries are allocated per iteration.
// A single FourierPreconditioner must not be applied (or first-used) from two
// threads concurrently; instances are non-copyable.
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
    // (nh x Ns) symbol |k|/Ns² at (kx, ky): wavenumber magnitude with the
    // unnormalised r2c+c2r round-trip scale folded in; wh_(0,0) = 0
    Eigen::MatrixXf wh_;

    // reusable scratch, sized lazily per precision on first use:
    // G (Ns x Ns real, laid out G(ix, iy)) and C (nh x Ns complex half
    // spectrum, laid out C(kx, ky))
    mutable Eigen::MatrixXd Gd_;
    mutable Eigen::MatrixXcd Cd_;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;

    // FFT engine (FFTW plans / pocketfft size bookkeeping), created on first
    // use per precision and bound to the scratch pointers (so the scratch is
    // never reallocated afterwards)
    mutable std::unique_ptr<fft::SquareR2C<double>> fft_d_;
    mutable std::unique_ptr<fft::SquareR2C<float>> fft_f_;
};

} // namespace hmc
