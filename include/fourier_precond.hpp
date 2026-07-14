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

    // Indexed (compressed) variants for the O(N_c) active-set solve: gc,
    // contact_c and zc are compressed vectors whose entry k lives at flat
    // grid index grid_index[k] (all indices distinct). The masked residual is
    // scattered into the owned Ns×Ns grid (zeroed first — the FFT grid stays
    // full-size, the documented M3 trade-off), transformed as usual, and
    // gathered back at the same indices; the contact mean is removed over the
    // compressed contact set. Matches apply_into on the same data up to the
    // reduction summation order (zmean), i.e. to roundoff, not bit-for-bit.
    void apply_into_indexed(const Eigen::VectorXd& gc,
                            const std::vector<std::uint8_t>& contact_c,
                            const std::vector<int>& grid_index,
                            Eigen::VectorXd& zc) const;
    void apply_single_into_indexed(const Eigen::VectorXf& gc,
                                   const std::vector<std::uint8_t>& contact_c,
                                   const std::vector<int>& grid_index,
                                   Eigen::VectorXf& zc) const;

    // Free the owned grid/spectrum scratch (~4N reals per precision in use)
    // and the FFT engines. Safe at any point; the next apply reallocates
    // lazily. The active-set driver calls this after the last CG iteration so
    // the end-of-solve pressure scatter does not stack on top of it.
    void release_scratch() const;

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

// 2×2 spectral preconditioner for the tangential projected CG (friction M4).
// The tangential operator has the matrix symbol Ĉ(k) ∝ (1/|k|)(I − ν k̂k̂ᵀ);
// its analytic inverse (up to the CG-irrelevant overall scale) is
//   M⁻¹(k) = |k| (I + γ k̂k̂ᵀ),   γ = ν/(1−ν),
// applied per mode to the mask-restricted stacked residual [g_x; g_y]:
// scatter (masked, per point) → 2 fwd FFTs → per-mode symmetric 2×2 multiply
// → 2 inv FFTs → masked gather (per-component mask-mean removed only under
// force control, where the load constraint fixes the mean). Integer
// wavenumbers, DC zeroed, unnormalised round-trip scale 1/Ns² folded in —
// exactly the FourierPreconditioner conventions. Non-padded (periodic)
// transforms: like the normal |q| preconditioner this is an approximate
// inverse; only CG direction quality depends on it, never the solution.
// Grid scratch is object-owned, lazily sized, reused; no concurrent applies
// on one instance; non-copyable. Double-only in M4.
class TangentialFourierPreconditioner {
public:
    TangentialFourierPreconditioner(int Ns, double nu);
    ~TangentialFourierPreconditioner();
    TangentialFourierPreconditioner(const TangentialFourierPreconditioner&) =
        delete;
    TangentialFourierPreconditioner&
    operator=(const TangentialFourierPreconditioner&) = delete;

    void apply_into(const Eigen::VectorXd& g,
                    const std::vector<std::uint8_t>& mask, bool remove_mean,
                    Eigen::VectorXd& z) const;

private:
    int Ns_, nh_; // nh_ = Ns/2 + 1 stored kx modes
    // (nh x Ns) inverse-symbol entries with 1/Ns² folded in; DC zeroed:
    //   wxx = |k|(1 + γ k̂x²), wyy = |k|(1 + γ k̂y²), wxy = |k| γ k̂x k̂y
    Eigen::MatrixXd wxx_, wyy_, wxy_;
    mutable Eigen::MatrixXd G1_, G2_;
    mutable Eigen::MatrixXcd C1_, C2_;
    mutable std::unique_ptr<fft::SquareR2C<double>> fft1_, fft2_;
};

} // namespace hmc
