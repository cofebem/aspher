#pragma once

#include "boussinesq_kernel.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }

struct FFTInfo {
    int N = 0, Ns = 0, M = 0; // M = 2 Ns padded grid side
    std::int64_t bytes_spectrum = 0; // stored real kernel spectrum (per precision)
    std::int64_t bytes_scratch = 0;  // padded grid + half-spectrum work buffers
    std::int64_t bytes_total = 0;
};

// Exact FFT-convolution operator for the translation-invariant Boussinesq
// influence matrix on the uniform Ns x Ns grid: u = S p is the aperiodic
// convolution of p with the Love (1929) element table T(dx, dy) — the same
// numbers the dense matrix uses — evaluated as a circular convolution on a
// (2Ns)² zero-padded grid (Hockney). The kernel table is embedded once in
// wrap-around order and r2c-transformed; T is even in both axes, so its
// spectrum is real (verified at build) and stored as such, with the
// unnormalised fwd+inv round-trip scale 1/(2Ns)² folded in. Per matvec:
// zero-pad scatter -> r2c -> pointwise multiply -> c2r -> gather. Equals the
// dense matvec to roundoff (no interpolation, no spectral truncation — the
// near field IS the far field here), unlike the H2 operator's ~1e-4 (q=4)
// interpolation error. Free-space (non-periodic) BCs are exact: the padding
// removes all wrap-around aliasing for an Ns-support pressure.
//
// O(N log N) matvec, ~10 N reals of double scratch. Padded scratch and the
// FFT engine state are object-owned, sized lazily per precision, and reused
// across calls: a single FFTOperator must not be applied from two threads
// concurrently (same contract as H2Operator). Instances are non-copyable.
class FFTOperator {
public:
    explicit FFTOperator(const BoussinesqKernel& kernel);
    ~FFTOperator();
    FFTOperator(const FFTOperator&) = delete;
    FFTOperator& operator=(const FFTOperator&) = delete;

    void build();

    // u = S x, with x and u in natural flat order (global = iy*Ns + ix).
    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;
    // Allocation-free variant: writes into y (resized if needed).
    void matvec_into(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    // Single-precision matvec: float copy of the kernel spectrum (built
    // lazily, idempotent) and float transforms, halving the O(N) working set.
    Eigen::VectorXf matvec_single(const Eigen::VectorXf& x) const;
    void matvec_single_into(const Eigen::VectorXf& x, Eigen::VectorXf& y) const;
    void build_single_caches() const;

    FFTInfo info() const;
    void print_statistics() const;

private:
    const BoussinesqKernel* kernel_;
    int Ns_, M_, nh_; // M_ = 2 Ns_, nh_ = M_/2 + 1 = Ns_ + 1

    // (nh x M) real kernel spectrum with the 1/M² round-trip scale folded in
    Eigen::MatrixXd Kh_;
    mutable bool have_single_ = false;
    mutable Eigen::MatrixXf Kh_f_;

    // padded scratch per precision, sized lazily on first matvec: G (M x M
    // real, laid out G(ix, iy)) and C (nh x M complex half spectrum)
    mutable Eigen::MatrixXd Gd_;
    mutable Eigen::MatrixXcd Cd_;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;
    mutable std::unique_ptr<fft::SquareR2C<double>> fft_d_;
    mutable std::unique_ptr<fft::SquareR2C<float>> fft_f_;

    template <class S>
    void matvec_impl(const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
                     Eigen::Matrix<S, Eigen::Dynamic, 1>& y,
                     const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Kh,
                     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& G,
                     Eigen::Matrix<std::complex<S>, Eigen::Dynamic,
                                   Eigen::Dynamic>& C,
                     std::unique_ptr<fft::SquareR2C<S>>& eng) const;
};

} // namespace hmc
