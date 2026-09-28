#pragma once

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <memory>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }

// Spectral normal-compliance operator for a square, doubly periodic elastic
// half-space. For every non-zero reciprocal-lattice vector q,
//
//     u_hat(q) = 2 p_hat(q) / (E* |q|).
//
// The zero mode is set to zero: under force control it is an arbitrary rigid
// displacement and is absorbed by the contact solver's approach. The FFTs
// are unnormalised, so the symbol carries 1/Ns^2. The symbol is evaluated
// on the fly (no table); with the default pocketfft engine the only scratch
// is the (Ns/2+1) x Ns complex half spectrum per precision, and x/y may
// alias. One operator must not be applied from two threads concurrently.
class PeriodicFFTOperator {
public:
    PeriodicFFTOperator(int grid_size, double domain_size, double E_star);
    ~PeriodicFFTOperator();
    PeriodicFFTOperator(const PeriodicFFTOperator&) = delete;
    PeriodicFFTOperator& operator=(const PeriodicFFTOperator&) = delete;

    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;
    void matvec_into(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    Eigen::VectorXf matvec_single(const Eigen::VectorXf& x) const;
    void matvec_single_into(const Eigen::VectorXf& x, Eigen::VectorXf& y) const;
    void build_single_caches() const;
    void release_single_caches() const;
    std::size_t scratch_bytes() const; // workspace currently held

private:
    int Ns_ = 0, nh_ = 0;
    double scale_ = 0.0;               // L / (pi E* Ns^2)
    mutable Eigen::MatrixXd Gd_;       // FFTW engine only (plan-bound grid)
    mutable Eigen::MatrixXcd Cd_;
    mutable Eigen::MatrixXf Gf_;
    mutable Eigen::MatrixXcf Cf_;
    mutable std::unique_ptr<fft::SquareR2C<double>> fft_d_;
    mutable std::unique_ptr<fft::SquareR2C<float>> fft_f_;
};

} // namespace hmc
