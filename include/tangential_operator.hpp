#pragma once

#include "cerruti_kernel.hpp"
#include "h2_operator.hpp"

#include <Eigen/Dense>
#include <memory>

namespace hmc {

namespace fft { template <class S> class SquareR2C; }

// Exact FFT-convolution operator for the tangential (Cerruti) 2x2 block:
// u = C q with q = [q_x; q_y] and u = [u_x; u_y] stacked (each N = Ns² in
// natural flat order), matching CerrutiKernel::assemble_dense's layout
// [[XX, XY],[XY, YY]]. Same zero-padded (2Ns)² Hockney scheme as
// FFTOperator: the three kernel tables are embedded in wrap-around order and
// r2c-transformed once at build(). All three tables are even under JOINT
// offset negation (xx/yy even in each axis; xy odd in each axis, hence
// even under (dx,dy) -> (-dx,-dy)), so all three spectra are REAL
// (matching the real continuum symbol; verified at build) and the
// per-mode mix is all-real:
//   Ux(k) = Kxx(k)·Qx(k) + Kxy(k)·Qy(k)
//   Uy(k) = Kxy(k)·Qx(k) + Kyy(k)·Qy(k)
// (all spectra carry the unnormalised fwd+inv 1/M² scale). Per matvec:
// 2 zero-pad scatters -> 2 fwd -> in-place 2x2 spectral mix -> 2 inv ->
// 2 gathers. Matches the dense block matvec to roundoff.
//
// The kernel must outlive the operator. Object-owned scratch (two padded
// grids + two half-spectra) is sized lazily and reused: a single instance
// must not be applied from two threads concurrently. Non-copyable.
class TangentialFFTOperator {
public:
    explicit TangentialFFTOperator(const CerrutiKernel& kernel);
    ~TangentialFFTOperator();
    TangentialFFTOperator(const TangentialFFTOperator&) = delete;
    TangentialFFTOperator& operator=(const TangentialFFTOperator&) = delete;

    void build();

    Eigen::VectorXd matvec(const Eigen::VectorXd& q) const;
    void matvec_into(const Eigen::VectorXd& q, Eigen::VectorXd& u) const;

private:
    const CerrutiKernel* kernel_;
    int Ns_, M_, nh_; // M_ = 2 Ns_, nh_ = Ns_ + 1

    // (nh x M) real spectra, 1/M² folded in (all three kernels are even
    // under joint negation, so all three spectra are real)
    Eigen::MatrixXd Kxx_, Kyy_, Kxy_;

    // padded scratch, sized lazily on first matvec (one grid+spectrum pair
    // per traction component; each engine binds one pair)
    mutable Eigen::MatrixXd G1_, G2_;
    mutable Eigen::MatrixXcd C1_, C2_;
    mutable std::unique_ptr<fft::SquareR2C<double>> fft1_, fft2_;
};

// Matrix-free O(N) tangential operator: three scalar H2Operator instances
// (xx, yy, xy) built through the M1 kernel-functor constructor from a
// CerrutiKernel (which must outlive this object). Blocked matvec
//   u_x = XX q_x + XY q_y,  u_y = XY q_x + YY q_y
// costs 4 scalar H2 applies + 2 adds. Same stacked [q_x; q_y] layout and
// dense-reference semantics as TangentialFFTOperator. Component scratch is
// object-owned and reused: no concurrent applies on one instance.
class TangentialH2Operator {
public:
    TangentialH2Operator(const CerrutiKernel& kernel, H2Params params);

    void build();

    Eigen::VectorXd matvec(const Eigen::VectorXd& q) const;
    void matvec_into(const Eigen::VectorXd& q, Eigen::VectorXd& u) const;

private:
    int Ns_;
    H2Operator Hxx_, Hyy_, Hxy_;
    mutable Eigen::VectorXd qc_, tc_; // component input / output scratch
};

} // namespace hmc
