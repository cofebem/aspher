#include "stencil_precond.hpp"

#include "fft_engine.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hmc {

namespace {

// Real-space kernel of the symbol |k| on an Nw x Nw periodic grid, returned
// normalised by w(0,0). The shape converges to grid-independent constants
// (spec §2.1), so a small reference grid suffices for any Ns; the coarse
// levels below Nw use their own Ns and are therefore exact.
std::vector<double> kernel_grid(int Nw) {
    const int nh = Nw / 2 + 1;
    Eigen::MatrixXd G(Nw, Nw);          // G(ix, iy), ix contiguous
    Eigen::MatrixXcd C(nh, Nw);         // C(kx, ky), kx contiguous
    fft::SquareR2C<double> eng;
    eng.bind(Nw, G.data(), C.data());

    auto kof = [Nw](int i) { return (i < Nw / 2) ? i : i - Nw; };
    for (int ky = 0; ky < Nw; ++ky)
        for (int kx = 0; kx < nh; ++kx)
            C(kx, ky) = std::complex<double>(
                std::hypot(static_cast<double>(kx),
                           static_cast<double>(kof(ky))),
                0.0);
    C(0, 0) = std::complex<double>(0.0, 0.0); // DC zeroed, as in the FFT path

    eng.inv(); // G = unnormalised inverse transform = w up to a scale

    std::vector<double> w(static_cast<std::size_t>(Nw) * Nw);
    const double c = G(0, 0);
    for (int iy = 0; iy < Nw; ++iy)
        for (int ix = 0; ix < Nw; ++ix)
            w[static_cast<std::size_t>(iy) * Nw + ix] = G(ix, iy) / c;
    return w;
}

// One masked stencil evaluation. `fetch(jx, jy)` returns the residual at the
// wrapped grid position if it is in contact and 0 otherwise; every tap is
// summed UNCONDITIONALLY (0.0 substituted for an out-of-contact/out-of-set
// neighbour, never skipped) so both apply paths perform the same tap-sum
// reduction. The two paths' rounding can still differ slightly downstream
// (see the contact-mean subtraction in apply_full/apply_blocked and gate
// S3's R5 tolerances) -- that is expected and does not require pinning this
// loop's codegen or accumulation order.
template <class Fetch>
double tap_sum(const std::vector<int>& dx, const std::vector<int>& dy,
                      const std::vector<double>& w, int ix, int iy,
                      Fetch fetch) {
    double acc = 0.0;
    const int T = static_cast<int>(w.size());
    for (int t = 0; t < T; ++t) acc += w[t] * fetch(ix + dx[t], iy + dy[t]);
    return acc;
}

template <class S>
void apply_full(int Ns, const std::vector<int>& dx, const std::vector<int>& dy,
                const std::vector<double>& w,
                const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
                const std::vector<std::uint8_t>& contact,
                Eigen::Matrix<S, Eigen::Dynamic, 1>& z) {
    const int N = Ns * Ns;
    if (z.size() != N) z.resize(N);

    double zsum = 0.0;
    long nc = 0;
#pragma omp parallel for schedule(static) reduction(+ : zsum, nc)
    for (int iy = 0; iy < Ns; ++iy) {
        for (int ix = 0; ix < Ns; ++ix) {
            const std::ptrdiff_t i =
                static_cast<std::ptrdiff_t>(iy) * Ns + ix;
            if (!contact[i]) { z(i) = S(0); continue; }
            const double acc = tap_sum(
                dx, dy, w, ix, iy, [&](int jx, int jy) -> double {
                    if (jx < 0) jx += Ns; else if (jx >= Ns) jx -= Ns;
                    if (jy < 0) jy += Ns; else if (jy >= Ns) jy -= Ns;
                    const std::ptrdiff_t j =
                        static_cast<std::ptrdiff_t>(jy) * Ns + jx;
                    return contact[j] ? static_cast<double>(g(j)) : 0.0;
                });
            z(i) = static_cast<S>(acc);
            zsum += acc;
            ++nc;
        }
    }
    if (nc) {
        const S zmean = static_cast<S>(zsum / static_cast<double>(nc));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
}

// Compressed apply. Each occupied block gathers an (ls+2R)^2 tile -- its own
// elements plus an R-cell halo from the eight neighbouring blocks, zero where
// a neighbour block is unoccupied or the element is out of contact -- then
// applies the taps over the ls^2 interior.
//
// Zero is the CORRECT value for an element outside the candidate set, not an
// approximation: on the restricted path contact is a subset of the candidate
// set, so such an element is provably a zero of the residual. This is the
// same argument that makes the masked H2 matvec exact.
template <class S>
void apply_blocked(int R, const std::vector<int>& dx,
                   const std::vector<int>& dy, const std::vector<double>& w,
                   const Eigen::Matrix<S, Eigen::Dynamic, 1>& gc,
                   const std::vector<std::uint8_t>& contact_c,
                   const StencilBlockLayout& L,
                   Eigen::Matrix<S, Eigen::Dynamic, 1>& zc) {
    const int ls = L.ls, nl = L.nl(), Ns = L.Ns, ls2 = ls * ls;
    const int tile = ls + 2 * R;
    const std::ptrdiff_t S_total =
        static_cast<std::ptrdiff_t>(L.nslots) * ls2;
    if (zc.size() != S_total) zc.resize(S_total);

    double zsum = 0.0;
    long nc = 0;
#pragma omp parallel reduction(+ : zsum, nc)
    {
        std::vector<double> buf(static_cast<std::size_t>(tile) * tile);
#pragma omp for schedule(static)
        for (int s = 0; s < L.nslots; ++s) {
            const int bx = L.slot_bx[s], by = L.slot_by[s];
            for (int ty = 0; ty < tile; ++ty) {
                int gy = by * ls - R + ty;
                gy = (gy % Ns + Ns) % Ns;
                const int nby = gy / ls, wy = gy % ls;
                for (int tx = 0; tx < tile; ++tx) {
                    int gx = bx * ls - R + tx;
                    gx = (gx % Ns + Ns) % Ns;
                    const int nbx = gx / ls, wx = gx % ls;
                    const int ss =
                        L.block_slot[static_cast<std::size_t>(nby) * nl + nbx];
                    double v = 0.0;
                    if (ss >= 0) {
                        const std::ptrdiff_t k =
                            static_cast<std::ptrdiff_t>(ss) * ls2 + wy * ls + wx;
                        if (contact_c[k]) v = static_cast<double>(gc(k));
                    }
                    buf[static_cast<std::size_t>(ty) * tile + tx] = v;
                }
            }
            // Reuse the exact same accumulation statement as apply_full (via
            // tap_sum) so the two paths perform identical floating-point
            // operations in identical order -- bit-for-bit tap sums, not
            // merely numerically close ones.
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) { zc(k) = S(0); continue; }
                    const double acc = tap_sum(
                        dx, dy, w, lx + R, ly + R,
                        [&](int jx, int jy) -> double {
                            return buf[static_cast<std::size_t>(jy) * tile +
                                       jx];
                        });
                    zc(k) = static_cast<S>(acc);
                    zsum += acc;
                    ++nc;
                }
        }
    }
    if (nc) {
        const S zmean = static_cast<S>(zsum / static_cast<double>(nc));
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < S_total; ++k)
            if (contact_c[k]) zc(k) -= zmean;
    }
}

} // namespace

StencilPreconditioner::StencilPreconditioner(int Ns, int radius)
    : Ns_(Ns), R_(radius) {
    if (radius < 1) throw std::invalid_argument("stencil radius must be >= 1");
    if (2 * radius + 1 > Ns)
        throw std::invalid_argument("stencil radius too large for the grid");

    // The kernel shape is grid-independent from ~Ns=512 (spec §2.1); use the
    // level's own Ns below that, so coarse levels are exact rather than
    // approximated by the large-Ns limit.
    const int Nw = (Ns < 512) ? Ns : 512;
    const std::vector<double> w = kernel_grid(Nw);

    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            const int wx = (dx % Nw + Nw) % Nw;
            const int wy = (dy % Nw + Nw) % Nw;
            dx_.push_back(dx);
            dy_.push_back(dy);
            w_.push_back(w[static_cast<std::size_t>(wy) * Nw + wx]);
        }
}

void StencilPreconditioner::apply_into(
    const Eigen::VectorXd& g, const std::vector<std::uint8_t>& contact,
    Eigen::VectorXd& z) const {
    apply_full<double>(Ns_, dx_, dy_, w_, g, contact, z);
}

void StencilPreconditioner::apply_single_into(
    const Eigen::VectorXf& g, const std::vector<std::uint8_t>& contact,
    Eigen::VectorXf& z) const {
    apply_full<float>(Ns_, dx_, dy_, w_, g, contact, z);
}

void StencilPreconditioner::apply_into_blocked(
    const Eigen::VectorXd& gc, const std::vector<std::uint8_t>& contact_c,
    const StencilBlockLayout& layout, Eigen::VectorXd& zc) const {
    if (R_ > layout.ls)
        throw std::invalid_argument(
            "stencil radius exceeds the block side: the halo would reach "
            "past the immediate neighbour blocks");
    apply_blocked<double>(R_, dx_, dy_, w_, gc, contact_c, layout, zc);
}

void StencilPreconditioner::apply_single_into_blocked(
    const Eigen::VectorXf& gc, const std::vector<std::uint8_t>& contact_c,
    const StencilBlockLayout& layout, Eigen::VectorXf& zc) const {
    if (R_ > layout.ls)
        throw std::invalid_argument(
            "stencil radius exceeds the block side: the halo would reach "
            "past the immediate neighbour blocks");
    apply_blocked<float>(R_, dx_, dy_, w_, gc, contact_c, layout, zc);
}

} // namespace hmc
