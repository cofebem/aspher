# Sparse Stencil Preconditioner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-grid FFT application of the `|k|` preconditioner with a 13-tap real-space stencil evaluated only on the contact set, turning the last O(N log N) component of an otherwise O(N_c) solve into O(N_c).

**Architecture:** A new `StencilPreconditioner` mirrors `FourierPreconditioner`'s interface and is selected by default; the FFT class stays untouched as an equivalence reference and escape hatch. Two apply paths: a full-grid masked loop for the standard solve, and a leaf-tile halo gather over the compressed slot-blocked state for the active-set solve. Underneath both, a measured cost gate can disable the preconditioner for a level when its apply costs more than the iterations it saves.

**Tech Stack:** C++17, Eigen 3, OpenMP, pocketfft (bundled) / FFTW3 (opt-in), pybind11, CTest, Python 3.12 + numpy.

**Spec:** `doc/specs/2026-09-09-stencil-preconditioner-design.md` (approved 2026-09-09). Read it before Task 1 — §2 is the mathematical justification, §3 the measurements that chose R=2, §4.4 the halo-gather argument.

## Global Constraints

- **Build**: `conda activate fenicsx-env`; `-DCMAKE_CXX_COMPILER=/usr/bin/g++`; `-Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)`. Conda's gcc 12 breaks pybind11 headers; use system g++ 11.4. Full command in `CLAUDE.md`.
- **Default radius R = 2** — the 13-point ℓ2 disc `dx²+dy² ≤ 4`. Not 5×5, not 3×3.
- **`R ≤ leaf_side` and `2R+1 ≤ Ns`** are preconditions, asserted, never assumed.
- **Weights normalised to `w(0,0) = 1`.** The overall scale of `M⁻¹` is irrelevant to CG and the solver is scale-invariant by A02 — do not attempt to reproduce the FFT path's absolute scale.
- **Tap sums accumulate in `double` even when `Real=float`.** House rule from the 2026-07 precision pass; 13 terms, so it is free.
- **All taps are summed unconditionally**, substituting `0` for a neighbour that is out of contact or out of the candidate set. Do not `continue` past them — the unconditional form is what makes the two apply paths bit-for-bit identical.
- **`FourierPreconditioner` is not modified.** Not renamed, not refactored, not "cleaned up".
- **Contact-mean removal semantics are unchanged**: mean over the contact set, subtracted from the contact set, accumulated in `double`.
- **`pybind11/*.h` includes must precede all project headers** in `python/bindings.cpp` (existing quirk; see `CLAUDE.md`).
- Tests are plain `main()` returning 0/1 with a `CHECK` macro — follow `tests/test_precond.cpp`. No test framework.

---

### Task 1: `StencilPreconditioner` — weights and the full-grid apply

Delivers the class, its weight construction, and the masked full-grid apply in
both precisions. Gates S1 and S2 land here.

**Files:**
- Create: `include/stencil_precond.hpp`
- Create: `src/stencil_precond.cpp`
- Create: `tests/test_stencil.cpp`
- Modify: `CMakeLists.txt:71` (source list), `CMakeLists.txt:102` (test list)

**Interfaces:**
- Consumes: `hmc::fft::SquareR2C<double>` from `src/fft_engine.hpp` (`bind(n, real*, complex*)`, `fwd()`, `inv()`).
- Produces: `hmc::StencilPreconditioner` with `apply_into`, `apply_single_into`, `tap_dx()`, `tap_dy()`, `tap_w()`, `radius()`, `release_scratch()`; and `hmc::StencilBlockLayout` (fields only — Task 2 uses it).

- [ ] **Step 1: Write the header**

Create `include/stencil_precond.hpp`:

```cpp
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
```

- [ ] **Step 2: Write the failing test for the weights (gate S1) and positivity (gate S2)**

Create `tests/test_stencil.cpp`:

```cpp
#include "stencil_precond.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Universal constants of w = F^-1|k| normalised by w(0,0), converged to six
// figures by Ns=512 (spec §2.1). This table is the ORACLE: the implementation
// derives the weights from an FFT and must reproduce these.
static double oracle(int dx, int dy) {
    const int a = std::abs(dx), b = std::abs(dy);
    const int lo = std::min(a, b), hi = std::max(a, b);
    if (hi == 0 && lo == 0) return 1.0;
    if (hi == 1 && lo == 0) return -0.180875;
    if (hi == 1 && lo == 1) return -0.032860;
    if (hi == 2 && lo == 0) return 0.021189;
    if (hi == 2 && lo == 1) return -0.003912;
    if (hi == 2 && lo == 2) return -0.003198;
    if (hi == 3 && lo == 0) return -0.015473;
    if (hi == 4 && lo == 0) return 0.006277;
    return 0.0 / 0.0; // not tabulated
}

int main() {
    // ── S1: weights match the oracle to six figures, at every Ns >= 512 ────
    for (int Ns : {512, 1024, 2048}) {
        hmc::StencilPreconditioner sp(Ns, 2);
        CHECK(sp.ntaps() == 13); // l2 disc of radius 2
        for (int t = 0; t < sp.ntaps(); ++t) {
            const double ref = oracle(sp.tap_dx()[t], sp.tap_dy()[t]);
            CHECK(ref == ref); // every radius-2 tap must be tabulated
            const double got = sp.tap_w()[t];
            if (std::abs(got - ref) > 5e-6) {
                std::printf("Ns=%d tap (%d,%d): got %.9f want %.9f\n", Ns,
                            sp.tap_dx()[t], sp.tap_dy()[t], got, ref);
                return 1;
            }
        }
        std::printf("S1 Ns=%5d: 13 taps match the oracle\n", Ns);
    }

    // the centre tap must be exactly 1 (the normalisation)
    {
        hmc::StencilPreconditioner sp(1024, 2);
        for (int t = 0; t < sp.ntaps(); ++t)
            if (sp.tap_dx()[t] == 0 && sp.tap_dy()[t] == 0)
                CHECK(sp.tap_w()[t] == 1.0);
    }

    // ── S2: the truncated symbol stays strictly positive off DC ────────────
    // Direct DFT of the tap set: what_R(k) = sum_t w_t cos(2pi k.d/Ns) (w is
    // real and even, so the sine part cancels). If this went negative the
    // preconditioner would not be SPD and the CG theory would not apply.
    for (int R : {1, 2, 4}) {
        const int Ns = 64;
        hmc::StencilPreconditioner sp(Ns, R);
        double mn = 1e300, mx = -1e300;
        for (int ky = 0; ky < Ns; ++ky)
            for (int kx = 0; kx < Ns; ++kx) {
                if (kx == 0 && ky == 0) continue; // DC is zeroed by design
                double s = 0.0;
                for (int t = 0; t < sp.ntaps(); ++t)
                    s += sp.tap_w()[t] *
                         std::cos(2.0 * M_PI *
                                  (double(kx) * sp.tap_dx()[t] +
                                   double(ky) * sp.tap_dy()[t]) / Ns);
                mn = std::min(mn, s);
                mx = std::max(mx, s);
            }
        std::printf("S2 R=%d: symbol in [%.5f, %.5f], min/max %.5f\n", R, mn,
                    mx, mn / mx);
        CHECK(mn > 0.0);
    }

    std::printf("test_stencil: OK\n");
    return 0;
}
```

- [ ] **Step 3: Register the source and the test in CMake, then run to verify it fails**

In `CMakeLists.txt`, add `  src/stencil_precond.cpp` to the `aspher_core`
source list immediately after `src/fourier_precond.cpp` (line 71), and add
`stencil` to the `foreach(t ...)` test list (line 102), after `precond`.

Run:
```bash
conda activate fenicsx-env
cmake --build build -j$(nproc)
```
Expected: FAIL — `src/stencil_precond.cpp` does not exist.

- [ ] **Step 4: Implement the weights and the full-grid apply**

Create `src/stencil_precond.cpp`:

```cpp
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
// summed UNCONDITIONALLY so the two apply paths stay bit-for-bit identical.
template <class Fetch>
inline double tap_sum(const std::vector<int>& dx, const std::vector<int>& dy,
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

} // namespace hmc
```

Note: `apply_into_blocked` / `apply_single_into_blocked` are declared but not
yet defined — Task 2 adds them. Comment out their declarations in the header
for this task, or the link will fail; Task 2 restores them.

- [ ] **Step 5: Build and run the test to verify it passes**

Run:
```bash
conda activate fenicsx-env
cmake --build build -j$(nproc) && ./build/test_stencil
```
Expected: PASS. `S1` prints three lines with 13 matching taps; `S2` prints
min/max ≈ 0.160 (R=1), 0.137 (R=2), 0.065 (R=4), all with `mn > 0`.

- [ ] **Step 6: Commit**

```bash
git add include/stencil_precond.hpp src/stencil_precond.cpp \
        tests/test_stencil.cpp CMakeLists.txt
git commit -m "feat(precond): StencilPreconditioner weights + full-grid apply

The |k| symbol is positive-order, so its real-space kernel is short-ranged:
the radius-2 disc holds 83.9% of sum|w| and reproduces the full FFT
preconditioner's iteration count everywhere measured. Weights come from one
FFT on a min(Ns,512) grid and match the universal constants of spec §2.1 to
six figures (gate S1); the truncated symbol stays strictly positive, so
M^-1 remains SPD (gate S2)."
```

---

### Task 2: Compressed apply via leaf-tile halo gather

Delivers the O(N_c) path used by the active-set solve, and gate S3.

**Files:**
- Modify: `include/stencil_precond.hpp` (restore the two blocked declarations)
- Modify: `src/stencil_precond.cpp` (add the blocked apply)
- Modify: `tests/test_stencil.cpp` (add the equivalence test)

**Interfaces:**
- Consumes: `hmc::StencilBlockLayout` and the tap arrays from Task 1.
- Produces: `StencilPreconditioner::apply_into_blocked(gc, contact_c, layout, zc)` and `apply_single_into_blocked(...)`, both taking compressed vectors of `layout.nslots * layout.ls * layout.ls` entries in slot-blocked order.

- [ ] **Step 1: Write the failing equivalence test**

Append to `tests/test_stencil.cpp`, before the final `return 0;`. It builds a
synthetic block layout directly (no H2 dependency), scatters the same data
into a full grid and into compressed form, applies both, and compares.

```cpp
    // ── S3: compressed apply == full-grid apply on the same data ──────────
    // Bit-for-bit on the tap sums. The two paths reduce the contact mean over
    // different traversal orders, so z differs from the tap sums by a single
    // constant per path; subtracting the difference's own mean isolates that.
    {
        const int Ns = 64, ls = 8, R = 2;
        const int nl = Ns / ls, N = Ns * Ns, ls2 = ls * ls;
        hmc::StencilPreconditioner sp(Ns, R);

        // occupied blocks: a diagonal band plus a block on the domain edge,
        // so the periodic wrap is genuinely exercised
        hmc::StencilBlockLayout L;
        L.Ns = Ns; L.ls = ls;
        L.block_slot.assign(static_cast<std::size_t>(nl) * nl, -1);
        for (int by = 0; by < nl; ++by)
            for (int bx = 0; bx < nl; ++bx) {
                const bool keep = (bx == by) || (bx == 0 && by == nl - 1) ||
                                  (bx == nl - 1 && by == 0);
                if (!keep) continue;
                L.block_slot[static_cast<std::size_t>(by) * nl + bx] =
                    static_cast<int>(L.slot_bx.size());
                L.slot_bx.push_back(bx);
                L.slot_by.push_back(by);
            }
        L.nslots = static_cast<int>(L.slot_bx.size());
        CHECK(L.nslots > 2);

        const std::ptrdiff_t S = static_cast<std::ptrdiff_t>(L.nslots) * ls2;
        Eigen::VectorXd gc(S), zc;
        std::vector<std::uint8_t> contact_c(S, 0);
        Eigen::VectorXd gfull = Eigen::VectorXd::Zero(N), zfull;
        std::vector<std::uint8_t> contact(N, 0);

        // deterministic pseudo-random data; roughly half the candidate
        // entries are in contact
        std::uint64_t st = 0x9E3779B97F4A7C15ull;
        auto rnd = [&st]() {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            return static_cast<double>(st >> 11) / 9007199254740992.0;
        };
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const std::ptrdiff_t i =
                        static_cast<std::ptrdiff_t>(gy) * Ns + gx;
                    const double v = rnd() - 0.5;
                    const std::uint8_t c = (rnd() < 0.5) ? 1 : 0;
                    gc(k) = v; contact_c[k] = c;
                    gfull(i) = v; contact[i] = c;
                }

        sp.apply_into(gfull, contact, zfull);
        sp.apply_into_blocked(gc, contact_c, L, zc);

        // gather the difference on the contact set; it must be CONSTANT
        double dmin = 1e300, dmax = -1e300;
        long ncheck = 0;
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) { CHECK(zc(k) == 0.0); continue; }
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const double d =
                        zc(k) - zfull(static_cast<std::ptrdiff_t>(gy) * Ns + gx);
                    dmin = std::min(dmin, d); dmax = std::max(dmax, d);
                    ++ncheck;
                }
        CHECK(ncheck > 100);
        std::printf("S3 double: %ld contact entries, spread of the "
                    "mean offset %.3e\n", ncheck, dmax - dmin);
        CHECK(dmax - dmin == 0.0); // tap sums bit-for-bit identical
        CHECK(std::abs(dmax) < 1e-15);

        // same in float
        Eigen::VectorXf gcf = gc.cast<float>(), zcf;
        Eigen::VectorXf gff = gfull.cast<float>(), zff;
        sp.apply_single_into(gff, contact, zff);
        sp.apply_single_into_blocked(gcf, contact_c, L, zcf);
        float fmin = 1e30f, fmax = -1e30f;
        for (int s = 0; s < L.nslots; ++s)
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) continue;
                    const int gx = L.slot_bx[s] * ls + lx;
                    const int gy = L.slot_by[s] * ls + ly;
                    const float d =
                        zcf(k) - zff(static_cast<std::ptrdiff_t>(gy) * Ns + gx);
                    fmin = std::min(fmin, d); fmax = std::max(fmax, d);
                }
        std::printf("S3 float:  spread of the mean offset %.3e\n",
                    double(fmax - fmin));
        CHECK(fmax - fmin == 0.0f);
    }
```

- [ ] **Step 2: Restore the blocked declarations and run to verify it fails**

Uncomment `apply_into_blocked` and `apply_single_into_blocked` in
`include/stencil_precond.hpp`.

Run:
```bash
conda activate fenicsx-env && cmake --build build -j$(nproc)
```
Expected: FAIL at link — `undefined reference to hmc::StencilPreconditioner::apply_into_blocked`.

- [ ] **Step 3: Implement the blocked apply**

Add to the anonymous namespace in `src/stencil_precond.cpp`, before the
closing `} // namespace`:

```cpp
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
            const int T = static_cast<int>(w.size());
            for (int ly = 0; ly < ls; ++ly)
                for (int lx = 0; lx < ls; ++lx) {
                    const std::ptrdiff_t k =
                        static_cast<std::ptrdiff_t>(s) * ls2 + ly * ls + lx;
                    if (!contact_c[k]) { zc(k) = S(0); continue; }
                    double acc = 0.0;
                    for (int t = 0; t < T; ++t)
                        acc += w[t] *
                               buf[static_cast<std::size_t>(ly + R + dy[t]) *
                                       tile + (lx + R + dx[t])];
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
```

And the two public wrappers, after `apply_single_into`:

```cpp
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
```

- [ ] **Step 4: Build and run to verify it passes**

Run:
```bash
conda activate fenicsx-env
cmake --build build -j$(nproc) && ./build/test_stencil
```
Expected: PASS, with `S3 double: ... spread of the mean offset 0.000e+00` and
the same for float. A nonzero spread means the tap sums differ — the wrap
arithmetic or the tap order is wrong, not a rounding issue.

- [ ] **Step 5: Trace the out-of-range index the spec flags (spec §8)**

While probing the design, a deliberately degenerate preconditioner (a symbol
band-limited to `k_max/m`) aborted the nested/active path with
`Eigen ... DenseCoeffsBase<Derived,1>::operator()(Index)` on a
`Matrix<float,-1,1>` — an out-of-range index, not a convergence failure. The
single-level path merely converged slowly on the same input. The trigger may
be unreachable in practice, but an out-of-range index is never benign, and
this task is the moment to look: the compressed path is where a mis-sized
vector would show up.

Reproduce with a build that has assertions on and a preconditioner that
returns a near-null direction:

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
cmake --build build-dbg -j$(nproc)
```

Then run the active-set path with `StencilPreconditioner(Ns, 1)` whose weights
have been overwritten with zeros in a scratch build, under `gdb --args`, and
read the backtrace.

Outcome is one of three, all acceptable:
- a genuine bounds bug → fix it here, with a regression test, and say so in
  the commit;
- provably unreachable with any SPD preconditioner → record the argument in
  `experiments/precond_probe_20260909.md` under the crash note and move on;
- not reproducible in 30 minutes → stop, record what was tried in that same
  note, and leave the spec's open item open. Do not let this block the task.

- [ ] **Step 6: Commit**

```bash
git add include/stencil_precond.hpp src/stencil_precond.cpp tests/test_stencil.cpp
git commit -m "feat(precond): compressed stencil apply via leaf-tile halo gather

Each occupied block gathers an (ls+2R)^2 tile from itself and its eight
neighbours and applies the taps over the interior -- no hash map, no
grid->slot table, O(1) extra memory, parallel over blocks. Zero for an
element outside the candidate set is exact, not an approximation: contact
is a subset of the candidate set on the restricted path.

Gate S3: tap sums bit-for-bit identical to the full-grid apply in both
precisions, on data including edge-touching blocks so the periodic wrap is
exercised."
```

---

### Task 3: Plumbing — selector, radius, nested solve, bindings

Makes the stencil the default and delivers gates S4 and S5.

**Files:**
- Modify: `include/h2_operator.hpp` (declare `block_layout`)
- Modify: `src/h2_operator.cpp` (define `block_layout`, next to `slot_grid_indices` at line 392)
- Modify: `include/nested_solve.hpp` (`precond_engine`, `precond_radius`)
- Modify: `src/nested_solve.cpp` (construct and dispatch)
- Modify: `python/bindings.cpp` (`"fourier-fft"`, `precond_engine`, `precond_radius`)
- Modify: `tests/test_precond.cpp` (stencil-vs-FFT solve equivalence)

**Interfaces:**
- Consumes: `StencilPreconditioner` and `StencilBlockLayout` from Tasks 1–2; `H2Mask`, `H2Operator::build_mask`, `H2Operator::slot_grid_indices`.
- Produces: `H2Operator::block_layout(const H2Mask&) const -> StencilBlockLayout`; `NestedParams::precond_engine` (`enum class PrecondEngine { stencil, fft }`, default `stencil`) and `NestedParams::precond_radius` (default 2); Python kwargs `precond_engine="stencil"|"fft"` and `precond_radius=2` on `solve_nested`, and `precond="fourier-fft"` on `ContactSolver.solve`.

- [ ] **Step 1: Add `H2Operator::block_layout`**

In `include/h2_operator.hpp`, add `#include "stencil_precond.hpp"` after the
existing includes, and declare next to `slot_grid_indices` (line 186):

```cpp
    // Block layout for the stencil preconditioner's compressed apply: the
    // same slot convention as slot_grid_indices, expressed as block
    // coordinates plus a block->slot map, so the preconditioner can find a
    // slot's neighbours without an N-sized grid->slot table.
    StencilBlockLayout block_layout(const H2Mask& mask) const;
```

In `src/h2_operator.cpp`, after `slot_grid_indices` (ends line 405):

```cpp
StencilBlockLayout H2Operator::block_layout(const H2Mask& mask) const {
    const auto& boxes = tree_.boxes();
    StencilBlockLayout L;
    L.Ns = Ns_;
    L.ls = ls_;
    L.nslots = mask.nslots();
    const int nl = Ns_ / ls_;
    L.slot_bx.resize(L.nslots);
    L.slot_by.resize(L.nslots);
    L.block_slot.assign(static_cast<std::size_t>(nl) * nl, -1);
    for (int s = 0; s < L.nslots; ++s) {
        const H2Box& b = boxes[leaves_[mask.slot_leaf[s]]];
        const int bx = b.ix0 / ls_, by = b.iy0 / ls_;
        L.slot_bx[s] = bx;
        L.slot_by[s] = by;
        L.block_slot[static_cast<std::size_t>(by) * nl + bx] = s;
    }
    return L;
}
```

- [ ] **Step 2: Add the parameters**

In `include/nested_solve.hpp`, inside `struct NestedParams`, after
`bool precond = true;`:

```cpp
    // Which implementation applies the |q| preconditioner.
    //   stencil — a 13-tap real-space disc evaluated only on the contact set,
    //             O(taps·N_c) with no grid allocation (the default).
    //   fft     — the historical full-grid transform, O(N log N) per apply.
    //             Kept as the equivalence reference and an escape hatch; it
    //             was 77–81% of an Ns=16384 active-set run.
    // Both apply the same operator; see
    // doc/specs/2026-09-09-stencil-preconditioner-design.md.
    enum class PrecondEngine { stencil, fft };
    PrecondEngine precond_engine = PrecondEngine::stencil;
    // l2 disc radius of the stencil, in cells. 2 (13 taps) matched or beat
    // the full transform in every regime measured; 1 (5 taps) costs up to 13%
    // more iterations. Must not exceed leaf_side.
    int precond_radius = 2;
```

- [ ] **Step 3: Write the failing solve-equivalence test (gates S4, S5)**

Append to `tests/test_precond.cpp` before the final `return 0;`:

```cpp
    // ── S4/S5: the stencil preconditioner solves the same problem ─────────
    // Same operator, different implementation: the answer must agree to
    // roundoff and the iteration count must stay in a documented band.
    {
        hmc::StencilPreconditioner sp(Ns, 2);
        hmc::Precond pcs = [&sp](const Eigen::VectorXd& g,
                                 const std::vector<std::uint8_t>& contact) {
            Eigen::VectorXd z;
            sp.apply_into(g, contact, z);
            return z;
        };
        auto r4 = hmc::solve_contact(op, g0, p_bar, 1e-10, 5000, true, pcs);
        CHECK(r4.converged);
        const double rel = (r4.pressure - r1.pressure).norm() / r1.pressure.norm();
        std::printf("stencil: %d it (fft %d), pressure rel %.2e, dArea %.2e\n",
                    r4.iterations, r1.iterations, rel,
                    std::abs(r4.contact_fraction - r1.contact_fraction));
        CHECK(rel < 1e-6);                                        // S4
        CHECK(r4.contact_fraction == r1.contact_fraction);        // S4
        CHECK(r4.iterations <= r1.iterations * 6 / 5 + 1);        // S5: +20%
    }
```

Add `#include "stencil_precond.hpp"` to the includes at the top of the file.

Run:
```bash
conda activate fenicsx-env && cmake --build build -j$(nproc) && ./build/test_precond
```
Expected: PASS already (Task 1 delivered `apply_into`) — this test guards
Task 3's plumbing against regressions rather than driving it. If it fails,
stop: the weights or the full-grid apply are wrong, not the plumbing.

- [ ] **Step 4: Wire the engine into `nested_solve.cpp`**

`src/nested_solve.cpp` currently takes `const FourierPreconditioner* fp` in
`active_level` (line 102) and builds one `FourierPreconditioner fp(n)` per
level (line 621). Make both engines available:

1. Add `#include "stencil_precond.hpp"` next to the `fourier_precond.hpp`
   include (line 5).
2. Change `active_level`'s preconditioner parameters from
   `const FourierPreconditioner* fp` to
   `const FourierPreconditioner* fp, const StencilPreconditioner* sp`.
3. Inside `active_level`, find the `PrecondIntoT<Real> pc;` block that reads

```cpp
    PrecondIntoT<Real> pc;
    if (fp) {
        pc = [fp, &gi](const Vec& g, const std::vector<std::uint8_t>& contact,
                       Vec& z) {
            if constexpr (is_double) fp->apply_into_indexed(g, contact, gi, z);
            else fp->apply_single_into_indexed(g, contact, gi, z);
        };
    }
```
   (immediately after the compressed warm start, around line 175) and replace
   it with:

```cpp
    PrecondIntoT<Real> pc;
    if (sp) {
        // O(N_c): no grid-sized allocation, so nothing to release later
        StencilBlockLayout layout = h2.block_layout(mask);
        pc = [sp, layout = std::move(layout)](
                 const Vec& g, const std::vector<std::uint8_t>& contact,
                 Vec& z) {
            if constexpr (is_double) sp->apply_into_blocked(g, contact, layout, z);
            else sp->apply_single_into_blocked(g, contact, layout, z);
        };
    } else if (fp) {
        pc = [fp, &gi](const Vec& g, const std::vector<std::uint8_t>& contact,
                       Vec& z) {
            if constexpr (is_double) fp->apply_into_indexed(g, contact, gi, z);
            else fp->apply_single_into_indexed(g, contact, gi, z);
        };
    }
```

   The layout is captured by value because the candidate set is rebuilt each
   round; recompute it wherever `mask` is rebuilt inside the round loop.

4. At the per-level construction site, replace line 621

```cpp
        FourierPreconditioner fp(n);
```
   with

```cpp
        FourierPreconditioner fp(n);
        // one FFT on a min(n,512) grid per level; the radius is clamped so a
        // small coarse level cannot ask for a stencil wider than itself
        const int r_lvl = std::min(np.precond_radius, std::max(1, n / 4));
        StencilPreconditioner sp(n, r_lvl);
        const bool use_stencil =
            np.precond &&
            np.precond_engine == NestedParams::PrecondEngine::stencil;
```

5. Replace the double preconditioner functor (lines 624–628)

```cpp
        PrecondIntoT<double> pc;
        if (np.precond)
            pc = [&fp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) { fp.apply_into(g, contact, z); };
```
   with

```cpp
        PrecondIntoT<double> pc;
        if (use_stencil)
            pc = [&sp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) { sp.apply_into(g, contact, z); };
        else if (np.precond)
            pc = [&fp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) { fp.apply_into(g, contact, z); };
```

6. Replace the float preconditioner functor (lines 760–766)

```cpp
            PrecondIntoT<float> pcf;
            if (np.precond)
                pcf = [&fp](const Eigen::VectorXf& g,
                            const std::vector<std::uint8_t>& contact,
                            Eigen::VectorXf& z) {
                    fp.apply_single_into(g, contact, z);
                };
```
   with

```cpp
            PrecondIntoT<float> pcf;
            if (use_stencil)
                pcf = [&sp](const Eigen::VectorXf& g,
                            const std::vector<std::uint8_t>& contact,
                            Eigen::VectorXf& z) {
                    sp.apply_single_into(g, contact, z);
                };
            else if (np.precond)
                pcf = [&fp](const Eigen::VectorXf& g,
                            const std::vector<std::uint8_t>& contact,
                            Eigen::VectorXf& z) {
                    fp.apply_single_into(g, contact, z);
                };
```

7. Replace the active-set pointer (line 711)

```cpp
            const FourierPreconditioner* fpp = np.precond ? &fp : nullptr;
```
   with

```cpp
            const StencilPreconditioner* spp = use_stencil ? &sp : nullptr;
            const FourierPreconditioner* fpp =
                (np.precond && !use_stencil) ? &fp : nullptr;
```
   and add `spp` to both `active_level(...)` call sites, immediately after
   `fpp`, matching the parameter order from item 2.

- [ ] **Step 5: Wire the bindings**

In `python/bindings.cpp`:

1. Add `#include "stencil_precond.hpp"` after the `fourier_precond.hpp`
   include (line 12) — after the pybind11 headers, as the file requires.
2. In `PyContactSolver::solve`, replace the selector at lines 187–196

```cpp
        if (precond == "fourier") {
            auto fp = std::make_shared<hmc::FourierPreconditioner>(
                Ns_);
            pc = [fp](const Eigen::VectorXd& g,
                      const std::vector<std::uint8_t>& contact) {
                return fp->apply(g, contact);
            };
        } else if (precond != "none" && !precond.empty()) {
            throw std::invalid_argument("precond must be 'none' or 'fourier'");
        }
```
   with

```cpp
        if (precond == "fourier") {
            // "fourier" now means the stencil: the same operator, applied as
            // a 13-tap real-space disc on the contact set instead of two
            // full-grid transforms
            auto sp = std::make_shared<hmc::StencilPreconditioner>(Ns_, 2);
            pc = [sp](const Eigen::VectorXd& g,
                      const std::vector<std::uint8_t>& contact) {
                Eigen::VectorXd z;
                sp->apply_into(g, contact, z);
                return z;
            };
        } else if (precond == "fourier-fft") {
            auto fp = std::make_shared<hmc::FourierPreconditioner>(Ns_);
            pc = [fp](const Eigen::VectorXd& g,
                      const std::vector<std::uint8_t>& contact) {
                return fp->apply(g, contact);
            };
        } else if (precond != "none" && !precond.empty()) {
            throw std::invalid_argument(
                "precond must be 'none', 'fourier' or 'fourier-fft'");
        }
```
3. In `py_solve_nested` (line 334), add parameters
   `const std::string& precond_engine, int precond_radius` at the end of the
   signature, and after `np.precond = precond;`:

```cpp
    if (precond_engine == "stencil")
        np.precond_engine = hmc::NestedParams::PrecondEngine::stencil;
    else if (precond_engine == "fft")
        np.precond_engine = hmc::NestedParams::PrecondEngine::fft;
    else
        throw std::invalid_argument("precond_engine must be 'stencil' or 'fft'");
    np.precond_radius = precond_radius;
```
4. Register the kwargs at the `m.def("solve_nested", ...)` list (line 785),
   after `py::arg("precond") = true`:

```cpp
          py::arg("precond_engine") = "stencil", py::arg("precond_radius") = 2,
```
   Keep them adjacent to `precond` so the argument order matches the
   signature.

- [ ] **Step 6: Build, run the full suite, and check the two engines agree end to end**

Run:
```bash
conda activate fenicsx-env
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```
Expected: all tests PASS, including `precond`, `stencil`, `active`,
`certification`, `contracts`.

Then the Python-level check:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, "python")
import numpy as np, aspher as hc, rfgen as rf
Ns = 1024
rng = np.random.default_rng(42)
h = rf.selfaffine_field(dim=2, N=Ns, Hurst=0.8, k_low=12.0/Ns, k_high=0.33,
                        plateau=False, noise=True, rng=rng, verbose=False)
h *= 0.02/np.std(h)
gap = (-h).astype(np.float64).ravel()
out = {}
for eng in ("fft", "stencil"):
    r = hc.solve_nested(grid_size=Ns, gap=gap, p_nominal=0.002, coarsest=64,
                        q=4, leaf_side=16, precond=True, precond_engine=eng,
                        tol=1e-8, coarse_tol=1e-4, light_result=True,
                        active_set=True)
    out[eng] = r
    print(f"{eng:8s} {r.status:10s} it={r.iterations:4d} "
          f"area={r.contact_area:.8f} precond={r.timings['precond']:.3f}s "
          f"total={r.timings['total']:.3f}s")
a, b = out["fft"], out["stencil"]
pa, pb = np.asarray(a.pressure), np.asarray(b.pressure)
print("pressure rel-L2:", np.linalg.norm(pb-pa)/np.linalg.norm(pa))
print("area delta     :", abs(b.contact_area - a.contact_area))
EOF
```
Expected: both `converged`, identical contact area, pressure rel-L2 ≤ 1e-6,
iterations within +20%, and the stencil's `precond` time smaller by orders of
magnitude.

- [ ] **Step 7: Commit**

```bash
git add include/h2_operator.hpp src/h2_operator.cpp include/nested_solve.hpp \
        src/nested_solve.cpp python/bindings.cpp tests/test_precond.cpp
git commit -m "feat(precond): make the stencil the default preconditioner

precond=True / precond='fourier' now mean the stencil; precond='fourier-fft'
keeps the transform, and solve_nested gains precond_engine/precond_radius.
H2Operator::block_layout exposes the slot convention as block coordinates so
the compressed apply can find a slot's neighbours without an N-sized
grid->slot table.

Gates S4/S5: same contact area, pressure agreeing to roundoff, iteration
count within the documented +20% band."
```

---

### Task 4: The measured cost gate (safety valve)

The stencil makes this inert — expected ratio ~1e-3 — but it protects the
`fft` engine and any future regime. Spec §4.5.

**Files:**
- Modify: `include/contact_solver.hpp` (`SolveOptions::precond_cost_gate`)
- Modify: `src/contact_solver.cpp` (both solver bodies)
- Modify: `tests/test_precond.cpp` (gate test)

**Interfaces:**
- Consumes: the existing `t_matvec` / `t_precond` accumulators and the `delta` conjugation switch in `solve_contact_impl`.
- Produces: `SolveOptions::precond_cost_gate` (double, default 1.5, `0` disables) and `ContactResult::precond_dropped` (bool).

- [ ] **Step 1: Add the option and the result field**

In `include/contact_solver.hpp`, inside `struct SolveOptions` after
`int stall_limit = 200;`:

```cpp
    // Drop the preconditioner mid-solve when it stops paying for itself:
    // after the first iteration, if one apply costs more than
    // precond_cost_gate matvecs, the remaining iterations run unpreconditioned
    // and the CG direction restarts. Measured (spec §3.1): running without
    // the preconditioner wins exactly when that ratio exceeds m-1, where m is
    // the iteration multiplier, itself 1.8-4.6 and growing with Ns; 1.5 fires
    // on the cases that lose and not on the ones at parity. 0 disables.
    // With the stencil engine the ratio is ~1e-3 and this never fires.
    double precond_cost_gate = 1.5;
```

In `struct ContactResult`, beside the other diagnostics:

```cpp
    // the cost gate fired and the solve finished unpreconditioned
    bool precond_dropped = false;
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_precond.cpp` before the final `return 0;`:

```cpp
    // ── the cost gate fires on a deliberately expensive preconditioner ────
    {
        hmc::FourierPreconditioner slow_fp(Ns);
        hmc::PrecondIntoT<double> slow =
            [&slow_fp](const Eigen::VectorXd& g,
                       const std::vector<std::uint8_t>& contact,
                       Eigen::VectorXd& z) {
                // same operator, applied 40 times: correct, just costly
                for (int rep = 0; rep < 40; ++rep)
                    slow_fp.apply_into(g, contact, z);
            };
        hmc::MatVecIntoT<double> opi = [&S](const Eigen::VectorXd& v,
                                            Eigen::VectorXd& out) {
            out = S * v;
        };
        hmc::SolveOptions go;
        go.tol = 1e-10;
        go.max_iter = 5000;
        auto rg = hmc::solve_contact_impl<double>(opi, g0, p_bar, go, slow,
                                                  nullptr);
        CHECK(rg.converged);
        CHECK(rg.precond_dropped);
        std::printf("cost gate: dropped=%d, %d it, area matches %d\n",
                    int(rg.precond_dropped), rg.iterations,
                    int(rg.contact_fraction == r0.contact_fraction));
        CHECK(std::abs(rg.contact_fraction - r0.contact_fraction) < 1e-3);

        // and does NOT fire when disabled
        go.precond_cost_gate = 0.0;
        auto rn = hmc::solve_contact_impl<double>(opi, g0, p_bar, go, slow,
                                                  nullptr);
        CHECK(rn.converged);
        CHECK(!rn.precond_dropped);
    }
```

Run:
```bash
conda activate fenicsx-env && cmake --build build -j$(nproc) && ./build/test_precond
```
Expected: FAIL — `ContactResult` has no member `precond_dropped`.

- [ ] **Step 3: Implement the gate in both solver bodies**

In `src/contact_solver.cpp`, in `solve_contact_impl` (the `apply_precond`
lambda is at line 199) add just after it:

```cpp
    // Cost gate: after the first preconditioned iteration, compare one apply
    // against one matvec and drop the preconditioner if it is not paying for
    // itself. Restarting the direction (delta = 0) is required -- the
    // conjugacy recurrence is only valid for a fixed M.
    bool precond_on = static_cast<bool>(precond);
    bool precond_dropped = false;
    auto cost_gate = [&]() {
        if (!precond_on || opt.precond_cost_gate <= 0.0) return;
        if (pcount < 1 || mv < 1) return;
        const double per_pc = t_precond / static_cast<double>(pcount);
        const double per_mv = t_matvec / static_cast<double>(mv);
        if (per_mv > 0.0 && per_pc > opt.precond_cost_gate * per_mv) {
            precond_on = false;
            precond_dropped = true;
            delta = Real(0); // restart conjugacy: M changed
        }
    };
```

Change the use site (line 362) from `if (precond)` to `if (precond_on)`, and
call `cost_gate();` immediately after the `apply_precond(g, contact, z);`
line. Set the result field beside the other assignments (line 524):

```cpp
    res.precond_dropped = precond_dropped;
```

Repeat all of this in `solve_contact_active_impl` (its `apply_precond` is at
line 643, its use site at line 809, its result assignments at line 955).

- [ ] **Step 4: Build and run to verify it passes**

Run:
```bash
conda activate fenicsx-env && cmake --build build -j$(nproc) && ./build/test_precond
```
Expected: PASS — the gate fires on the 40× preconditioner, does not fire when
disabled, and the answer is unchanged either way.

- [ ] **Step 5: Confirm the gate stays inert on the default path**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: all PASS. No existing test may start reporting `precond_dropped` —
if `test_active` or `test_certification` changes behaviour, the threshold is
firing where it should not.

- [ ] **Step 6: Commit**

```bash
git add include/contact_solver.hpp src/contact_solver.cpp tests/test_precond.cpp
git commit -m "feat(precond): measured cost gate for the preconditioner

After the first iteration, compare one apply against one matvec and drop the
preconditioner when it costs more than the iterations it saves, restarting
conjugacy because M changed. Threshold 1.5 from spec §3.1: running without
wins exactly when the ratio exceeds m-1, with m measured at 1.8-4.6.

Inert on the stencil path (ratio ~1e-3); it exists to protect the fft engine,
where it is worth 1.95x (f32) / 2.33x (f64) at Ns=16384."
```

---

### Task 5: Benchmarks, including the required 50% occupancy point

**Files:**
- Modify: `bench/harness.py` (variants)
- Create: `doc/bench/2026-09-09-stencil-preconditioner.md`

**Interfaces:**
- Consumes: the `precond_engine` kwarg from Task 3.
- Produces: ledger rows in `data/bench_stencil.jsonl` and a bench note.

- [ ] **Step 1: Add the variants**

**Hazard first.** Task 3 flips what `precond=True` means, so the *existing*
variant names (`h2-f32-active`, `h2-f64-active`, …) silently change meaning:
rows recorded before the flip were FFT-preconditioned, rows after are stencil,
and neither carries `precond_engine` to tell them apart. `analyze.py` pairs by
variant name, so **never pair a pre-flip row against a post-flip row of the
same name.** The A/B below therefore uses new names that state the engine
explicitly on *both* arms.

In `bench/harness.py`, add to `VARIANTS` after the `h2-f64-active-all` entry
(line 116):

```python
    # Stencil vs the historical full-grid transform
    # (doc/specs/2026-09-09-stencil-preconditioner-design.md). Both arms name
    # the engine explicitly: the default flipped, so a bare "h2-f32-active"
    # row means different things before and after that commit.
    "h2-f32-active-stencil": dict(backend="h2", precision="float",
                                  allow_tolerance_relaxation=True,
                                  active_set=True, precond_engine="stencil"),
    "h2-f32-active-fft":     dict(backend="h2", precision="float",
                                  allow_tolerance_relaxation=True,
                                  active_set=True, precond_engine="fft"),
    "h2-f32-active-nopc":    dict(backend="h2", precision="float",
                                  allow_tolerance_relaxation=True,
                                  active_set=True, precond=False),
    "h2-f64-active-stencil": dict(backend="h2", precision="double",
                                  active_set=True, precond_engine="stencil"),
    "h2-f64-active-fft":     dict(backend="h2", precision="double",
                                  active_set=True, precond_engine="fft"),
```

The worker's forwarding allowlist already carries `precond`,
`precond_engine` and `precond_radius` (fixed in `74b7935` — a key set in
`VARIANTS` but absent from that list is silently dropped and the run measures
the default instead, which is exactly how B04's evolving-C arm went wrong).
Confirm from the first ledger row that `solver_args.precond_engine` is
present before trusting any of these numbers.

- [ ] **Step 2: Run the paired A/B at three grid sizes**

Long runs must be launched **outside the agent sandbox** — each sandboxed
shell gets its own PID namespace and everything in it dies when the call ends
(`nohup`, `disown`, `setsid` do not help). From a normal terminal:

```bash
cd /home/vyastrebov/WORK/PROJECTS/Hcontact
OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 setsid nohup \
  python bench/harness.py run --workload rough-H0.8 --ns 1024 4096 \
    --variants h2-f32-active-stencil,h2-f32-active-fft,h2-f32-active-nopc \
    --reps 5 --ledger data/bench_stencil.jsonl \
  > data/bench_stencil.log 2>&1 < /dev/null & disown
```
Then the same for `--ns 16384` with `--reps 1` (it runs behind the preflight
gate; check `bench/harness.py preflight --ns 16384 --variant h2-f32-active`
first).

A killed job leaves **no ledger row**; a job that fails inside the harness
leaves an `oom`/`timeout`/`error` row. That difference is the diagnostic.
`ps` inside the sandbox cannot see a host job — check the log's mtime and the
ledger.

- [ ] **Step 3: Run the occupancy sweep, including the required 50% point**

The spec makes a 50%-contact measurement a **gate, not future work** —
§3.3 stops at 24.4% and near-full contact is where `ŵ_R` differs most from
`|k|`. Use the B04 load dial (`workload@p_bar`):

```bash
OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 setsid nohup \
  python bench/harness.py run \
    --workload rough-H0.8@0.002 rough-H0.8@0.05 rough-H0.8@0.5 \
    --ns 1024 --variants h2-f32-active-stencil,h2-f32-active-fft \
    --reps 5 --ledger data/bench_stencil.jsonl \
  > data/bench_stencil_occ.log 2>&1 < /dev/null & disown
```
Confirm from the rows that the achieved contact fraction of the `@0.5` arm is
≥ 0.4; if it is not, raise the load until it is, and record the value used.

- [ ] **Step 4: Analyse and write the bench note**

```bash
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil \
    --metric wall_cold_s --ledger data/bench_stencil.jsonl
python bench/analyze.py --pair h2-f32-active-fft,h2-f32-active-stencil \
    --metric peak_rss_gib --ledger data/bench_stencil.jsonl --target 0.0
```

Write `doc/bench/2026-09-09-stencil-preconditioner.md` following the shape of
`doc/bench/2026-09-09-a11-all-levels-restriction.md`: conditions, the paired
table with medians and bootstrap CIs, the per-phase breakdown showing the
preconditioner's share collapsing, the occupancy sweep with the 50% point
called out explicitly, memory, and an explicit promotion verdict against the
plan's rule (≥10% median for a new default, ≥5 paired samples, no case
regressing beyond 5%).

**If the 50% point regresses beyond 5%**, do not flip the default silently:
report it and propose an occupancy gate for the engine choice, the way B04
did for the active-set restriction.

- [ ] **Step 5: Commit**

```bash
git add bench/harness.py doc/bench/2026-09-09-stencil-preconditioner.md \
        data/bench_stencil.jsonl
git commit -m "bench: stencil vs FFT preconditioner, paired A/B + occupancy sweep

Includes the 50%-contact point the spec required before the default flip."
```

---

### Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (PCG section bullet, validated-numbers table)
- Modify: `doc/theory/pcg.tex` (§ "Spectral Preconditioning and Finite-Precision Implementation")
- Modify: `doc/specs/2026-09-08-accuracy-efficiency-improvements.md` (strike A13's coarse space)

- [ ] **Step 1: Update `CLAUDE.md`**

In the PCG section, amend the **Spectral preconditioner** bullet to say the
operator is now applied as a 13-tap real-space stencil on the contact set,
that `precond="fourier-fft"` selects the transform, and that the kernel is
short-ranged because `|k|` is positive-order. Add measured rows to the
validated-numbers table: the stencil-vs-FFT wall and iteration comparison at
Ns=16384, the per-apply cost collapse, and the memory saved.

Use the numbers the Task 5 ledger actually produced, not the spec's
projections. If they differ from the projection, say so — a projection that
did not hold is a finding.

- [ ] **Step 2: Update `doc/theory/pcg.tex`**

Add a subsection to § "Spectral Preconditioning and Finite-Precision
Implementation" covering: why a positive-order symbol has a short-ranged
kernel; the truncation and its retained mass; the proof that restricting to
the contact set is exact rather than approximate; the positivity of `ŵ_R`;
and the measured iteration counts. Compile to check:

```bash
cd doc/theory && pdflatex pcg.tex && bibtex pcg && pdflatex pcg.tex && pdflatex pcg.tex
```

- [ ] **Step 3: Strike A13's coarse space from the roadmap**

In `doc/specs/2026-09-08-accuracy-efficiency-improvements.md`, mark A13's
coarse-space half as a measured no-go, citing spec §3.2 (band-limiting to
`k_max/2` turns 27 iterations into 3985 — worse than no preconditioner), and
note that the sparse half is delivered by the stencil preconditioner. Approved
2026-09-09.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md doc/theory/pcg.tex \
        doc/specs/2026-09-08-accuracy-efficiency-improvements.md
git commit -m "docs: stencil preconditioner; A13 coarse space recorded as a no-go"
```

---

## Gate summary

| id | where | criterion |
|---|---|---|
| S1 | Task 1 | weights match the §2.1 oracle to 6 figures, Ns ≥ 512 |
| S2 | Task 1 | `ŵ_R` > 0 off DC for R ∈ {1,2,4} |
| S3 | Task 2 | compressed apply ≡ full-grid apply, tap sums bit-for-bit, both precisions, wrap exercised |
| S4 | Task 3 | contact area identical, pressure rel-L2 ≤ 1e-6 vs the FFT engine |
| S5 | Task 3 | iteration count within +20% of the FFT engine |
| S6 | Tasks 3–4 | `ctest` green, `precond_dropped` false everywhere on the default path |
| B | Task 5 | paired A/B ≥ 10% median at ≥ 5 samples, no case worse than 5%, **including a ≥40% contact point** |
