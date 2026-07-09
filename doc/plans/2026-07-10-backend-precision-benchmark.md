# Backend × Precision × Ns Benchmark Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep the rough-contact problem across `backend ∈ {h2, fft}` ×
`precision ∈ {double, float}` × `Ns ∈ {256..16384}` (28 cases), recording
wall time, peak memory, and PCG convergence, then produce plots and a
written summary.

**Architecture:** A small, opt-in, off-by-default C++ hook adds
per-iteration error history to `ContactResult`/`solve_nested` (Tasks 1-3).
Two new Python scripts do the actual sweep: a dual-mode worker/orchestrator
script that isolates each case in its own subprocess (clean peak RSS,
crash/OOM containment, resumability) and an analysis script that turns the
collected JSONL into plots + a summary (Tasks 4-6).

**Tech Stack:** C++17 (Eigen, OpenMP), pybind11, Python 3.12 (numpy,
matplotlib), CMake/ctest, conda env `fenicsx-env` for all Python/ctest
work, system `/usr/bin/g++` for cmake per this repo's build notes.

## Global Constraints

- Build with `conda activate fenicsx-env`, `cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++
  -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)`,
  per `CLAUDE.md`.
- The built module `aspher.cpython-312-*.so` must be imported from `python/`
  (`sys.path.insert(0, ".../python")`); `import hmatrix_contact` also works
  as an alias.
- Both `build/` and `build-fftw/` write the same `python/aspher*.so` —
  always rebuild the intended engine (bundled pocketfft; do not touch
  `-DASPHER_USE_FFTW`) **last**, and this plan never touches `build-fftw/`.
- This machine has 20 cores / 31 GiB RAM (co-tenancy caveat: wall-clock
  timings are noisy ±30%; only the study's own min-of-N/min-of-3 policy
  applies, no additional profiling).
- Spec of record: `doc/specs/2026-07-10-backend-precision-benchmark-design.md`.
  Every setting below (surface params, solver settings, q/leaf_side table,
  repetition policy, output schema) is copied from that spec — do not
  re-derive or second-guess it here.

---

## Task 1: Core solver — opt-in per-iteration error history

**Files:**
- Modify: `include/contact_solver.hpp`
- Modify: `src/contact_solver.cpp`
- Modify: `tests/test_contact.cpp`

**Interfaces:**
- Consumes: nothing new (extends the existing `ContactResult`,
  `solve_contact_impl<Real>`, `solve_contact`).
- Produces: `ContactResult::error_history` (`std::vector<double>`, empty
  unless requested); `solve_contact_impl<Real>(..., bool light = false,
  bool record_history = false)`; `solve_contact(..., bool light = false,
  bool record_history = false)`. Task 2 consumes both of these.

- [ ] **Step 1: Add the `error_history` field to `ContactResult`**

In `include/contact_solver.hpp`, change:

```cpp
struct ContactResult {
    Eigen::VectorXd pressure;
    Eigen::VectorXd displacement; // u = S p
    Eigen::VectorXd gap;          // u + g0 - approach (>= 0, = 0 in contact)
    double approach = 0.0;        // rigid-body shift (mean gap over contact)
    double objective = 0.0;       // W = 1/2 p.u + p.g0
    double error = 0.0;
    int iterations = 0;
    bool converged = false;
    double contact_fraction = 0.0;
    double mean_pressure = 0.0;
};
```

to:

```cpp
struct ContactResult {
    Eigen::VectorXd pressure;
    Eigen::VectorXd displacement; // u = S p
    Eigen::VectorXd gap;          // u + g0 - approach (>= 0, = 0 in contact)
    double approach = 0.0;        // rigid-body shift (mean gap over contact)
    double objective = 0.0;       // W = 1/2 p.u + p.g0
    double error = 0.0;
    int iterations = 0;
    bool converged = false;
    double contact_fraction = 0.0;
    double mean_pressure = 0.0;
    std::vector<double> error_history; // per-iteration complementarity error;
                                        // empty unless record_history requested
};
```

- [ ] **Step 2: Add `record_history` parameters to the declarations**

In `include/contact_solver.hpp`, change:

```cpp
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light = false);
```

to:

```cpp
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light = false,
                                 bool record_history = false);
```

and change:

```cpp
ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000, bool use_pr = true,
                            const Precond& precond = {},
                            const Eigen::VectorXd* p_init = nullptr,
                            bool light = false);
```

to:

```cpp
ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000, bool use_pr = true,
                            const Precond& precond = {},
                            const Eigen::VectorXd* p_init = nullptr,
                            bool light = false, bool record_history = false);
```

- [ ] **Step 3: Thread `record_history` through the implementation**

In `src/contact_solver.cpp`, change the definition signature:

```cpp
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light) {
```

to:

```cpp
template <class Real>
ContactResult solve_contact_impl(const MatVecIntoT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondIntoT<Real>& precond,
                                 const VecT<Real>* p_init, bool light,
                                 bool record_history) {
```

Then, right where `res.error` is computed (the line
`res.error = e / (static_cast<double>(P_total) * static_cast<double>(g_scale));`,
immediately before the `if (res.error < static_cast<double>(tol))` check),
add one line so the history captures every iteration including the
converged one:

```cpp
        res.error = e / (static_cast<double>(P_total) * static_cast<double>(g_scale));
        if (record_history) res.error_history.push_back(res.error);
        if (res.error < static_cast<double>(tol)) {
            res.converged = true;
            break;
        }
```

Update the two explicit instantiation declarations at the bottom of the
file:

```cpp
template ContactResult solve_contact_impl<double>(
    const MatVecIntoT<double>&, const VecT<double>&, double, double, int, bool,
    const PrecondIntoT<double>&, const VecT<double>*, bool);
template ContactResult solve_contact_impl<float>(
    const MatVecIntoT<float>&, const VecT<float>&, float, float, int, bool,
    const PrecondIntoT<float>&, const VecT<float>*, bool);
```

to:

```cpp
template ContactResult solve_contact_impl<double>(
    const MatVecIntoT<double>&, const VecT<double>&, double, double, int, bool,
    const PrecondIntoT<double>&, const VecT<double>*, bool, bool);
template ContactResult solve_contact_impl<float>(
    const MatVecIntoT<float>&, const VecT<float>&, float, float, int, bool,
    const PrecondIntoT<float>&, const VecT<float>*, bool, bool);
```

And thread the flag through the `solve_contact` wrapper:

```cpp
ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light) {
    MatVecIntoT<double> Si = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S(x);
    };
    PrecondIntoT<double> pi;
    if (precond)
        pi = [&precond](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact,
                        Eigen::VectorXd& z) { z = precond(g, contact); };
    return solve_contact_impl<double>(Si, g0, p_bar, tol, max_iter, use_pr,
                                      pi, p_init, light);
}
```

to:

```cpp
ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light, bool record_history) {
    MatVecIntoT<double> Si = [&S](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y = S(x);
    };
    PrecondIntoT<double> pi;
    if (precond)
        pi = [&precond](const Eigen::VectorXd& g,
                        const std::vector<std::uint8_t>& contact,
                        Eigen::VectorXd& z) { z = precond(g, contact); };
    return solve_contact_impl<double>(Si, g0, p_bar, tol, max_iter, use_pr,
                                      pi, p_init, light, record_history);
}
```

- [ ] **Step 4: Add the test case**

In `tests/test_contact.cpp`, after the existing block:

```cpp
    // dense and H-matrix solutions must agree
    const double pdiff = (rd.pressure - rh.pressure).norm() / rd.pressure.norm();
    std::printf("  dense vs H pressure rel L2 diff: %.2e\n", pdiff);
    CHECK(pdiff < 1e-3);
    CHECK(std::abs(rd.objective / rh.objective - 1.0) < 1e-6);

    std::printf("test_contact: all checks passed\n");
    return 0;
}
```

insert a new block before `std::printf("test_contact: all checks passed\n");`:

```cpp
    // opt-in convergence history: off by default (no cost, no data)
    CHECK(rd.error_history.empty());

    // with record_history=true, the trace is non-empty and its last entry
    // matches the reported final error (see the design spec for why this
    // is checked instead of size() == iterations)
    const auto rd_hist = hmc::solve_contact(dense_op, g0, p_bar, 1e-10, 4000,
                                            true, hmc::Precond{}, nullptr,
                                            false, true);
    CHECK(!rd_hist.error_history.empty());
    CHECK(std::abs(rd_hist.error_history.back() - rd_hist.error) < 1e-12);
    std::printf("  error_history: %zu entries, last=%.3e\n",
                rd_hist.error_history.size(), rd_hist.error_history.back());

    std::printf("test_contact: all checks passed\n");
    return 0;
}
```

- [ ] **Step 5: Build and run the test**

```bash
source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env
cmake --build build -j$(nproc) --target test_contact
build/test_contact
```

Expected: `test_contact: all checks passed` printed, including the new
`error_history: N entries, last=...` line, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add include/contact_solver.hpp src/contact_solver.cpp tests/test_contact.cpp
git commit -m "$(cat <<'EOF'
feat(solver): opt-in per-iteration error history on ContactResult

Small off-by-default hook (one push_back where res.error is already
computed) for the backend/precision benchmark study; zero cost when
record_history=false.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Thread the flag through the nested (cascadic) solve

**Files:**
- Modify: `include/nested_solve.hpp`
- Modify: `src/nested_solve.cpp`

**Interfaces:**
- Consumes: `solve_contact_impl<Real>(..., bool light, bool record_history)`
  from Task 1 (exact positional order: `..., p_init, light, record_history`).
- Produces: `NestedParams::record_error_history` (bool, default false);
  `solve_contact_nested` returns a `ContactResult` whose `error_history` is
  populated (finest level only) when that flag is set. Task 3 consumes this
  field name.

- [ ] **Step 1: Add the field to `NestedParams`**

In `include/nested_solve.hpp`, change:

```cpp
struct NestedParams {
    int coarsest = 64;        // coarsest grid side (power of two, divides Ns)
    int q = 6;                // H2 Chebyshev order on every level
    int leaf_side = 8;        // H2 leaf side on every level
    bool precond = true;      // |q| spectral preconditioner per level
    double coarse_tol = 1e-4; // cascadic: looser tolerance on coarse levels
    bool single_precision = false; // run each level's solve in float (~half RAM)
    bool light_result = false;     // skip displacement/gap in the result (~2 N arrays)
    std::string backend = "h2";    // per-level operator: "h2" or "fft"
};
```

to:

```cpp
struct NestedParams {
    int coarsest = 64;        // coarsest grid side (power of two, divides Ns)
    int q = 6;                // H2 Chebyshev order on every level
    int leaf_side = 8;        // H2 leaf side on every level
    bool precond = true;      // |q| spectral preconditioner per level
    double coarse_tol = 1e-4; // cascadic: looser tolerance on coarse levels
    bool single_precision = false; // run each level's solve in float (~half RAM)
    bool light_result = false;     // skip displacement/gap in the result (~2 N arrays)
    std::string backend = "h2";    // per-level operator: "h2" or "fft"
    bool record_error_history = false; // finest-level per-iteration error trace
};
```

- [ ] **Step 2: Pass `record_history` into both finest-level solve calls**

In `src/nested_solve.cpp`, change:

```cpp
        const bool finest = (li + 1 == levels.size());
        double lvl_tol = finest ? tol : np.coarse_tol;
        // float arithmetic cannot drive the complementarity error below ~1e-6,
        // so clamp the requested tolerance to a reachable floor in that mode.
        if (np.single_precision) lvl_tol = std::max(lvl_tol, 2e-6);
        // coarse levels only need the pressure (for prolongation), so drop their
        // displacement/gap unconditionally; the finest honours light_result.
        const bool light = finest ? np.light_result : true;
```

to:

```cpp
        const bool finest = (li + 1 == levels.size());
        double lvl_tol = finest ? tol : np.coarse_tol;
        // float arithmetic cannot drive the complementarity error below ~1e-6,
        // so clamp the requested tolerance to a reachable floor in that mode.
        if (np.single_precision) lvl_tol = std::max(lvl_tol, 2e-6);
        // coarse levels only need the pressure (for prolongation), so drop their
        // displacement/gap unconditionally; the finest honours light_result.
        const bool light = finest ? np.light_result : true;
        // only the finest level's trace is meaningful (iterations is also
        // finest-level-only); coarse levels never record history.
        const bool record_history = finest && np.record_error_history;
```

Then change the two `solve_contact_impl` call sites:

```cpp
            res = solve_contact_impl<float>(
                mvf, g0f, static_cast<float>(p_bar), static_cast<float>(lvl_tol),
                max_iter, use_pr, pcf, have_init ? &p0f : nullptr, light);
        } else {
            res = solve_contact_impl<double>(
                mv, gap[li], p_bar, lvl_tol, max_iter, use_pr, pc,
                have_init ? &p_init : nullptr, light);
            gap[li].resize(0);
        }
```

to:

```cpp
            res = solve_contact_impl<float>(
                mvf, g0f, static_cast<float>(p_bar), static_cast<float>(lvl_tol),
                max_iter, use_pr, pcf, have_init ? &p0f : nullptr, light,
                record_history);
        } else {
            res = solve_contact_impl<double>(
                mv, gap[li], p_bar, lvl_tol, max_iter, use_pr, pc,
                have_init ? &p_init : nullptr, light, record_history);
            gap[li].resize(0);
        }
```

- [ ] **Step 3: Build (no new test yet — Task 3's Python test exercises this)**

```bash
cmake --build build -j$(nproc) --target aspher_core
```

Expected: clean build, no errors (this target has no `main`, so nothing to
run yet; the full ctest suite is re-checked at the end of Task 3).

- [ ] **Step 4: Commit**

```bash
git add include/nested_solve.hpp src/nested_solve.cpp
git commit -m "$(cat <<'EOF'
feat(nested): thread record_error_history to the finest level

Mirrors the existing finest/light pattern: coarse levels never record,
only the finest level (whose .iterations is already the only one
reported) does.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Expose to Python and verify end-to-end

**Files:**
- Modify: `python/bindings.cpp`
- Modify: `tests/test_nested.py`

**Interfaces:**
- Consumes: `ContactResult::error_history`, `NestedParams::record_error_history`
  from Tasks 1-2.
- Produces: `ContactResult.error_history` (numpy array, Python); `solve_nested(...,
  record_error_history=False)` kwarg. Tasks 4-5 consume this exact kwarg name
  and property name.

- [ ] **Step 1: Expose `error_history` as a read-only property**

In `python/bindings.cpp`, change:

```cpp
        .def_property_readonly("error", [](const PyResult& s) { return s.r.error; })
        .def_property_readonly("converged",
                               [](const PyResult& s) { return s.r.converged; })
```

to:

```cpp
        .def_property_readonly("error", [](const PyResult& s) { return s.r.error; })
        .def_property_readonly(
            "error_history",
            [](const PyResult& s) {
                return py::array_t<double>(s.r.error_history.size(),
                                           s.r.error_history.data());
            })
        .def_property_readonly("converged",
                               [](const PyResult& s) { return s.r.converged; })
```

- [ ] **Step 2: Add the `record_error_history` parameter to `py_solve_nested`**

Change:

```cpp
PyResult py_solve_nested(
    int grid_size,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gap,
    double p_nominal, double domain_size, double E_star, int coarsest, int q,
    int leaf_side, bool precond, double tol, double coarse_tol, int max_iter,
    bool use_pr, bool single_precision, bool light_result,
    const std::string& backend) {
    Eigen::VectorXd g0 = to_vector(gap, grid_size * grid_size);
    hmc::NestedParams np{coarsest, q, leaf_side, precond, coarse_tol,
                         single_precision, light_result, backend};
    PyResult out;
    out.Ns = grid_size;
    {
        py::gil_scoped_release release;
        out.r = hmc::solve_contact_nested(grid_size, domain_size, E_star, g0,
                                          p_nominal, tol, max_iter, use_pr, np);
    }
    return out;
}
```

to:

```cpp
PyResult py_solve_nested(
    int grid_size,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gap,
    double p_nominal, double domain_size, double E_star, int coarsest, int q,
    int leaf_side, bool precond, double tol, double coarse_tol, int max_iter,
    bool use_pr, bool single_precision, bool light_result,
    const std::string& backend, bool record_error_history) {
    Eigen::VectorXd g0 = to_vector(gap, grid_size * grid_size);
    hmc::NestedParams np{coarsest, q, leaf_side, precond, coarse_tol,
                         single_precision, light_result, backend,
                         record_error_history};
    PyResult out;
    out.Ns = grid_size;
    {
        py::gil_scoped_release release;
        out.r = hmc::solve_contact_nested(grid_size, domain_size, E_star, g0,
                                          p_nominal, tol, max_iter, use_pr, np);
    }
    return out;
}
```

- [ ] **Step 3: Add the pybind11 argument default**

Change:

```cpp
    m.def("solve_nested", &py_solve_nested, py::arg("grid_size"),
          py::arg("gap"), py::arg("p_nominal"), py::arg("domain_size") = 1.0,
          py::arg("E_star") = 1.0, py::arg("coarsest") = 64, py::arg("q") = 6,
          py::arg("leaf_side") = 8, py::arg("precond") = true,
          py::arg("tol") = 1e-8, py::arg("coarse_tol") = 1e-4,
          py::arg("max_iter") = 20000, py::arg("use_pr") = true,
          py::arg("single_precision") = false, py::arg("light_result") = false,
          py::arg("backend") = "h2",
          "Single-entry nested-grid (cascadic/FMG) contact solve: builds the "
          "coarse->fine hierarchy and H2 operators internally and warm-starts "
          "each level with the prolonged coarse pressure. grid_size must equal "
          "coarsest * 2^k. Returns a ContactResult. backend='h2' (O(N) memory) "
          "or 'fft' (exact convolution, fastest at Ns<=8192).");
```

to:

```cpp
    m.def("solve_nested", &py_solve_nested, py::arg("grid_size"),
          py::arg("gap"), py::arg("p_nominal"), py::arg("domain_size") = 1.0,
          py::arg("E_star") = 1.0, py::arg("coarsest") = 64, py::arg("q") = 6,
          py::arg("leaf_side") = 8, py::arg("precond") = true,
          py::arg("tol") = 1e-8, py::arg("coarse_tol") = 1e-4,
          py::arg("max_iter") = 20000, py::arg("use_pr") = true,
          py::arg("single_precision") = false, py::arg("light_result") = false,
          py::arg("backend") = "h2", py::arg("record_error_history") = false,
          "Single-entry nested-grid (cascadic/FMG) contact solve: builds the "
          "coarse->fine hierarchy and H2 operators internally and warm-starts "
          "each level with the prolonged coarse pressure. grid_size must equal "
          "coarsest * 2^k. Returns a ContactResult. backend='h2' (O(N) memory) "
          "or 'fft' (exact convolution, fastest at Ns<=8192). "
          "record_error_history=True fills .error_history with the finest "
          "level's per-iteration complementarity error (off by default).");
```

- [ ] **Step 4: Extend the Python nested-solve test**

In `tests/test_nested.py`, change the end of the `for Ns in (256, 512):` loop
body (after the existing `single_precision`/`light_result` block) — i.e.
change:

```python
        print(f"Ns={Ns}: single+light={sp.iterations} it, conv={sp.converged}, "
              f"dArea={d_area_sp:.2e}, relL2={relL2_sp:.1e}, "
              f"disp_none={sp.displacement is None}")
        assert sp.converged           # reaches the float floor via stagnation guard
        assert d_area_sp < 2e-3, d_area_sp
        assert relL2_sp < 1e-3, relL2_sp
        assert sp.displacement is None  # light_result: displacement not stored


if __name__ == "__main__":
```

to:

```python
        print(f"Ns={Ns}: single+light={sp.iterations} it, conv={sp.converged}, "
              f"dArea={d_area_sp:.2e}, relL2={relL2_sp:.1e}, "
              f"disp_none={sp.displacement is None}")
        assert sp.converged           # reaches the float floor via stagnation guard
        assert d_area_sp < 2e-3, d_area_sp
        assert relL2_sp < 1e-3, relL2_sp
        assert sp.displacement is None  # light_result: displacement not stored

        # opt-in convergence history: off by default, populated on request
        assert nest.error_history.size == 0
        hist = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR,
                               coarsest=64, q=6, tol=1e-8,
                               record_error_history=True)
        assert hist.error_history.size > 0
        assert abs(float(hist.error_history[-1]) - hist.error) < 1e-9, (
            hist.error_history[-1], hist.error)
        print(f"Ns={Ns}: error_history has {hist.error_history.size} entries, "
              f"last={hist.error_history[-1]:.3e}")


if __name__ == "__main__":
```

- [ ] **Step 5: Rebuild the Python extension and run the test**

```bash
source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env
cmake --build build -j$(nproc)
ldd python/aspher.cpython-312-*.so | grep -i fftw || echo "no FFTW (bundled pocketfft, expected)"
python tests/test_nested.py
```

Expected: `no FFTW (bundled pocketfft, expected)` (confirms `build/`, not
`build-fftw/`, produced the loaded `.so`), then per-Ns lines ending with
`OK` and no `AssertionError`.

- [ ] **Step 6: Run the full ctest suite to confirm nothing else broke**

```bash
ctest --test-dir build --output-on-failure
```

Expected: all tests (`kernel`, `hmatrix`, `contact`, `h2`, `precond`, `fft`)
pass.

- [ ] **Step 7: Commit**

```bash
git add python/bindings.cpp tests/test_nested.py
git commit -m "$(cat <<'EOF'
feat(python): expose error_history / record_error_history to solve_nested

Completes the opt-in convergence-history hook for the backend/precision
benchmark study.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Benchmark worker + orchestrator script

**Files:**
- Create: `bench_backend_precision_study.py`

**Interfaces:**
- Consumes: `aspher.solve_nested(..., record_error_history=...)` from Task 3;
  `rfgen.selfaffine_field` (existing, see `example_rough_contact.py` usage).
- Produces: `data/backend_precision_study.jsonl` (one JSON object per line,
  schema in the design spec §6). Task 5 consumes this exact path and schema.

- [ ] **Step 1: Write the script**

Create `bench_backend_precision_study.py`:

```python
"""Backend x precision x Ns benchmark/convergence sweep for the rough-contact
problem (see doc/specs/2026-07-10-backend-precision-benchmark-design.md).

Worker mode (runs ONE case in-process, prints one JSON line):
    python bench_backend_precision_study.py --backend h2 --precision double --ns 1024

Orchestrator mode (default; spawns one subprocess per case, resumable):
    python bench_backend_precision_study.py [--max-ns 16384] [--timeout 1800] [--force]
"""
import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NS_ALL = [256, 512, 1024, 2048, 4096, 8192, 16384]
BACKENDS = ["h2", "fft"]
PRECISIONS = ["double", "float"]
RESULTS_PATH = os.path.join(HERE, "data", "backend_precision_study.jsonl")

SEED = 42
HURST = 0.8
RMS = 0.02
K_HIGH = 0.33
P_BAR = 0.002
COARSEST = 64


def h2_settings(Ns):
    return (6, 8) if Ns <= 4096 else (4, 16)


def build_gap(Ns):
    """Rough surface via rfgen (see design spec Sec.1); returns -height,
    ravelled. Frees intermediates and returns large allocations to the OS
    before the solve at big Ns, mirroring example_rough_contact.py."""
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "python"))
    import rfgen as rf

    rng = np.random.default_rng(SEED)
    roughness = rf.selfaffine_field(
        dim=2, N=Ns, Hurst=HURST, k_low=12.0 / Ns, k_high=K_HIGH,
        plateau=False, noise=True, rng=rng, verbose=False,
    )
    roughness *= RMS / np.std(roughness)
    gap0 = (-roughness).astype(np.float64).ravel()
    del roughness
    gc.collect()
    if Ns >= 8192:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    return gap0


def worker(backend, precision, Ns):
    import resource
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "python"))
    import aspher as hc

    gap0 = build_gap(Ns)
    q, leaf_side = h2_settings(Ns) if backend == "h2" else (None, None)
    single = (precision == "float")
    reps = 3 if Ns <= 512 else 1

    kwargs = dict(
        grid_size=Ns, gap=gap0, p_nominal=P_BAR, coarsest=COARSEST,
        precond=True, tol=1e-8, coarse_tol=1e-4, max_iter=20000,
        single_precision=single, light_result=True, backend=backend,
        record_error_history=True,
    )
    if backend == "h2":
        kwargs["q"] = q
        kwargs["leaf_side"] = leaf_side

    times = []
    res = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = hc.solve_nested(**kwargs)
        times.append(time.perf_counter() - t0)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {
        "backend": backend, "precision": precision, "Ns": Ns, "N": Ns * Ns,
        "q": q, "leaf_side": leaf_side,
        "wall_time_s": min(times), "wall_time_all_s": times,
        "rss_gib": rss_kb / 1048576.0,
        "iterations": int(res.iterations), "converged": bool(res.converged),
        "final_error": float(res.error), "contact_area": float(res.contact_area),
        "mean_pressure": float(res.mean_pressure),
        "error_history": [float(v) for v in np.asarray(res.error_history)],
        "status": "ok",
        "seed": SEED, "p_bar": P_BAR, "rms": RMS, "Hurst": HURST,
        "k_low": 12.0 / Ns, "k_high": K_HIGH,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("JSON " + json.dumps(out), flush=True)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((d.get("backend"), d.get("precision"), d.get("Ns")))
    return done


def classify_failure(returncode, stderr):
    if returncode == -9:
        return "oom"
    if "bad_alloc" in stderr or "MemoryError" in stderr or "Killed" in stderr:
        return "oom"
    return "error"


def sweep(timeout, force, max_ns):
    done = set() if force else load_done(RESULTS_PATH)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    cases = [(b, p, n) for n in NS_ALL if n <= max_ns
             for p in PRECISIONS for b in BACKENDS]

    for backend, precision, Ns in cases:
        if (backend, precision, Ns) in done:
            print(f"skip {backend:3s} {precision:6s} Ns={Ns:6d} (already recorded)",
                  flush=True)
            continue
        print(f"running {backend:3s} {precision:6s} Ns={Ns:6d} ...", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--backend", backend, "--precision", precision, "--ns", str(Ns)]
        t0 = time.time()
        record = None
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = next((l for l in r.stdout.splitlines() if l.startswith("JSON ")),
                        None)
            if line is not None and r.returncode == 0:
                record = json.loads(line[5:])
            else:
                record = {
                    "backend": backend, "precision": precision, "Ns": Ns,
                    "N": Ns * Ns, "status": classify_failure(r.returncode, r.stderr),
                    "stderr_tail": r.stderr[-2000:],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
        except subprocess.TimeoutExpired as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            record = {
                "backend": backend, "precision": precision, "Ns": Ns, "N": Ns * Ns,
                "status": "timeout", "stderr_tail": stderr[-2000:],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        dt = time.time() - t0
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  -> status={record.get('status')} wall={dt:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, help="single-case worker mode")
    ap.add_argument("--precision", choices=PRECISIONS)
    ap.add_argument("--ns", type=int)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-ns", type=int, default=16384)
    args = ap.parse_args()
    if args.backend:
        worker(args.backend, args.precision, args.ns)
    else:
        sweep(args.timeout, args.force, args.max_ns)
```

- [ ] **Step 2: Smoke-test worker mode on the cheapest case**

```bash
source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env
python bench_backend_precision_study.py --backend h2 --precision double --ns 256
```

Expected: one line starting with `JSON {...}` containing
`"status": "ok"`, `"Ns": 256`, `"converged": true`, a non-empty
`"error_history"` list, and a `"contact_area"` roughly consistent with the
`rms=0.02`, `p_bar=0.002` regime (order 1e-2–1e-1; exact value isn't
asserted here, just sanity-checked by eye).

- [ ] **Step 3: Smoke-test orchestrator mode + resumability on 2 tiny cases**

```bash
rm -f /tmp/claude-1769/-home-vyastrebov-WORK-PROJECTS-Hcontact/c1e4744c-95b1-4f1d-9438-96f1a3737af1/scratchpad/smoke.jsonl
python - <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bps", "bench_backend_precision_study.py")
bps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bps)
bps.RESULTS_PATH = "/tmp/claude-1769/-home-vyastrebov-WORK-PROJECTS-Hcontact/c1e4744c-95b1-4f1d-9438-96f1a3737af1/scratchpad/smoke.jsonl"
bps.NS_ALL = [256]
bps.sweep(timeout=120, force=False, max_ns=256)
print("--- second call (should skip all) ---")
bps.sweep(timeout=120, force=False, max_ns=256)
EOF
```

Expected: first call runs all 4 (backend, precision) combos at Ns=256 and
writes 4 lines; second call prints `skip ...` for all 4 and appends
nothing (`wc -l` on the file stays at 4).

- [ ] **Step 4: Commit**

```bash
git add bench_backend_precision_study.py
git commit -m "$(cat <<'EOF'
feat(bench): backend x precision x Ns sweep worker/orchestrator

Subprocess-per-case (clean peak RSS via resource.getrusage, crash/OOM
containment) following bench_h2_cputime.py's proven pattern; resumable
via a JSONL results file. Per doc/specs/2026-07-10-backend-precision-
benchmark-design.md.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Analysis / plotting script

**Files:**
- Create: `analyze_backend_precision_study.py`

**Interfaces:**
- Consumes: `data/backend_precision_study.jsonl` (schema from Task 4/spec §6).
- Produces: `doc/backend_precision_study/fig_walltime_vs_ns.png`,
  `fig_memory_vs_ns.png`, `fig_iterations_vs_ns.png`,
  `fig_convergence_curves.png`, `summary.md`.

- [ ] **Step 1: Write the script**

Create `analyze_backend_precision_study.py`:

```python
"""Reads data/backend_precision_study.jsonl and writes plots + a summary
to doc/backend_precision_study/. See
doc/specs/2026-07-10-backend-precision-benchmark-design.md Sec.4/6."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "backend_precision_study.jsonl")
OUT_DIR = os.path.join(HERE, "doc", "backend_precision_study")

COMBOS = [("h2", "double"), ("h2", "float"), ("fft", "double"), ("fft", "float")]
STYLE = {
    ("h2", "double"): dict(color="C0", marker="o", label="H2 double"),
    ("h2", "float"): dict(color="C0", marker="o", linestyle="--", label="H2 float"),
    ("fft", "double"): dict(color="C1", marker="s", label="FFT double"),
    ("fft", "float"): dict(color="C1", marker="s", linestyle="--", label="FFT float"),
}


def load_records():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group(records):
    by_combo = {c: {} for c in COMBOS}
    failed = []
    for r in records:
        combo = (r.get("backend"), r.get("precision"))
        if r.get("status") != "ok":
            failed.append(r)
            continue
        if combo in by_combo:
            by_combo[combo][r["Ns"]] = r
    return by_combo, failed


def plot_metric(by_combo, key, ylabel, title, fname, logy=True):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for combo in COMBOS:
        pts = sorted(by_combo[combo].items())
        if not pts:
            continue
        xs = [ns for ns, _ in pts]
        ys = [rec[key] for _, rec in pts]
        ax.plot(xs, ys, **STYLE[combo])
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Ns")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def plot_convergence(by_combo, ns_list=(1024, 4096)):
    ns_list = [n for n in ns_list if any(by_combo[c].get(n) for c in COMBOS)]
    if not ns_list:
        return
    fig, axes = plt.subplots(1, len(ns_list), figsize=(5.5 * len(ns_list), 4.2),
                             squeeze=False)
    for ax, ns in zip(axes[0], ns_list):
        for combo in COMBOS:
            rec = by_combo[combo].get(ns)
            if rec is None or not rec.get("error_history"):
                continue
            hist = rec["error_history"]
            ax.plot(range(1, len(hist) + 1), hist, **STYLE[combo])
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("complementarity error")
        ax.set_title(f"Ns={ns}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_convergence_curves.png"), dpi=150)
    plt.close(fig)


def write_summary(by_combo, failed):
    lines = ["# Backend x precision x Ns benchmark study — summary", ""]
    lines.append("| backend | precision | Ns | wall[s] | RSS[GiB] | iters | "
                 "converged | contact_area |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for combo in COMBOS:
        for ns, rec in sorted(by_combo[combo].items()):
            lines.append(
                f"| {combo[0]} | {combo[1]} | {ns} | {rec['wall_time_s']:.2f} | "
                f"{rec['rss_gib']:.2f} | {rec['iterations']} | "
                f"{rec['converged']} | {rec['contact_area']:.5f} |")
    if failed:
        lines.append("")
        lines.append("## Failed / incomplete cases")
        for r in failed:
            lines.append(f"- {r.get('backend')} {r.get('precision')} "
                         f"Ns={r.get('Ns')}: status={r.get('status')}")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_records()
    by_combo, failed = group(records)
    plot_metric(by_combo, "wall_time_s", "wall time [s]",
               "Solve wall time vs Ns", "fig_walltime_vs_ns.png")
    plot_metric(by_combo, "rss_gib", "peak RSS [GiB]",
               "Peak memory vs Ns", "fig_memory_vs_ns.png")
    plot_metric(by_combo, "iterations", "PCG iterations",
               "Iteration count vs Ns", "fig_iterations_vs_ns.png", logy=False)
    plot_convergence(by_combo)
    write_summary(by_combo, failed)
    print(f"wrote plots + summary.md to {OUT_DIR}")
    if failed:
        print(f"{len(failed)} case(s) did not complete:")
        for r in failed:
            print(f"  {r.get('backend')} {r.get('precision')} Ns={r.get('Ns')}: "
                  f"{r.get('status')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on the partial data from Task 4's smoke tests**

```bash
mkdir -p data
cp /tmp/claude-1769/-home-vyastrebov-WORK-PROJECTS-Hcontact/c1e4744c-95b1-4f1d-9438-96f1a3737af1/scratchpad/smoke.jsonl data/backend_precision_study.jsonl
python analyze_backend_precision_study.py
ls doc/backend_precision_study/
rm data/backend_precision_study.jsonl  # remove the smoke-test stand-in before the real sweep
```

Expected: `wrote plots + summary.md to .../doc/backend_precision_study`
printed; the 4 PNG files plus `summary.md` exist (the convergence-curves
plot may be a near-empty figure since the smoke data is Ns=256 only —
that's fine, it's just verifying the script runs without crashing on
partial/small data). The final `rm` clears the smoke-test stand-in so
Task 6 starts the real sweep from an empty results file.

- [ ] **Step 3: Commit**

```bash
git add analyze_backend_precision_study.py
git commit -m "$(cat <<'EOF'
feat(bench): analysis/plotting script for the backend/precision study

Reads data/backend_precision_study.jsonl, writes 4 comparison plots and
a summary table to doc/backend_precision_study/. Failed/incomplete
cases are listed, not silently dropped.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run the full sweep and produce the final report

**Files:** none (execution only; reads/writes `data/backend_precision_study.jsonl`
and `doc/backend_precision_study/*`).

**Interfaces:**
- Consumes: Tasks 4 and 5's scripts, unmodified.
- Produces: the final populated `data/backend_precision_study.jsonl` (up to
  28 lines) and the final `doc/backend_precision_study/` plots + summary.

- [ ] **Step 1: Launch the full sweep in the background**

```bash
source /home/users02/vyastrebov/DISTR/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env
cd /home/vyastrebov/WORK/PROJECTS/Hcontact
nohup python bench_backend_precision_study.py > /tmp/claude-1769/-home-vyastrebov-WORK-PROJECTS-Hcontact/c1e4744c-95b1-4f1d-9438-96f1a3737af1/scratchpad/sweep.log 2>&1 &
echo "sweep pid: $!"
```

Expected: a PID is printed; the process runs unattended. This is expected
to take a long time (the 8192/16384 cases alone can run tens of minutes to
hours each per CLAUDE.md's validated numbers) — do not block waiting on it
synchronously; monitor periodically instead (`tail -f` the log, or
`wc -l data/backend_precision_study.jsonl` to see cases complete one by one,
since the orchestrator appends after every case).

- [ ] **Step 2: Wait for completion, then verify no more than the expected failures**

```bash
wc -l data/backend_precision_study.jsonl
grep -c '"status": "ok"' data/backend_precision_study.jsonl
grep -v '"status": "ok"' data/backend_precision_study.jsonl || true
```

Expected: 28 total lines. Every case except possibly `h2 double Ns=16384`
and `fft double Ns=16384` should show `"status": "ok"` (per the design
spec §1, double precision at Ns=16384 may legitimately OOM/timeout — that
is an accepted, documented outcome, not a bug to chase). Any `"status"`
other than `"ok"` at a *smaller* Ns is unexpected and should be
investigated (re-run that single case directly via worker mode with
`--backend/--precision/--ns` to see the real error) before proceeding.

- [ ] **Step 3: Generate the final plots and summary**

```bash
python analyze_backend_precision_study.py
cat doc/backend_precision_study/summary.md
```

Expected: the printed summary table covers all completed cases; any
failures are listed under "Failed / incomplete cases" rather than silently
missing.

- [ ] **Step 4: Commit the results and report**

```bash
git add data/backend_precision_study.jsonl doc/backend_precision_study/
git commit -m "$(cat <<'EOF'
data(bench): full backend x precision x Ns sweep results + report

28-case sweep (h2/fft x double/float x Ns=256..16384) per
doc/specs/2026-07-10-backend-precision-benchmark-design.md. See
doc/backend_precision_study/summary.md for the results table and
doc/backend_precision_study/fig_*.png for the comparison plots.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Report the findings to the user**

Summarize (in the conversation, not a new file): which backend/precision
is fastest and most memory-efficient at each Ns, how iteration counts and
convergence curves compare between h2 and fft and between double and
float, and what happened at Ns=16384 double (if it failed, say how —
OOM/timeout — and note that this was an anticipated, accepted risk from
the design spec, not a defect).
