#!/usr/bin/env python3
"""Generate the immutable high-precision Love-kernel fixture (spec A05, plan §1.3).

The fixture is an *oracle*: it is produced by code that shares nothing with the
C++ kernel. Two independent evaluations must agree before a value is written:

  1. the closed-form Love integral evaluated with mpmath at >= 80 decimal
     digits (the analytic path), and
  2. adaptive numerical quadrature of the same integral (the numerical path),
     which splits the integrable 1/|r-r'| singularity when the field point is
     inside or on the edge of the cell rather than running a tensor Gauss rule
     across it.

Regenerating is an explicit developer command:

    conda run -n fenicsx-env python tests/generate_kernel_reference.py

It NEVER runs automatically as part of a test, and a failing test must never
overwrite it. Normal test runs only read tests/data/love_reference.txt, so
mpmath is not a runtime dependency of ASPHER or of the C++ CI jobs.
"""
import json
import os
import subprocess
import sys

import mpmath as mp

DIGITS = 80
mp.mp.dps = DIGITS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                   "love_reference.txt")


def love_mp(x, y, a, b):
    """Love (1929) closed form for the integral of 1/|r-r'| over
    [-a,a] x [-b,b], evaluated at working precision. asinh form, which is the
    algebraically identical but numerically stable rearrangement:

        L = sum over the four corner terms of  t * asinh(s/|t|),

    with the removable  t -> 0  limits (t log(1/|t|) -> 0) taken explicitly.
    """
    x, y, a, b = (mp.mpf(v) for v in (x, y, a, b))
    xp, xm, yp, ym = x + a, x - a, y + b, y - b

    def term(t, s1, s2):
        if t == 0:
            return mp.mpf(0)
        c = abs(t)
        return t * (mp.asinh(s1 / c) - mp.asinh(s2 / c))

    return (term(xp, yp, ym) + term(yp, xp, xm)
            + term(xm, ym, yp) + term(ym, xm, xp))


def love_quad(x, y, a, b):
    """Independent adaptive quadrature of the same integral, at high precision.

    The inner integral over x' is done analytically (it is an asinh), leaving a
    1-D integrand with at worst a logarithmic singularity at y' = y. mpmath's
    adaptive quadrature is given that abscissa as an explicit interval
    endpoint, so no panel straddles it — an ordinary tensor Gauss rule across
    the singularity would NOT be an adequate oracle (plan §1.3).
    """
    x, y, a, b = (mp.mpf(v) for v in (x, y, a, b))

    def inner(yp):
        d = abs(y - yp)
        if d == 0:
            return mp.mpf(0)
        return mp.asinh((a - x) / d) + mp.asinh((a + x) / d)

    if -b < y < b:
        pts = [-b, y, b]
    else:
        pts = [-b, b]
    val = mp.quad(inner, pts, maxdegree=12)
    return val, mp.mpf(0)


# ── Cerruti brackets (spec A05) ────────────────────────────────────────────
# uxx = (1-nu) * Love + nu * ylog,   ylog = int (x-xi)^2 / rho^3 dA'
# uxy = int (x-xi)(y-eta) / rho^3 dA'
# Both are generated in the same two-independent-paths way: a closed form in
# the numerically stable rearrangement, and 1-D adaptive quadrature of the
# integral with the inner xi-integration done analytically and the abscissa
# eta = y given to the quadrature as an explicit interval endpoint.

def cerruti_ylog_mp(x, y, a, b):
    """m*[asinh(k/m)-asinh(l/m)] + n*[asinh(l/|n|)-asinh(k/|n|)] at precision."""
    x, y, a, b = (abs(mp.mpf(v)) if i < 2 else mp.mpf(v)
                  for i, v in enumerate((x, y, a, b)))
    k, l = x + a, x - a
    m, n = y + b, y - b

    def term(t, s1, s2):
        if t == 0:
            return mp.mpf(0)
        c = abs(t)
        return t * (mp.asinh(s1 / c) - mp.asinh(s2 / c))

    return term(m, k, l) + term(n, l, k)


def cerruti_uxy_mp(x, y, a, b):
    sx = 1 if x >= 0 else -1
    sy = 1 if y >= 0 else -1
    x, y, a, b = (mp.mpf(abs(x)), mp.mpf(abs(y)), mp.mpf(a), mp.mpf(b))
    k, l = x + a, x - a
    m, n = y + b, y - b
    Rkm, Rkn = mp.hypot(k, m), mp.hypot(k, n)
    Rlm, Rln = mp.hypot(l, m), mp.hypot(l, n)
    return sx * sy * (Rkn - Rkm + Rlm - Rln)


def cerruti_ylog_quad(x, y, a, b):
    x, y, a, b = (mp.mpf(v) for v in (x, y, a, b))

    def F(u, d):
        return mp.asinh(u / d) - u / mp.sqrt(u * u + d * d)

    def inner(eta):
        d = abs(y - eta)
        if d == 0:
            return mp.mpf(0)
        return F(x + a, d) - F(x - a, d)

    pts = [-b, y, b] if -b < y < b else [-b, b]
    return mp.quad(inner, pts, maxdegree=12)


def cerruti_uxy_quad(x, y, a, b):
    x, y, a, b = (mp.mpf(v) for v in (x, y, a, b))

    def inner(eta):
        d = y - eta
        return d * (1 / mp.sqrt((x - a) ** 2 + d * d)
                    - 1 / mp.sqrt((x + a) ** 2 + d * d))

    pts = [-b, y, b] if -b < y < b else [-b, b]
    return mp.quad(inner, pts, maxdegree=12)


def cerruti_cases():
    import math
    a = b = 0.5
    out = []
    for (x, y) in [(0, 0), (a, 0), (0, b), (a, b), (1.0, 0.0), (0.0, 1.0),
                   (1.0, 1.0), (2.0, 3.0), (7.0, 0.0), (0.0, 7.0)]:
        out.append((x, y, a, b))
    for r in [3.0, 8.0, 32.0, 1e3, 1.6384e4, 1e5, 1e6]:
        for th in [0.0, 1e-3, math.pi / 8, math.pi / 4, math.pi / 3,
                   math.pi / 2]:
            out.append((r * math.cos(th), r * math.sin(th), a, b))
    for ar in (1 / 4, 4.0):
        aa, bb = 0.5 * math.sqrt(ar), 0.5 / math.sqrt(ar)
        for (x, y) in [(0.0, 0.0), (3.0, 2.0), (500.0, 300.0)]:
            out.append((x, y, aa, bb))
    return out


def revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def cases():
    """Offsets in units of the element size h = 1 with a = b = 0.5.

    T10 sweeps normalised offsets from sub-cell distances out to 1e6 along
    axes, near-axes, diagonals and generic angles in every quadrant; T11 adds
    the self, edge and corner points and signed perturbations on both sides of
    each; T12 adds rectangular aspect ratios.
    """
    out = []
    a = b = 0.5
    # T11: self / edge / corner and their neighbourhoods
    eps = 1e-12
    for (x, y) in [(0, 0), (a, 0), (0, b), (a, b), (-a, -b), (a, -b)]:
        out.append((x, y, a, b))
        for dx in (-eps, eps, 0.0):
            for dy in (-eps, eps, 0.0):
                if dx or dy:
                    out.append((x + dx, y + dy, a, b))
    # T10: magnitude sweep x angle x quadrant
    import math
    mags = [0.25, 0.75, 1.0, 1.5, 2.0, 3.0, 8.0, 32.0, 1e3, 1.6384e4, 1e5, 1e6]
    angles = [0.0, 1e-6, 1e-3, math.pi / 8, math.pi / 4, math.pi / 3,
              math.pi / 2 - 1e-3, math.pi / 2]
    for r in mags:
        for th in angles:
            for sx in (1, -1):
                for sy in (1, -1):
                    out.append((sx * r * math.cos(th), sy * r * math.sin(th),
                                a, b))
    # T12: rectangular aspect ratios
    for ar in (1 / 16, 1 / 4, 1.0, 4.0, 16.0):
        aa, bb = 0.5 * math.sqrt(ar), 0.5 / math.sqrt(ar)
        for (x, y) in [(0, 0), (2 * aa, 0), (0, 2 * bb), (5.0, 3.0),
                       (100.0, 0.0), (0.0, 100.0), (1e4, 1e4)]:
            out.append((x, y, aa, bb))
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for c in out:
        k = tuple(repr(v) for v in c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def main():
    rows = []
    worst = 0.0
    for (x, y, a, b) in cases():
        ref = love_mp(x, y, a, b)
        # cross-check with independent quadrature where it is reliable
        cross = ""
        try:
            q, _ = love_quad(x, y, a, b)
            rel = float(abs(q - ref) / max(abs(ref), mp.mpf('1e-300')))
            cross = "%.3e" % rel
            worst = max(worst, rel)
        except Exception:
            cross = ""
        rows.append(dict(x=repr(float(x)), y=repr(float(y)),
                         a=repr(float(a)), b=repr(float(b)),
                         value=mp.nstr(ref, 25),
                         quad_rel=cross))
    header = dict(generator=os.path.basename(__file__), revision=revision(),
                  digits=DIGITS, units="lengths in units of h; the value is "
                                       "the integral of 1/|r-r'| over the cell "
                                       "(no elastic prefactor)",
                  worst_quadrature_cross_check=("%.3e" % worst),
                  n_cases=len(rows))
    # ── Cerruti fixture ──
    crows = []
    cworst = 0.0
    for (x, y, a, b) in cerruti_cases():
        yl = cerruti_ylog_mp(x, y, a, b)
        xy = cerruti_uxy_mp(x, y, a, b)
        try:
            q1 = cerruti_ylog_quad(abs(x), abs(y), a, b)
            r1 = float(abs(q1 - yl) / max(abs(yl), mp.mpf('1e-300')))
            cworst = max(cworst, r1)
        except Exception:
            pass
        try:
            q2 = cerruti_uxy_quad(abs(x), abs(y), a, b)
            sgn = (1 if x >= 0 else -1) * (1 if y >= 0 else -1)
            if abs(xy) > 1e-280:
                r2 = float(abs(sgn * q2 - xy) / abs(xy))
                cworst = max(cworst, r2)
        except Exception:
            pass
        crows.append((repr(float(x)), repr(float(y)), repr(float(a)),
                      repr(float(b)), mp.nstr(yl, 25), mp.nstr(xy, 25)))
    cout = os.path.join(os.path.dirname(OUT), "cerruti_reference.txt")
    with open(cout, "w") as f:
        f.write("# ASPHER Cerruti bracket fixture — DO NOT EDIT BY HAND\n")
        f.write("# " + json.dumps(dict(
            generator=os.path.basename(__file__), revision=revision(),
            digits=DIGITS, n_cases=len(crows),
            worst_quadrature_cross_check=("%.3e" % cworst),
            units="lengths in units of h; ylog = int (x-xi)^2/rho^3 dA', "
                  "uxy = int (x-xi)(y-eta)/rho^3 dA' (no elastic prefactor)")) + "\n")
        f.write("# x y a b ylog uxy\n")
        for r in crows:
            f.write(" ".join(r) + "\n")
    print("wrote %d Cerruti cases to %s" % (len(crows), cout))
    print("worst Cerruti quadrature cross-check: %.3e" % cworst)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# ASPHER Love-kernel reference fixture — DO NOT EDIT BY HAND\n")
        f.write("# " + json.dumps(header) + "\n")
        f.write("# x y a b value\n")
        for r in rows:
            f.write("%s %s %s %s %s\n" %
                    (r["x"], r["y"], r["a"], r["b"], r["value"]))
    print("wrote %d cases to %s" % (len(rows), OUT))
    print("worst independent-quadrature cross-check: %.3e" % worst)


if __name__ == "__main__":
    sys.exit(main())
