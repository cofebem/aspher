#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace hmc {

// Friction threshold models (spec §7): s_i = τ_c(p_i, |v_i|, T_i), the
// per-point bound |q_i| <= s_i of the tangential QP. Contract: s_i = 0
// wherever p_i <= 0 (no traction without contact) and s_i >= 0. The solver
// strategy is derivative-free — models only ever evaluate the threshold.
class FrictionModel {
public:
    virtual ~FrictionModel() = default;
    virtual void threshold(const Eigen::VectorXd& p, const Eigen::VectorXd& v,
                           const Eigen::VectorXd& T,
                           Eigen::VectorXd& s) const = 0;
    virtual bool velocity_dependent() const { return false; }
};

// Tresca: constant threshold τ_c inside the contact, 0 outside.
class TrescaModel final : public FrictionModel {
public:
    explicit TrescaModel(double tau_c) : tau_c_(tau_c) {}
    void threshold(const Eigen::VectorXd& p, const Eigen::VectorXd&,
                   const Eigen::VectorXd&, Eigen::VectorXd& s) const override {
        s.resizeLike(p);
        for (Eigen::Index i = 0; i < p.size(); ++i) {
            s(i) = (p(i) > 0.0) ? tau_c_ : 0.0;
        }
    }

private:
    double tau_c_;
};

// Coulomb: s = μ p (p clamped at 0).
class CoulombModel final : public FrictionModel {
public:
    explicit CoulombModel(double mu) : mu_(mu) {}
    void threshold(const Eigen::VectorXd& p, const Eigen::VectorXd&,
                   const Eigen::VectorXd&, Eigen::VectorXd& s) const override {
        s = mu_ * p.array().max(0.0);
    }

private:
    double mu_;
};

// Generic user law (M7 exposes this to Python): the callback fills s from
// (p, v, T); the wrapper then SANITIZES the output — clamped to >= 0 and
// zeroed wherever p <= 0 — so no user law can hand the solver an
// inadmissible threshold field.
class CallbackModel final : public FrictionModel {
public:
    using Fn = std::function<void(const Eigen::VectorXd& p,
                                  const Eigen::VectorXd& v,
                                  const Eigen::VectorXd& T,
                                  Eigen::VectorXd& s)>;
    CallbackModel(Fn fn, bool velocity_dependent)
        : fn_(std::move(fn)), vdep_(velocity_dependent) {}
    void threshold(const Eigen::VectorXd& p, const Eigen::VectorXd& v,
                   const Eigen::VectorXd& T,
                   Eigen::VectorXd& s) const override {
        fn_(p, v, T, s);
        // Validate size
        if (s.size() != p.size())
            throw std::invalid_argument(
                "CallbackModel: callback returned wrong-size threshold");
        // Sanitize: clamp to >= 0 and zero where p <= 0
        // (handle NaN: std::max(NaN, 0) returns NaN, so check isfinite)
        for (Eigen::Index i = 0; i < p.size(); ++i) {
            s(i) = (p(i) > 0.0 && std::isfinite(s(i)) && s(i) > 0.0) ? s(i)
                                                                      : 0.0;
        }
    }
    bool velocity_dependent() const override { return vdep_; }

private:
    Fn fn_;
    bool vdep_;
};

} // namespace hmc
