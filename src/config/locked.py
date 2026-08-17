from __future__ import annotations

import hashlib
import math

from src.config.loader import ExperimentConfig

# Methods defined by the experimental matrix (EXPERIMENTS-TODO.md §3).
METHOD_NAMES = frozenset(
    {
        "nonprivate",
        "dpfedavg_fixed",
        "fedprox_fixed",
        "pldpbo_nun",
        "pldpbo_utility",
        "pldpbo_retention",
        "pldpbo_efficiency",
        "pldpbo_perremaining",
        "pldpbo_snr",
        "pldpbo_agreement",
    }
)

FIXED_METHODS = frozenset({"dpfedavg_fixed", "fedprox_fixed"})
BO_METHODS = frozenset(
    {
        "pldpbo_nun",
        "pldpbo_utility",
        "pldpbo_retention",
        "pldpbo_efficiency",
        "pldpbo_perremaining",
        "pldpbo_snr",
        "pldpbo_agreement",
    }
)

# Locked §2 constants, normalized as {spec label: expected value}. This table
# is the single source of truth for the startup assertion and for
# `config_version` (the tag the aggregation script uses to gate reruns).
LOCKED_CONSTANTS: dict[str, object] = {
    "T": 200,  # communication rounds
    "K": 100,  # clients
    "rho": 0.1,  # participation fraction
    "min_fit_clients": 10,  # derived: rho * K ≈ clients/round
    "E": 5,  # local epochs
    "B": 64,  # local batch size
    "eta_server": 0.01,  # server learning rate
    "momentum": 0.9,  # local SGD momentum
    "clip_norm": 1.0,  # per-example clipping norm C
    "alpha0": 10.0,  # fixed RDP order
    "B_RDP": 10.0,  # per-client RDP budget, flat
    "fixed_rdp_target": 0.5,  # B_RDP / (rho * T) for the fixed baselines
    "R_min": 0.01,  # RDP search interval lower bound
    "R_max": 2.0,  # RDP search interval upper bound
    "lambda_aq": 0.1,  # acquisition penalty
    "G": 50,  # BO grid points
    "min_warmup": 10,  # warm-up participations (log-spaced grid, §9.3)
    "validation_frac": 0.1,  # per-client validation hold-out
}

_REL_TOL = 1e-9


class LockedConfigError(RuntimeError):
    """Raised when a run config deviates from the §2 locked constants."""


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=_REL_TOL)


def collect_violations(cfg: ExperimentConfig) -> list[str]:
    """Return every §2 / method-contract violation in *cfg* (empty if compliant)."""
    violations: list[str] = []

    def expect(label: str, actual: object, expected: object) -> None:
        if isinstance(actual, bool) or isinstance(expected, bool):
            ok = actual is expected
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            ok = _close(float(actual), float(expected))
        else:
            ok = actual == expected
        if not ok:
            violations.append(f"{label}: expected {expected!r}, got {actual!r}")

    data, fed, opt, priv, bo = cfg.data, cfg.federated, cfg.optimizer, cfg.privacy, cfg.bo

    # --- §2 locked constants -------------------------------------------------
    expect("T", fed.num_rounds, 200)
    expect("K", data.num_clients, 100)
    expect("rho", fed.fraction_fit, 0.1)
    expect("min_fit_clients", fed.min_fit_clients, 10)
    expect("E", fed.local_epochs, 5)
    expect("B", data.batch_size, 64)
    expect("eta_server", fed.server_learning_rate, 0.01)
    expect("local_opt", opt.name, "sgd")
    expect("momentum", opt.momentum, 0.9)
    expect("clip_norm", priv.update_clip_norm, 1.0)
    expect("alpha0", priv.rdp_alpha, 10.0)
    expect("accountant_mode", priv.accountant_mode, "rdp_native")
    expect("clipping_mode", priv.clipping_mode, "per_example")
    expect("mechanism", priv.mechanism, "gaussian")
    expect("B_RDP", priv.total_budget, 10.0)
    expect("personalization", cfg.personalization.enabled, False)
    expect("R_min", bo.rdp_min, 0.01)
    expect("R_max", bo.rdp_max, 2.0)
    expect("bounds_strategy", bo.bounds_strategy, "global")
    expect("lambda_aq", bo.acquisition_penalty, 0.1)
    expect("G", bo.grid_points, 50)
    expect("kernel", bo.gp_kernel, "matern52")
    expect("min_warmup", bo.min_warmup, 10)
    expect("validation_frac", data.val_split, 0.1)
    expect("enforce_budget", priv.enforce_budget, True)
    expect("weight_decay", opt.weight_decay, 0.0)
    expect("gradient_clip_norm", opt.gradient_clip_norm, 0.0)

    # --- method contract -----------------------------------------------------
    method = cfg.method
    if method not in METHOD_NAMES:
        violations.append(f"method: expected one of {sorted(METHOD_NAMES)}, got {method!r}")

    if fed.aggregation not in ("attenuation", "plain"):
        violations.append(
            f"aggregation: expected 'attenuation' or 'plain', got {fed.aggregation!r}"
        )

    if method == "nonprivate":
        if priv.enabled:
            violations.append("method 'nonprivate': privacy.enabled must be False")
        if bo.enabled:
            violations.append("method 'nonprivate': bo.enabled must be False")
        if fed.aggregation != "plain":
            violations.append("method 'nonprivate': aggregation must be 'plain'")
    elif method in FIXED_METHODS:
        if not priv.enabled:
            violations.append(f"method '{method}': privacy.enabled must be True")
        if bo.enabled:
            violations.append(f"method '{method}': bo.enabled must be False")
        if fed.aggregation != "attenuation":
            violations.append(f"method '{method}': aggregation must be 'attenuation'")
        expect(f"method '{method}' fixed_rdp_target", priv.fixed_rdp_target, 0.5)
    elif method in BO_METHODS:
        if not priv.enabled:
            violations.append(f"method '{method}': privacy.enabled must be True")
        if not bo.enabled:
            violations.append(f"method '{method}': bo.enabled must be True")
        if fed.aggregation != "attenuation":
            violations.append(f"method '{method}': aggregation must be 'attenuation'")

    if method == "fedprox_fixed":
        expect("mu_fedprox", fed.proximal_mu, 0.01)
    elif method in METHOD_NAMES:
        expect("mu_fedprox", fed.proximal_mu, 0.0)

    return violations


def assert_locked_config(cfg: ExperimentConfig) -> None:
    """Fail fast when *cfg* deviates from §2; no-op if cfg.assert_locked_config is False.

    Must be called right after `load_config` in the server and client entry
    points so that non-spec runs die before any training/evaluation happens.
    """
    if not cfg.assert_locked_config:
        return
    violations = collect_violations(cfg)
    if violations:
        raise LockedConfigError(
            "Locked-config violation(s):\n  - " + "\n  - ".join(violations),
        )


def config_version() -> str:
    """Stable sha256 over the §2 locked constants (EXPERIMENTS-TODO.md §4.1 tag)."""
    digest = hashlib.sha256()
    for name in sorted(LOCKED_CONSTANTS):
        digest.update(f"{name}={LOCKED_CONSTANTS[name]!r}\n".encode())
    return digest.hexdigest()
