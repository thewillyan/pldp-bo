from __future__ import annotations

import pytest

from src.config.loader import ExperimentConfig
from src.config.locked import (
    LOCKED_CONSTANTS,
    LockedConfigError,
    assert_locked_config,
    collect_violations,
    config_version,
)


def _apply(_cfg: ExperimentConfig, *pairs: tuple[object, str, object]) -> None:
    """Apply attribute writes for multi-setattr contract-violation cases."""
    for obj, name, value in pairs:
        setattr(obj, name, value)


def _locked_config() -> ExperimentConfig:
    """A config compliant with every §2 locked constant and the method contract."""
    cfg = ExperimentConfig()
    cfg.method = "pldpbo_nun"
    cfg.data.num_clients = 100
    cfg.data.batch_size = 64
    cfg.data.val_split = 0.1
    cfg.federated.num_rounds = 200
    cfg.federated.fraction_fit = 0.1
    cfg.federated.min_fit_clients = 10
    cfg.federated.local_epochs = 5
    cfg.federated.server_learning_rate = 0.01
    cfg.federated.proximal_mu = 0.0
    cfg.optimizer.name = "sgd"
    cfg.optimizer.momentum = 0.9
    cfg.optimizer.weight_decay = 0.0
    cfg.optimizer.gradient_clip_norm = 0.0
    cfg.privacy.enabled = True
    cfg.privacy.mechanism = "gaussian"
    cfg.privacy.clipping_mode = "per_example"
    cfg.privacy.update_clip_norm = 1.0
    cfg.privacy.accountant_mode = "rdp_native"
    cfg.privacy.rdp_alpha = 10.0
    cfg.privacy.total_budget = 10.0
    cfg.privacy.enforce_budget = True
    cfg.personalization.enabled = False
    cfg.bo.enabled = True
    cfg.bo.bounds_strategy = "global"
    cfg.bo.rdp_min = 0.01
    cfg.bo.rdp_max = 2.0
    cfg.bo.acquisition_penalty = 0.1
    cfg.bo.grid_points = 50
    cfg.bo.gp_kernel = "matern52"
    cfg.bo.min_warmup = 10
    return cfg


def test_compliant_config_passes() -> None:
    cfg = _locked_config()
    assert collect_violations(cfg) == []
    assert_locked_config(cfg)  # must not raise


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("T", lambda c: setattr(c.federated, "num_rounds", 199)),
        ("K", lambda c: setattr(c.data, "num_clients", 99)),
        ("rho", lambda c: setattr(c.federated, "fraction_fit", 0.2)),
        ("min_fit_clients", lambda c: setattr(c.federated, "min_fit_clients", 5)),
        ("E", lambda c: setattr(c.federated, "local_epochs", 4)),
        ("B", lambda c: setattr(c.data, "batch_size", 32)),
        ("eta_server", lambda c: setattr(c.federated, "server_learning_rate", 0.5)),
        ("local_opt", lambda c: setattr(c.optimizer, "name", "adam")),
        ("momentum", lambda c: setattr(c.optimizer, "momentum", 0.0)),
        ("clip_norm", lambda c: setattr(c.privacy, "update_clip_norm", 5.0)),
        ("alpha0", lambda c: setattr(c.privacy, "rdp_alpha", 20.0)),
        ("accountant_mode", lambda c: setattr(c.privacy, "accountant_mode", "epsilon")),
        ("clipping_mode", lambda c: setattr(c.privacy, "clipping_mode", "per_update")),
        ("mechanism", lambda c: setattr(c.privacy, "mechanism", "laplace")),
        ("B_RDP", lambda c: setattr(c.privacy, "total_budget", 120.0)),
        ("personalization", lambda c: setattr(c.personalization, "enabled", True)),
        ("R_min", lambda c: setattr(c.bo, "rdp_min", 0.1)),
        ("R_max", lambda c: setattr(c.bo, "rdp_max", 10.0)),
        ("bounds_strategy", lambda c: setattr(c.bo, "bounds_strategy", "from_rdp")),
        ("lambda_aq", lambda c: setattr(c.bo, "acquisition_penalty", 0.3)),
        ("G", lambda c: setattr(c.bo, "grid_points", 100)),
        ("kernel", lambda c: setattr(c.bo, "gp_kernel", "rbf")),
        ("min_warmup", lambda c: setattr(c.bo, "min_warmup", 5)),
        ("validation_frac", lambda c: setattr(c.data, "val_split", 0.2)),
        ("enforce_budget", lambda c: setattr(c.privacy, "enforce_budget", False)),
        ("weight_decay", lambda c: setattr(c.optimizer, "weight_decay", 0.001)),
        ("gradient_clip_norm", lambda c: setattr(c.optimizer, "gradient_clip_norm", 1.0)),
    ],
)
def test_constant_violations_raise(label: str, mutate: object) -> None:
    cfg = _locked_config()
    mutate(cfg)  # type: ignore[operator]
    violations = collect_violations(cfg)
    assert any(label in v for v in violations)
    with pytest.raises(LockedConfigError, match=label):
        assert_locked_config(cfg)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("method", lambda c: setattr(c, "method", "bogus")),
        ("method", lambda c: setattr(c, "method", "")),
        # nonprivate contract
        ("method", lambda c: setattr(c, "method", "nonprivate")),
        ("method", lambda c: _apply(c, (c, "method", "nonprivate"), (c.privacy, "enabled", True))),
        ("method", lambda c: _apply(c, (c, "method", "nonprivate"), (c.bo, "enabled", True))),
        (
            "method",
            lambda c: _apply(
                c,
                (c, "method", "nonprivate"),
                (c.federated, "aggregation", "attenuation"),
            ),
        ),
        # fixed baselines
        (
            "method",
            lambda c: _apply(
                c,
                (c, "method", "dpfedavg_fixed"),
                (c.privacy, "enabled", False),
                (c.bo, "enabled", False),
            ),
        ),
        ("method", lambda c: _apply(c, (c, "method", "dpfedavg_fixed"), (c.bo, "enabled", True))),
        (
            "fixed_rdp_target",
            lambda c: _apply(
                c,
                (c, "method", "dpfedavg_fixed"),
                (c.bo, "enabled", False),
                (c.privacy, "fixed_rdp_target", 1.0),
            ),
        ),
        # BO variants
        ("method", lambda c: _apply(c, (c, "method", "pldpbo_snr"), (c.bo, "enabled", False))),
        (
            "method",
            lambda c: _apply(c, (c, "method", "pldpbo_utility"), (c.privacy, "enabled", False)),
        ),
        (
            "method",
            lambda c: _apply(
                c,
                (c, "method", "pldpbo_agreement"),
                (c.federated, "aggregation", "plain"),
            ),
        ),
        # fedprox mu
        (
            "mu_fedprox",
            lambda c: _apply(
                c,
                (c, "method", "fedprox_fixed"),
                (c.bo, "enabled", False),
                (c.federated, "proximal_mu", 0.0),
            ),
        ),
        ("mu_fedprox", lambda c: setattr(c.federated, "proximal_mu", 0.01)),
        # aggregation value
        ("aggregation", lambda c: setattr(c.federated, "aggregation", "bogus")),
    ],
)
def test_method_contract_violations_raise(label: str, mutate: object) -> None:
    cfg = _locked_config()
    mutate(cfg)  # type: ignore[operator]
    violations = collect_violations(cfg)
    assert any(label in v for v in violations)
    with pytest.raises(LockedConfigError, match=label):
        assert_locked_config(cfg)


def test_all_violations_reported_at_once() -> None:
    cfg = _locked_config()
    cfg.federated.num_rounds = 10
    cfg.data.num_clients = 8
    cfg.bo.grid_points = 10
    violations = collect_violations(cfg)
    assert len(violations) == 3
    with pytest.raises(LockedConfigError) as excinfo:
        assert_locked_config(cfg)
    assert "T" in str(excinfo.value)
    assert "K" in str(excinfo.value)
    assert "G" in str(excinfo.value)


def test_assertion_disabled_bypasses() -> None:
    cfg = _locked_config()
    cfg.assert_locked_config = False
    cfg.federated.num_rounds = 10
    assert_locked_config(cfg)  # must not raise
    assert collect_violations(cfg)  # violations still detectable


def test_config_version_stable() -> None:
    assert config_version() == config_version()
    assert len(config_version()) == 64  # sha256 hex
    assert all(c in "0123456789abcdef" for c in config_version())


def test_config_version_changes_with_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    before = config_version()
    monkeypatch.setitem(LOCKED_CONSTANTS, "T", 201)
    assert config_version() != before


def test_every_method_name_is_valid() -> None:
    for name in (
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
    ):
        cfg = _locked_config()
        cfg.method = name
        if name == "nonprivate":
            cfg.privacy.enabled = False
            cfg.bo.enabled = False
            cfg.federated.aggregation = "plain"
        elif name in ("dpfedavg_fixed", "fedprox_fixed"):
            cfg.bo.enabled = False
            if name == "fedprox_fixed":
                cfg.federated.proximal_mu = 0.01
        assert collect_violations(cfg) == []
