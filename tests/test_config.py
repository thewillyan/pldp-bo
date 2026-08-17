from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from src.config.loader import ExperimentConfig, load_config


def test_load_default_config() -> None:
    config = load_config("config/default.yaml")
    assert config.data.name == "cifar10"
    assert config.federated.num_rounds == 50
    assert config.privacy.enabled is False


def test_load_experiment_config() -> None:
    config = load_config("config/dp_example.yaml")
    assert config.privacy.enabled is True


def test_config_override() -> None:
    config = load_config("config/default.yaml", overrides={"data.num_clients": 5})
    assert config.data.num_clients == 5


def test_config_from_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    config = load_config(str(config_path))
    assert isinstance(config, ExperimentConfig)


def test_config_from_custom_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "data": {"name": "mnist", "num_clients": 3},
                "federated": {"num_rounds": 5},
            }
        )
    )
    config = load_config(str(config_path))
    assert config.data.name == "mnist"
    assert config.federated.num_rounds == 5
    assert config.model.name == "cnn"  # default preserved


def test_load_personalization_config() -> None:
    config = load_config("config/personalized_custom.yaml")
    assert config.personalization.enabled is True
    assert config.personalization.strategy == "custom"
    assert cast(dict[int, float], config.personalization.client_epsilon_map)[0] == 1.0
    assert config.personalization.track_cumulative is True


def test_personalization_defaults_when_absent() -> None:
    config = load_config("config/default.yaml")
    assert config.personalization.enabled is False
    assert config.personalization.strategy == "uniform"
    assert config.personalization.client_epsilon_map == {}


def test_personalization_config_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={"personalization.enabled": True, "personalization.strategy": "heterogeneity"},
    )
    assert config.personalization.enabled is True
    assert config.personalization.strategy == "heterogeneity"


def test_bo_config_defaults() -> None:
    config = load_config("config/default.yaml")
    assert config.bo.enabled is False
    assert config.bo.min_warmup == 3
    assert config.bo.epsilon_budget == 10.0
    assert config.bo.optimization_metric == "nun"


def test_bo_config_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={
            "bo.enabled": True,
            "bo.min_warmup": 10,
            "bo.epsilon_budget": 5.0,
            "bo.optimization_metric": "utility",
        },
    )
    assert config.bo.enabled is True
    assert config.bo.min_warmup == 10
    assert config.bo.epsilon_budget == 5.0
    assert config.bo.optimization_metric == "utility"


def test_clipping_mode_default() -> None:
    config = ExperimentConfig()
    assert config.privacy.clipping_mode == "per_update"


def test_clipping_mode_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump({"privacy": {"enabled": True, "clipping_mode": "per_example"}}))
    config = load_config(str(cfg_file))
    assert config.privacy.clipping_mode == "per_example"


def test_clipping_mode_from_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={"privacy.clipping_mode": "per_example"},
    )
    assert config.privacy.clipping_mode == "per_example"


def test_locked_layer_fields_defaults() -> None:
    config = ExperimentConfig()
    assert config.method == ""
    assert config.assert_locked_config is True
    assert config.federated.aggregation == "attenuation"
    assert config.privacy.enforce_budget is True
    assert config.privacy.fixed_rdp_target == 0.5


def test_locked_layer_fields_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "matrix.yaml"
    cfg_file.write_text(
        yaml.dump(
            {
                "method": "pldpbo_nun",
                "assert_locked_config": True,
                "federated": {"aggregation": "attenuation"},
                "privacy": {"enforce_budget": True, "fixed_rdp_target": 0.5},
            }
        )
    )
    config = load_config(str(cfg_file))
    assert config.method == "pldpbo_nun"
    assert config.assert_locked_config is True
    assert config.federated.aggregation == "attenuation"
    assert config.privacy.enforce_budget is True
    assert config.privacy.fixed_rdp_target == 0.5


def test_locked_layer_fields_from_override() -> None:
    config = load_config(
        "config/default.yaml",
        overrides={
            "method": "nonprivate",
            "assert_locked_config": "false",
            "federated.aggregation": "plain",
            "privacy.enforce_budget": "false",
            "privacy.fixed_rdp_target": "1.0",
        },
    )
    assert config.method == "nonprivate"
    assert config.assert_locked_config is False
    assert config.federated.aggregation == "plain"
    assert config.privacy.enforce_budget is False
    assert config.privacy.fixed_rdp_target == 1.0


class TestSmokeConfigs:
    """IMPL-14: smoke configs load and satisfy the method contract."""

    @pytest.mark.parametrize(
        "name",
        [
            "nonprivate",
            "dpfedavg_fixed",
            "fedprox_fixed",
            "pldpbo_nun",
        ],
    )
    def test_smoke_cell_loads_and_contract_clean(self, name: str) -> None:
        from src.config.locked import collect_violations

        cfg = load_config(f"config/smoke/{name}.yaml")
        violations = collect_violations(cfg)
        # The fixed cells deliberately retarget R to ≈ B_RDP/T so the tiny
        # smoke horizon fully spends the budget without refusals (§9.3 math);
        # pldpbo_nun raises T to 25 so its BO rounds spend the post-warm-up
        # 8.6005 RDP (IMPL-14 Task 6 contingencies).
        assert not any(
            v.startswith("method") and "fixed_rdp_target" not in v for v in violations
        ), violations
        assert cfg.federated.num_rounds == (25 if name == "pldpbo_nun" else 20)
        assert cfg.data.num_clients == 4

    def test_femnist_loader_config_loads(self) -> None:
        cfg = load_config("config/smoke/femnist_loader.yaml")
        assert cfg.data.name == "femnist"
