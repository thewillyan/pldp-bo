from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common import Message, Metadata
from flwr.serverapp.grid.grid import Grid
from flwr.serverapp.strategy import FedAvg, FedProx

from src.tracking.tracker import ExperimentTracker

_EPS = 1e-12
_MIN_VALUES_FOR_STATS = 3


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _filter_valid_replies(replies: Iterable[Message]) -> list[Message]:
    """Filter replies to only those with valid training results.

    Replaces Flower's private ``FedAvg._check_and_log_replies`` to avoid
    depending on an unstable internal API.
    """
    valid: list[Message] = []
    for reply in replies:
        metrics_rec = reply.content.metric_records.get("metrics")
        if metrics_rec is None:
            continue
        num_examples = metrics_rec.get("num-examples", 0)
        if num_examples is None or num_examples <= 0:
            continue
        valid.append(reply)
    return valid


def _add_budgets_to_messages(
    messages: Iterable[Message],
    budgets: dict[int, float] | None,
    configrecord_key: str,
    node_to_partition: dict[int, int] | None = None,
    remaining_rdp_by_client: dict[int, float] | None = None,
) -> Iterable[Message]:
    if budgets is None:
        yield from messages
        return
    for msg in messages:
        dst = msg.metadata.dst_node_id
        partition_id = (node_to_partition or {}).get(dst, dst)
        budget = budgets.get(partition_id)
        if budget is not None:
            content = RecordDict()
            for key, rec in msg.content.config_records.items():
                if key == configrecord_key:
                    extra: dict[str, float] = {"per_client_budget": budget}
                    remaining = (remaining_rdp_by_client or {}).get(partition_id)
                    if remaining is not None:
                        extra["remaining_rdp"] = remaining
                    content[key] = ConfigRecord(
                        {
                            **rec,
                            **extra,
                        }
                    )
                else:
                    content[key] = rec
            for key, rec in msg.content.array_records.items():
                content[key] = rec
            for key, rec in msg.content.metric_records.items():
                content[key] = rec

            orig = msg.metadata
            new_metadata = Metadata(
                run_id=orig.run_id,
                message_id=orig.message_id,
                src_node_id=orig.src_node_id,
                dst_node_id=dst,
                reply_to_message_id=orig.reply_to_message_id,
                group_id=orig.group_id,
                created_at=orig.created_at,
                ttl=orig.ttl,
                message_type=orig.message_type,
                src_task_id=orig.src_task_id,
                dst_task_id=orig.dst_task_id,
            )
            yield Message(content=content, metadata=new_metadata)
        else:
            yield msg


class MetricLoggingMixin(FedAvg):
    _tracker: ExperimentTracker | None
    _per_client_budgets: dict[int, float] | None
    _node_to_partition: dict[int, int]
    _client_cum_rdp: dict[int, float]
    _client_cum_eps: dict[int, float]

    _PARTICIPATION_KEYS = (
        "r_t_candidate",
        "r_t_final",
        "cum_rdp",
        "remaining_rdp",
        "phase",
        "observed_m",
        "acct_cost",
    )
    _COMPONENT_KEYS = (
        ("L_clean", "utility_loss_clean"),
        ("L_noisy", "utility_loss"),
        ("update_norm_noisy", "update_norm"),
        ("update_norm_clean", "update_norm_clean"),
        ("sigma", "sigma"),
        ("agreement", "logit_disagreement"),
    )
    _PHASE_DECODE = {0.0: "warmup", 1.0: "bo", 2.0: "exhausted"}

    def _init_spec_state(self) -> None:
        self._client_state: dict[int, dict[str, Any]] = {}
        self._client_dropout_round: dict[int, int] = {}
        self._bo_time_total = 0.0
        self._acct_time_total = 0.0
        self._bytes_sent_round = 0
        self._remaining_rdp_sent: dict[int, float] = {}

    def _array_bytes(self, record: ArrayRecord | None) -> int:
        if record is None:
            return 0
        return sum(int(v.numpy().nbytes) for v in record.values())

    def _append_client_participation(
        self,
        cid: int,
        m: MetricRecord,
        server_round: int,
    ) -> None:
        phase_raw = m.get("phase")
        phase_float = _as_float(phase_raw)
        if phase_float is None:
            return
        phase = self._PHASE_DECODE.get(phase_float, "bo")
        state: dict[str, Any] = self._client_state.setdefault(
            cid,
            {key: [] for key in (*self._PARTICIPATION_KEYS, *(k for k, _ in self._COMPONENT_KEYS))}
            | {"warmup_rounds": [], "enforcement_count": 0},
        )
        state["r_t_candidate"].append(_as_float(m.get("r_t_candidate")))
        final = _as_float(m.get("r_t_final"))
        state["r_t_final"].append(final if final is not None else 0.0)
        state["cum_rdp"].append(_as_float(m.get("cumulative_rdp")))
        remaining = self._remaining_rdp_sent.get(cid)
        state["remaining_rdp"].append(remaining if remaining is not None else None)
        state["phase"].append(phase)
        state["observed_m"].append(_as_float(m.get("observed_m")))
        acct = _as_float(m.get("acct_cost"))
        state["acct_cost"].append(acct if acct is not None else 0.0)
        for spec_key, metric_key in self._COMPONENT_KEYS:
            state[spec_key].append(_as_float(m.get(metric_key)))
        if phase == "warmup":
            state["warmup_rounds"].append(server_round)
        if phase != "exhausted":
            candidate = _as_float(m.get("r_t_candidate"))
            if candidate is not None and final is not None and candidate != final:
                state["enforcement_count"] += 1

    def _record_exhausted(self, server_round: int, replies: list[Message]) -> None:
        """Track refused rounds (dropout) and exhausted participations (IMPL-11)."""
        for reply in replies:
            if not _is_budget_exhausted(reply):
                continue
            m = reply.content.metric_records.get("metrics")
            if m is None:
                continue
            client_id = _as_float(m.get("client-id"))
            if client_id is None:
                continue
            cid = int(client_id)
            self._client_dropout_round.setdefault(cid, server_round)
            self._append_client_participation(cid, m, server_round)

    def get_client_state(self) -> dict[int, dict[str, Any]]:
        """Per-client per-participation accumulation for the final artifact."""
        return {
            cid: {**state, "dropout_round": self._client_dropout_round.get(cid)}
            for cid, state in self._client_state.items()
        }

    def get_bo_time_total(self) -> float:
        return self._bo_time_total

    def get_acct_time_total(self) -> float:
        return self._acct_time_total

    def _log_metric(self, key: str, value: float, step: int) -> None:
        if self._tracker is not None:
            self._tracker.log_metrics({key: value}, step=step)

    def _remaining_rdp_map(self) -> dict[int, float] | None:
        """Remaining per-client RDP budget (B_RDP - last reported cum_rdp).

        Keyed exactly like ``_per_client_budgets`` (node or partition ids);
        clients without a prior reply get the full budget (round 1).
        """
        if self._per_client_budgets is None:
            return None
        result: dict[int, float] = {}
        for key, budget in self._per_client_budgets.items():
            partition_id = self._node_to_partition.get(key, key)
            cum = self._client_cum_rdp.get(partition_id)
            if cum is None:
                cum = self._client_cum_eps.get(partition_id)
            result[key] = max(0.0, budget - (cum if cum is not None else 0.0))
        return result

    def _log_metric_stats(self, prefix: str, values: list[float], server_round: int) -> None:
        if not values:
            return
        arr = np.array(values)
        self._log_metric(f"{prefix}_mean", float(arr.mean()), step=server_round)
        self._log_metric(f"{prefix}_std", float(arr.std()), step=server_round)
        if len(values) >= _MIN_VALUES_FOR_STATS:
            self._log_metric(f"{prefix}_min", float(arr.min()), step=server_round)
            self._log_metric(f"{prefix}_max", float(arr.max()), step=server_round)
            self._log_metric(f"{prefix}_median", float(np.median(arr)), step=server_round)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate client val evaluations, logging diagnostics only (IMPL-08).

        The per-client hold-out accuracy/loss stay out of §4.3; they are logged
        as ``acc_val_mean``/``val_loss_mean`` purely for diagnostics.
        """
        metrics = super().aggregate_evaluate(server_round, replies)
        if metrics is not None:
            for key, target in (("loss", "val_loss_mean"), ("accuracy", "acc_val_mean")):
                raw = metrics.get(key)
                if isinstance(raw, (int, float, np.floating, np.integer)):
                    self._log_metric(target, float(raw), step=server_round)
        return metrics

    def _log_client_metrics(
        self,
        server_round: int,
        reply_contents: list[RecordDict],
    ) -> None:
        epsilons: list[float] = []
        client_epsilons: list[float] = []
        rdp_costs: list[float] = []
        client_rdps: list[float] = []
        cumulative_rdps: list[float] = []
        update_norms: list[float] = []
        utility_losses: list[float] = []
        utility_efficiencies: list[float] = []
        snrs: list[float] = []
        utility_losses_clean: list[float] = []
        utility_retentions: list[float] = []
        utility_per_remainings: list[float] = []
        logit_disagreements: list[float] = []
        cumulative_epsilons: list[float] = []
        sigmas: list[float] = []
        per_example_clip_fractions: list[float] = []
        grad_norms_before_clip: list[float] = []
        grad_norms_after_clip: list[float] = []
        num_opt_steps_list: list[float] = []
        r_t_finals: list[float] = []
        bo_times: list[float] = []
        acct_times: list[float] = []

        for content in reply_contents:
            m = content.metric_records.get("metrics")
            if m is None:
                continue
            client_id = m.get("client-id")
            if client_id is None:
                continue

            cid = int(client_id)

            r_t_final = _as_float(m.get("r_t_final"))
            if r_t_final is not None:
                r_t_finals.append(r_t_final)

            bo_time = _as_float(m.get("bo_time"))
            if bo_time is not None:
                bo_times.append(bo_time)
                self._bo_time_total += bo_time

            acct_time = _as_float(m.get("acct_time"))
            if acct_time is not None:
                acct_times.append(acct_time)
                self._acct_time_total += acct_time

            self._append_client_participation(cid, m, server_round)

            epsilon = m.get("epsilon")
            if epsilon is not None:
                epsilons.append(float(epsilon))
                self._log_metric(f"client_{cid}_epsilon", float(epsilon), step=server_round)

            update_norm = m.get("update_norm")
            if update_norm is not None:
                update_norms.append(float(update_norm))
                self._log_metric(f"client_{cid}_update_norm", float(update_norm), step=server_round)

            utility_loss = m.get("utility_loss")
            if utility_loss is not None:
                utility_losses.append(float(utility_loss))
                self._log_metric(
                    f"client_{cid}_utility_loss",
                    float(utility_loss),
                    step=server_round,
                )

            utility_eff = m.get("utility_efficiency")
            if utility_eff is not None:
                utility_efficiencies.append(float(utility_eff))
                self._log_metric(
                    f"client_{cid}_utility_efficiency",
                    float(utility_eff),
                    step=server_round,
                )

            snr_val = m.get("snr")
            if snr_val is not None:
                snrs.append(float(snr_val))
                self._log_metric(f"client_{cid}_snr", float(snr_val), step=server_round)

            utility_loss_clean = m.get("utility_loss_clean")
            if utility_loss_clean is not None:
                utility_losses_clean.append(float(utility_loss_clean))
                self._log_metric(
                    f"client_{cid}_utility_loss_clean",
                    float(utility_loss_clean),
                    step=server_round,
                )

            utility_ret = m.get("utility_retention")
            if utility_ret is not None:
                utility_retentions.append(float(utility_ret))
                self._log_metric(
                    f"client_{cid}_utility_retention",
                    float(utility_ret),
                    step=server_round,
                )

            utility_per_rem = m.get("utility_per_remaining")
            if utility_per_rem is not None:
                utility_per_remainings.append(float(utility_per_rem))
                self._log_metric(
                    f"client_{cid}_utility_per_remaining",
                    float(utility_per_rem),
                    step=server_round,
                )

            logit_disagreement_val = m.get("logit_disagreement")
            if logit_disagreement_val is not None:
                logit_disagreements.append(float(logit_disagreement_val))
                self._log_metric(
                    f"client_{cid}_logit_disagreement",
                    float(logit_disagreement_val),
                    step=server_round,
                )

            cum_eps_val = m.get("cumulative_epsilon")
            cum_eps = float(cum_eps_val) if cum_eps_val is not None else None
            if cum_eps is not None:
                cumulative_epsilons.append(cum_eps)
                self._client_cum_eps[cid] = cum_eps
                self._log_metric(f"client_{cid}_cumulative_epsilon", cum_eps, step=server_round)

            client_eps = m.get("client_epsilon")
            if client_eps is not None:
                client_epsilons.append(float(client_eps))
                self._log_metric(
                    f"client_{cid}_client_epsilon",
                    float(client_eps),
                    step=server_round,
                )

            rdp_cost = m.get("rdp_cost")
            if rdp_cost is not None:
                rdp_costs.append(float(rdp_cost))
                self._log_metric(f"client_{cid}_rdp_cost", float(rdp_cost), step=server_round)

            client_rdp = m.get("client_rdp")
            if client_rdp is not None:
                client_rdps.append(float(client_rdp))
                self._log_metric(f"client_{cid}_client_rdp", float(client_rdp), step=server_round)

            cum_rdp_val = m.get("cumulative_rdp")
            cum_rdp = float(cum_rdp_val) if cum_rdp_val is not None else None
            if cum_rdp is not None:
                cumulative_rdps.append(cum_rdp)
                self._client_cum_rdp[cid] = cum_rdp
                self._log_metric(f"client_{cid}_cumulative_rdp", cum_rdp, step=server_round)

            sigma = m.get("sigma")
            if sigma is not None:
                sigmas.append(float(sigma))
                self._log_metric(f"client_{cid}_sigma", float(sigma), step=server_round)

            pe_clip = m.get("per_example_clip_fraction")
            if pe_clip is not None:
                per_example_clip_fractions.append(float(pe_clip))
                self._log_metric(
                    f"client_{cid}_per_example_clip_fraction",
                    float(pe_clip),
                    step=server_round,
                )

            gn_before = m.get("grad_norm_before_clip")
            if gn_before is not None:
                grad_norms_before_clip.append(float(gn_before))
                self._log_metric(
                    f"client_{cid}_grad_norm_before_clip",
                    float(gn_before),
                    step=server_round,
                )

            gn_after = m.get("grad_norm_after_clip")
            if gn_after is not None:
                grad_norms_after_clip.append(float(gn_after))
                self._log_metric(
                    f"client_{cid}_grad_norm_after_clip",
                    float(gn_after),
                    step=server_round,
                )

            nos = m.get("num_opt_steps")
            if nos is not None:
                num_opt_steps_list.append(float(nos))
                self._log_metric(
                    f"client_{cid}_num_opt_steps",
                    float(nos),
                    step=server_round,
                )

            if self._per_client_budgets is not None and cum_eps is not None:
                budget = self._per_client_budgets.get(cid)
                if budget is None and self._node_to_partition:
                    reversed_map = {v: k for k, v in self._node_to_partition.items()}
                    node_id = reversed_map.get(cid)
                    if node_id is not None:
                        budget = self._per_client_budgets.get(node_id)
                if budget is not None:
                    remaining = max(0.0, budget - float(cum_eps))
                    self._log_metric(f"client_{cid}_remaining_budget", remaining, step=server_round)

            if self._per_client_budgets is not None and cum_rdp is not None:
                budget = self._per_client_budgets.get(cid)
                if budget is None and self._node_to_partition:
                    reversed_map = {v: k for k, v in self._node_to_partition.items()}
                    node_id = reversed_map.get(cid)
                    if node_id is not None:
                        budget = self._per_client_budgets.get(node_id)
                if budget is not None:
                    remaining = max(0.0, budget - float(cum_rdp))
                    self._log_metric(
                        f"client_{cid}_remaining_rdp_budget",
                        remaining,
                        step=server_round,
                    )

        self._log_metric_stats("epsilon", epsilons, server_round)
        self._log_metric_stats("client_epsilon", client_epsilons, server_round)
        self._log_metric_stats("rdp_cost", rdp_costs, server_round)
        self._log_metric_stats("client_rdp", client_rdps, server_round)
        self._log_metric_stats("cumulative_rdp", cumulative_rdps, server_round)
        self._log_metric_stats("update_norm", update_norms, server_round)
        self._log_metric_stats("utility_loss", utility_losses, server_round)
        self._log_metric_stats("utility_efficiency", utility_efficiencies, server_round)
        self._log_metric_stats("snr", snrs, server_round)
        self._log_metric_stats("utility_loss_clean", utility_losses_clean, server_round)
        self._log_metric_stats("utility_retention", utility_retentions, server_round)
        self._log_metric_stats("utility_per_remaining", utility_per_remainings, server_round)
        self._log_metric_stats("logit_disagreement", logit_disagreements, server_round)
        self._log_metric_stats("cumulative_epsilon", cumulative_epsilons, server_round)
        self._log_metric_stats("sigma", sigmas, server_round)
        self._log_metric_stats(
            "per_example_clip_fraction",
            per_example_clip_fractions,
            server_round,
        )
        self._log_metric_stats("grad_norm_before_clip", grad_norms_before_clip, server_round)
        self._log_metric_stats("grad_norm_after_clip", grad_norms_after_clip, server_round)
        self._log_metric_stats("num_opt_steps", num_opt_steps_list, server_round)

        participants = 0
        received_bytes = 0
        for c in reply_contents:
            mm = c.metric_records.get("metrics")
            if mm is None or mm.get("client-id") is None:
                continue
            participants += 1
            received_bytes += self._array_bytes(c.array_records.get(self.arrayrecord_key))
        self._log_metric("n_participants", float(participants), step=server_round)
        self._log_metric(
            "bytes_round",
            float(self._bytes_sent_round) + float(received_bytes),
            step=server_round,
        )
        if r_t_finals:
            self._log_metric("mean_r_t", float(np.mean(r_t_finals)), step=server_round)
        if cumulative_rdps:
            self._log_metric(
                "mean_cum_rdp",
                float(np.mean(cumulative_rdps)),
                step=server_round,
            )
        if bo_times:
            self._log_metric("bo_time_round", float(np.mean(bo_times)), step=server_round)
        if acct_times:
            self._log_metric(
                "acct_time_round",
                float(np.mean(acct_times)),
                step=server_round,
            )


class MedianRobustAggregation(MetricLoggingMixin, FedAvg):
    def __init__(
        self,
        server_learning_rate: float = 1.0,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn=None,
        evaluate_metrics_aggr_fn=None,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}
        self._client_cum_rdp = {}
        self._client_cum_eps = {}
        self._init_spec_state()

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        remaining_rdp_by_client = self._remaining_rdp_map()
        self._remaining_rdp_sent = remaining_rdp_by_client or {}
        messages = list(
            _add_budgets_to_messages(
                super().configure_train(server_round, arrays, config, grid),
                self._per_client_budgets,
                self.configrecord_key,
                node_to_partition=self._node_to_partition,
                remaining_rdp_by_client=remaining_rdp_by_client,
            )
        )
        self._bytes_sent_round = sum(
            self._array_bytes(m.content.array_records.get(self.arrayrecord_key)) for m in messages
        )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        self._record_exhausted(server_round, replies_list)
        valid_replies = _filter_valid_replies(replies_list)

        if not valid_replies:
            return None, None

        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)

        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None

        global_arrays = self._current_arrays
        if global_arrays is None:
            return None, None

        global_keys = list(global_arrays.keys())
        global_ndarrays = [global_arrays[k].numpy() for k in global_keys]

        deltas: list[list[np.ndarray]] = []
        norms: list[float] = []
        reply_contents: list[RecordDict] = []

        for msg in active_replies:
            content = msg.content
            client_arrays = content.array_records.get(self.arrayrecord_key)
            if client_arrays is None:
                continue

            client_ndarrays = [client_arrays[k].numpy() for k in global_keys]

            delta = [c - g for c, g in zip(client_ndarrays, global_ndarrays, strict=True)]
            flat = np.concatenate([d.ravel() for d in delta])
            norm = float(np.linalg.norm(flat))

            deltas.append(delta)
            norms.append(norm)
            reply_contents.append(content)

        if not deltas:
            return None, None

        median_norm = float(np.median(norms))
        weights = np.array(
            [1.0 if r <= _EPS else min(1.0, median_norm / r) for r in norms],
            dtype=np.float64,
        )

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return None, None

        norm_weights = weights / total_weight

        num_layers = len(global_ndarrays)
        aggregated_delta = [
            np.sum(
                np.array([nw * d[i] for nw, d in zip(norm_weights, deltas, strict=True)]),
                axis=0,
            )
            for i in range(num_layers)
        ]

        new_ndarrays = [
            g + self._server_learning_rate * ad
            for g, ad in zip(global_ndarrays, aggregated_delta, strict=True)
        ]

        aggregated = ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in zip(global_keys, new_ndarrays, strict=True)},
        )

        metrics = None
        if self.train_metrics_aggr_fn is not None:
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return aggregated, metrics


def _is_budget_exhausted(reply: Message) -> bool:
    metrics_rec = reply.content.metric_records.get("metrics")
    if metrics_rec is None:
        return False
    raw = metrics_rec.get("budget_exhausted")
    if raw is None:
        return False
    if isinstance(raw, (list, tuple)):
        return any(bool(v) for v in raw)
    return bool(raw)


class SafeFedAvg(MetricLoggingMixin, FedAvg):
    def __init__(
        self,
        *args,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
        server_learning_rate: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}
        self._client_cum_rdp = {}
        self._client_cum_eps = {}
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None
        self._init_spec_state()

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        remaining_rdp_by_client = self._remaining_rdp_map()
        self._remaining_rdp_sent = remaining_rdp_by_client or {}
        messages = list(
            _add_budgets_to_messages(
                super().configure_train(server_round, arrays, config, grid),
                self._per_client_budgets,
                self.configrecord_key,
                node_to_partition=self._node_to_partition,
                remaining_rdp_by_client=remaining_rdp_by_client,
            )
        )
        self._bytes_sent_round = sum(
            self._array_bytes(m.content.array_records.get(self.arrayrecord_key)) for m in messages
        )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        self._record_exhausted(server_round, replies_list)
        valid_replies = _filter_valid_replies(replies_list)
        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)
        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None
        result_arrays, metrics = super().aggregate_train(server_round, active_replies)

        if result_arrays is not None and self._current_arrays is not None:
            global_keys = list(self._current_arrays.keys())
            scaled: dict[str, Array] = {}
            for k in global_keys:
                delta = result_arrays[k].numpy() - self._current_arrays[k].numpy()
                scaled[k] = Array(
                    self._current_arrays[k].numpy() + self._server_learning_rate * delta,
                )
            result_arrays = ArrayRecord(scaled)

        return result_arrays, metrics


class SafeFedProx(MetricLoggingMixin, FedProx):
    def __init__(
        self,
        *args,
        tracker: ExperimentTracker | None = None,
        per_client_budgets: dict[int, float] | None = None,
        node_to_partition: dict[int, int] | None = None,
        server_learning_rate: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracker = tracker
        self._per_client_budgets = per_client_budgets
        self._node_to_partition = node_to_partition or {}
        self._client_cum_rdp = {}
        self._client_cum_eps = {}
        self._server_learning_rate = server_learning_rate
        self._current_arrays: ArrayRecord | None = None
        self._init_spec_state()

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        remaining_rdp_by_client = self._remaining_rdp_map()
        self._remaining_rdp_sent = remaining_rdp_by_client or {}
        messages = list(
            _add_budgets_to_messages(
                super().configure_train(server_round, arrays, config, grid),
                self._per_client_budgets,
                self.configrecord_key,
                node_to_partition=self._node_to_partition,
                remaining_rdp_by_client=remaining_rdp_by_client,
            )
        )
        self._bytes_sent_round = sum(
            self._array_bytes(m.content.array_records.get(self.arrayrecord_key)) for m in messages
        )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        self._record_exhausted(server_round, replies_list)
        valid_replies = _filter_valid_replies(replies_list)
        all_contents = [r.content for r in valid_replies]
        self._log_client_metrics(server_round, all_contents)
        active_replies = [r for r in valid_replies if not _is_budget_exhausted(r)]
        if not active_replies:
            return None, None
        result_arrays, metrics = super().aggregate_train(server_round, active_replies)

        if result_arrays is not None and self._current_arrays is not None:
            global_keys = list(self._current_arrays.keys())
            scaled: dict[str, Array] = {}
            for k in global_keys:
                delta = result_arrays[k].numpy() - self._current_arrays[k].numpy()
                scaled[k] = Array(
                    self._current_arrays[k].numpy() + self._server_learning_rate * delta,
                )
            result_arrays = ArrayRecord(scaled)

        return result_arrays, metrics
