from __future__ import annotations

from collections.abc import Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

CLASSES_PER_CLIENT_PATHOLOGICAL = 2
WRITER_MERGE_THRESHOLD = 10


def _extract_targets(dataset: Dataset[Any]) -> torch.Tensor:
    """Return the label vector for a dataset (or a Subset view of one)."""
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, "targets"):
            return torch.as_tensor(dataset.dataset.targets)[dataset.indices]
        if hasattr(dataset.dataset, "tensors"):
            return cast(torch.Tensor, dataset.dataset.tensors[1][dataset.indices])
        raise ValueError(
            "Underlying dataset must have .targets or .tensors attribute "
            "for label-based partition"
        )
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets)
    if hasattr(dataset, "tensors"):
        return cast(torch.Tensor, dataset.tensors[1])
    raise ValueError(
        "Dataset must have .targets or .tensors attribute for label-based partition"
    )


def _extract_users(dataset: Dataset[Any]) -> torch.Tensor:
    """Return the per-sample writer-id vector (dataset must expose .users)."""
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, "users"):
            return torch.as_tensor(cast(Any, dataset.dataset).users)[dataset.indices]
        raise ValueError(
            "Underlying dataset must have a .users attribute for writer partition"
        )
    if hasattr(dataset, "users"):
        return torch.as_tensor(cast(Any, dataset).users)
    raise ValueError(
        "Dataset must have a .users attribute for writer partition"
    )


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    a = a + 1e-10
    b = b + 1e-10
    return float(np.sum(a * np.log(a / b)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two (normalized) histograms."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _plan_writer_partition(
    dataset: Dataset[Any],
    num_clients: int,
) -> list[list[int]]:
    """Writer partition (FEMNIST, spec §9.9): one client per writer cluster.

    Documented rule (user-approved, IMPL-07): clients are the *largest*
    writers by train-sample count (ties by lowest writer id). Every other
    writer with fewer than WRITER_MERGE_THRESHOLD samples is merged into the
    nearest client cluster — minimum JS divergence between the writer's label
    histogram and the client's *initial* label histogram (ties by lowest
    client id); merges proceed smallest-first (ties by lowest writer id).
    Remaining writers (>= threshold but not among the top clients) are
    excluded from the partition. The mapping is a pure function of (users,
    targets, num_clients): seed-independent, so partition_single is
    guaranteed consistent with partition_dataset. The min-30 rule is NOT
    applied (FEMNIST is exempt from the §3 matrix minimum).
    """
    users = _extract_users(dataset)
    targets = _extract_targets(dataset)
    n_writers = int(users.max().item()) + 1
    if num_clients > n_writers:
        raise ValueError(
            f"writer partition requires num_clients ({num_clients}) <= "
            f"number of writers ({n_writers})"
        )
    num_classes = int(torch.unique(targets).numel())
    writer_sizes = [int((users == w).sum().item()) for w in range(n_writers)]
    ranked = sorted(range(n_writers), key=lambda w: (-writer_sizes[w], w))
    clients = ranked[:num_clients]

    client_of: dict[int, int | None] = {w: None for w in range(n_writers)}
    for cid, writer in enumerate(clients):
        client_of[writer] = cid

    def histogram(mask: torch.Tensor) -> np.ndarray:
        counts = np.asarray(
            torch.bincount(targets[mask].long(), minlength=num_classes).numpy(),
            dtype=float,
        )
        return cast(np.ndarray, counts / counts.sum())

    client_hist = [histogram(users == w) for w in clients]

    small = sorted(
        (w for w in range(n_writers)
         if writer_sizes[w] < WRITER_MERGE_THRESHOLD and client_of[w] is None),
        key=lambda w: (writer_sizes[w], w),
    )
    for writer in small:
        hist = histogram(users == writer)
        best = min(
            range(num_clients), key=lambda c: _js_divergence(client_hist[c], hist),
        )
        client_of[writer] = best

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for w_id, c in client_of.items():
        if c is not None:
            client_indices[c].extend(torch.where(users == w_id)[0].tolist())
    return client_indices


def partition_iid(
    dataset: Dataset[Any], num_clients: int, seed: int | None = None
) -> list[Subset[Any]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(cast(Sized, dataset)))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def _plan_dirichlet_partition(
    dataset: Dataset[Any],
    num_clients: int,
    alpha: float,
    seed: int | None,
) -> list[list[int]]:
    """Seeded Dirichlet label-skew plan: full per-client index lists.

    A single rng stream, consumed identically on every call, so the full
    partition and any single client's partition are guaranteed consistent
    (IMPL-05, item 4).
    """
    rng = np.random.RandomState(seed)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    targets = _extract_targets(dataset)
    num_classes = int(torch.unique(targets).numel())
    class_indices = [torch.where(targets == c)[0] for c in range(num_classes)]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        perm = torch.randperm(len(class_indices[c]), generator=generator)
        class_indices[c] = class_indices[c][perm]

        raw_proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(raw_proportions, 1e-6)
        proportions /= proportions.sum()
        sizes = (proportions * len(class_indices[c])).astype(int)
        diff = len(class_indices[c]) - sizes.sum()
        if diff != 0:
            sizes[-1] += diff

        offset = 0
        for i in range(num_clients):
            if sizes[i] > 0:
                client_indices[i].extend(
                    class_indices[c][offset:offset + sizes[i]].tolist()
                )
            offset += sizes[i]

    empty = [i for i in range(num_clients) if len(client_indices[i]) == 0]
    if empty:
        all_selected = set(idx for client in client_indices for idx in client)
        all_indices = list(set(range(len(cast(Sized, dataset)))) - all_selected)
        if not all_indices:
            for eid in empty:
                donor = max(
                    (i for i in range(num_clients) if len(client_indices[i]) > 0),
                    key=lambda i: len(client_indices[i]),
                    default=None,
                )
                if donor is None:
                    break
                moved = client_indices[donor].pop()
                client_indices[eid].append(moved)
        else:
            rng.shuffle(all_indices)
            filler = np.array_split(all_indices, len(empty))
            for i, eid in enumerate(empty):
                client_indices[eid] = filler[i].tolist()

    return client_indices


def partition_noniid_dirichlet(
    dataset: Dataset[Any],
    num_clients: int,
    alpha: float = 1.0,
    seed: int | None = None,
    partition_id: int | None = None,
) -> list[Subset[Any]] | Subset[Any]:
    client_indices = _plan_dirichlet_partition(dataset, num_clients, alpha, seed)
    if partition_id is not None:
        return Subset(dataset, client_indices[partition_id])
    return [Subset(dataset, idxs) for idxs in client_indices]


def partition_pathological(
    dataset: Dataset[Any],
    num_clients: int,
    seed: int | None = None,
) -> list[list[int]]:
    """Pathological partition: exactly 2 non-overlapping classes per client.

    Class -> client assignment: iterating clients in order, each takes the two
    least-covered classes (ties broken by class id), so every class is covered
    by exactly ceil(2*K/C) clients when 2*K is divisible by C (MNIST: 20/class
    at K=100, C=10; CIFAR-100: 2/class at K=100, C=100). Per class the samples
    are seeded-shuffled and split into balanced chunks, one per covering
    client, so every dataset sample is assigned to exactly one client.
    """
    rng = np.random.RandomState(seed)
    targets = _extract_targets(dataset)
    num_classes = int(torch.unique(targets).numel())
    class_indices = [torch.where(targets == c)[0].tolist() for c in range(num_classes)]

    cover_counts = [0] * num_classes
    client_classes: list[list[int]] = []
    for _ in range(num_clients):
        ordered = sorted(range(num_classes), key=lambda c: (cover_counts[c], c))
        pair = ordered[:CLASSES_PER_CLIENT_PATHOLOGICAL]
        for c in pair:
            cover_counts[c] += 1
        client_classes.append(pair)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        covering = [i for i, cls in enumerate(client_classes) if c in cls]
        rng.shuffle(class_indices[c])
        chunks = np.array_split(np.array(class_indices[c]), len(covering))
        for j, cid in enumerate(covering):
            client_indices[cid].extend(chunks[j].tolist())

    return client_indices


def _enforce_min_samples(
    client_indices: list[list[int]],
    min_samples: int,
) -> list[list[int]]:
    """Top-up deficient clients from the largest donors (deterministic).

    Documented rule (IMPL-05, item 3): deficient clients (size < min_samples)
    are processed smallest-first (ties by lowest client id); each takes the
    samples it needs from the largest donor (ties by lowest id), a donor
    giving at most (donor_size - min_samples) so it never drops below the
    minimum. Transferred samples are the donor's trailing slice, so the rule
    depends only on client sizes and sample order — reproducible in
    partition_single. Degenerate case: if the donor pool is exhausted (total
    data below K * min_samples), remaining clients stay below the minimum;
    impossible for the paper cells (MNIST/CIFAR-100 sizes >> 30).
    """
    if min_samples <= 0:
        return client_indices
    sizes = [len(c) for c in client_indices]
    deficient = sorted(
        (i for i, s in enumerate(sizes) if s < min_samples),
        key=lambda i: (sizes[i], i),
    )
    for i in deficient:
        needed = min_samples - sizes[i]
        donors = sorted(
            (j for j in range(len(client_indices)) if j != i and sizes[j] > min_samples),
            key=lambda j: (-sizes[j], j),
        )
        for d in donors:
            if needed <= 0:
                break
            give = min(needed, sizes[d] - min_samples)
            if give <= 0:
                continue
            client_indices[i].extend(client_indices[d][-give:])
            del client_indices[d][-give:]
            sizes[d] -= give
            sizes[i] += give
            needed -= give
    return client_indices


def _plan_partition(
    dataset: Dataset[Any],
    num_clients: int,
    partition_type: str,
    alpha: float,
    seed: int | None,
) -> list[list[int]]:
    if partition_type == "iid":
        rng = np.random.RandomState(seed)
        indices = np.arange(len(cast(Sized, dataset)))
        rng.shuffle(indices)
        return [split.tolist() for split in np.array_split(indices, num_clients)]
    if partition_type in ("dirichlet", "noniid"):
        # noniid is a deprecated alias of dirichlet with alpha=0.5 (archive
        # configs still use it).
        dirichlet_alpha = 0.5 if partition_type == "noniid" else alpha
        return _plan_dirichlet_partition(dataset, num_clients, dirichlet_alpha, seed)
    if partition_type == "pathological":
        return partition_pathological(dataset, num_clients, seed)
    if partition_type == "writer":
        return _plan_writer_partition(dataset, num_clients)
    raise ValueError(f"Unknown partition type: {partition_type}")


def partition_dataset(
    dataset: Dataset[Any],
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 1.0,
    seed: int | None = None,
    min_samples: int = 0,
) -> list[Subset[Any]]:
    client_indices = _plan_partition(dataset, num_clients, partition_type, alpha, seed)
    if partition_type != "writer":
        client_indices = _enforce_min_samples(client_indices, min_samples)
    return [Subset(dataset, idxs) for idxs in client_indices]


def partition_single(
    dataset: Dataset[Any],
    num_clients: int,
    partition_id: int,
    partition_type: str = "iid",
    alpha: float = 1.0,
    seed: int = 42,
    min_samples: int = 0,
) -> Subset[Any]:
    """Return only the requested client's partition.

    The full index plan is computed (O(dataset) integers) so that the
    min-30 top-up rule reproduces the full partition exactly; only the
    requested client's Subset is materialized.
    """
    if partition_id < 0 or partition_id >= num_clients:
        raise ValueError(
            f"partition_id {partition_id} out of range [0, {num_clients})"
        )
    client_indices = _plan_partition(dataset, num_clients, partition_type, alpha, seed)
    if partition_type != "writer":
        client_indices = _enforce_min_samples(client_indices, min_samples)
    return Subset(dataset, client_indices[partition_id])


def build_partition_kwargs(partition_type: str, alpha: float = 1.0) -> dict[str, float | int | str]:
    """JSON-serializable partition description for the §4.2 partition_kwargs param."""
    if partition_type == "noniid":
        return {"type": "dirichlet", "alpha": 0.5}
    if partition_type == "dirichlet":
        return {"type": "dirichlet", "alpha": alpha}
    if partition_type == "pathological":
        return {
            "type": "pathological",
            "classes_per_client": CLASSES_PER_CLIENT_PATHOLOGICAL,
        }
    if partition_type == "writer":
        return {
            "type": "writer",
            "merge_threshold": WRITER_MERGE_THRESHOLD,
        }
    return {"type": partition_type}


def split_holdout(
    subset: Subset[Any],
    val_frac: float,
    seed: int,
) -> tuple[Subset[Any], Subset[Any]]:
    """Split a client partition into (train, val) subsets with a seeded hold-out.

    The split is a pure function of (subset indices, val_frac, seed): it uses a
    dedicated RandomState, so it is fixed across rounds and does not consume
    the global RNG. The returned subsets index the *same* underlying dataset,
    so train and val indices are disjoint by construction.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    rng = np.random.RandomState(seed)
    indices = np.asarray(subset.indices)
    rng.shuffle(indices)
    val_size = int(len(indices) * val_frac)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return (
        Subset(subset.dataset, train_indices),
        Subset(subset.dataset, val_indices),
    )
