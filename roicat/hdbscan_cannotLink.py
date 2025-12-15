# fast_hdbscan/precomputed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Union

import numba
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as sp_csgraph

from fast_hdbscan.disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from fast_hdbscan.hdbscan import clusters_from_spanning_tree

Number = Union[int, float]
CSR = sp.csr_matrix


# ------------------------------- constraints -------------------------------


@dataclass
class MergeConstraint:
    """
    A thin wrapper around hard merge constraints used during MST construction.

    The constrained MST builder only needs one operation:
        any_cannot_link_across(component_a, component_b) -> bool

    Optionally, we also support:
        pair_cannot_link(i, j) -> bool
    which enables a "soft" mode that only penalizes *direct* forbidden edges.

    Notes:
        - This class is intentionally minimal.
        - If you want intragroup constraints, implement pair_cannot_link(i,j)
          in terms of group IDs (or any other metadata) and pass it in via
          `from_pair_cannot_link`.
    """

    any_cannot_link_across: Callable[[np.ndarray, np.ndarray], bool]
    pair_cannot_link: Optional[Callable[[int, int], bool]] = None
    iter_cannot_link_pairs: Optional[Callable[[], Iterable[Tuple[int, int]]]] = None

    # Optional CSR payload for fast strict=True JIT path.
    cannot_link_indptr: Optional[np.ndarray] = None  # shape (N+1,), int64
    cannot_link_indices: Optional[np.ndarray] = None  # shape (nnz,), int32

    @staticmethod
    def from_cannot_link_matrix(
        cannot_link: Union[np.ndarray, CSR],
        *,
        n_points: int,
    ) -> "MergeConstraint":
        """
        Build a MergeConstraint from a dense (N,N) bool array or sparse CSR bool matrix.

        Args:
            cannot_link:
                Dense or sparse structure where cannot_link[i,j] == True means
                points i and j must never be merged into the same MST component.
            n_points:
                Number of points, N.

        Returns:
            (MergeConstraint):
                A constraint object with fast `any_cannot_link_across` checks.
        """
        if sp.issparse(cannot_link):
            cannot_link_csr = cannot_link.tocsr().astype(bool)
            if cannot_link_csr.shape != (n_points, n_points):
                raise ValueError("cannot_link sparse matrix must have shape (N, N).")

            # enforce symmetry and zero diagonal (CRITICAL for strict=True correctness)
            cannot_link_csr.setdiag(False)
            cannot_link_csr.eliminate_zeros()
            cannot_link_csr = ((cannot_link_csr + cannot_link_csr.T) > 0).tocsr()
            cannot_link_csr.setdiag(False)
            cannot_link_csr.eliminate_zeros()
            cannot_link_csr.sum_duplicates()
            cannot_link_csr.sort_indices()

            def any_cannot_link_across_sparse(comp_a: np.ndarray, comp_b: np.ndarray) -> bool:
                comp_b_set = set(int(x) for x in comp_b.tolist())
                for idx_a in comp_a.tolist():
                    row_indices = cannot_link_csr.getrow(int(idx_a)).indices
                    for idx_b in row_indices:
                        if int(idx_b) in comp_b_set:
                            return True
                return False

            def pair_cannot_link_sparse(i: int, j: int) -> bool:
                row_indices = cannot_link_csr.getrow(int(i)).indices
                return bool(np.any(row_indices == int(j)))

            def iter_pairs_sparse() -> Iterable[Tuple[int, int]]:
                coo = sp.triu(cannot_link_csr, k=1).tocoo()
                for i, j in zip(coo.row.tolist(), coo.col.tolist()):
                    yield int(i), int(j)

            cannot_link_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
            cannot_link_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

            return MergeConstraint(
                any_cannot_link_across=any_cannot_link_across_sparse,
                pair_cannot_link=pair_cannot_link_sparse,
                iter_cannot_link_pairs=iter_pairs_sparse,
                cannot_link_indptr=cannot_link_indptr,
                cannot_link_indices=cannot_link_indices,
            )

        cannot_link_dense = np.asarray(cannot_link, dtype=bool)
        if cannot_link_dense.shape != (n_points, n_points):
            raise ValueError("cannot_link dense array must have shape (N, N).")

        # enforce symmetry and zero diagonal (CRITICAL for strict=True correctness)
        np.fill_diagonal(cannot_link_dense, False)
        cannot_link_dense = np.logical_or(cannot_link_dense, cannot_link_dense.T)

        def any_cannot_link_across_dense(comp_a: np.ndarray, comp_b: np.ndarray) -> bool:
            return bool(np.any(cannot_link_dense[np.ix_(comp_a, comp_b)]))

        def pair_cannot_link_dense(i: int, j: int) -> bool:
            return bool(cannot_link_dense[int(i), int(j)])

        def iter_pairs_dense() -> Iterable[Tuple[int, int]]:
            for i in range(n_points):
                js = np.flatnonzero(cannot_link_dense[i])
                for j in js.tolist():
                    if j > i:
                        yield int(i), int(j)

        # Build CSR payload for the JIT strict=True path (intended for small/medium N).
        cannot_link_csr = sp.csr_matrix(cannot_link_dense)
        cannot_link_csr.setdiag(False)
        cannot_link_csr.eliminate_zeros()
        cannot_link_csr.sum_duplicates()
        cannot_link_csr.sort_indices()
        cannot_link_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cannot_link_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        return MergeConstraint(
            any_cannot_link_across=any_cannot_link_across_dense,
            pair_cannot_link=pair_cannot_link_dense,
            iter_cannot_link_pairs=iter_pairs_dense,
            cannot_link_indptr=cannot_link_indptr,
            cannot_link_indices=cannot_link_indices,
        )

    @staticmethod
    def from_pair_cannot_link(
        pair_cannot_link: Callable[[int, int], bool],
    ) -> "MergeConstraint":
        """
        Build a MergeConstraint from a pairwise callback.

        This is intentionally thin and may be slow for large components because
        it falls back to nested loops to answer:
            "is there any forbidden pair across these two components?"
        """

        def any_cannot_link_across_func(comp_a: np.ndarray, comp_b: np.ndarray) -> bool:
            for i in comp_a.tolist():
                for j in comp_b.tolist():
                    if pair_cannot_link(int(i), int(j)):
                        return True
            return False

        return MergeConstraint(
            any_cannot_link_across=any_cannot_link_across_func,
            pair_cannot_link=pair_cannot_link,
            iter_cannot_link_pairs=None,  # cannot enumerate in general
            cannot_link_indptr=None,
            cannot_link_indices=None,
        )


# ------------------------------- helpers ---------------------------------


@numba.njit(cache=True)
def _has_cannot_link_violation_csr_numba(
    labels: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    noise_label: int,
) -> bool:
    """
    Fast violation check:
        returns True if there exists (i,j) cannot-link with labels[i]==labels[j]!=noise_label.

    Assumes CSR adjacency; does not require symmetry but will catch violations
    as long as at least one direction of (i,j) is present.
    """
    n = labels.shape[0]
    for i in range(n):
        li = labels[i]
        if li == noise_label:
            continue
        start = cannot_link_indptr[i]
        end = cannot_link_indptr[i + 1]
        for k in range(start, end):
            j = cannot_link_indices[k]
            if j == i:
                continue
            if labels[j] == li:
                return True
    return False


def _maybe_split_labels_if_cannot_link_violated(
    labels: np.ndarray,
    *,
    merge_constraint: "MergeConstraint",
    distances: Optional[Union[np.ndarray, CSR]],
    noise_label: int = -1,
) -> np.ndarray:
    """
    If posthoc cleanup is requested, avoid doing any splitting work unless
    there is an actual cannot-link violation.
    """
    labels_in = np.asarray(labels, dtype=np.int64)

    if merge_constraint.pair_cannot_link is None:
        raise ValueError("posthoc_cleanup requires merge_constraint.pair_cannot_link to be available.")

    has_violation = False

    # ---- Path 1: numba CSR scan if payload exists ----
    if (merge_constraint.cannot_link_indptr is not None) and (merge_constraint.cannot_link_indices is not None):
        cannot_link_indptr = np.asarray(merge_constraint.cannot_link_indptr, dtype=np.int64)
        cannot_link_indices = np.asarray(merge_constraint.cannot_link_indices, dtype=np.int64)

        if cannot_link_indptr.ndim == 1 and cannot_link_indptr.size == labels_in.shape[0] + 1:
            has_violation = bool(
                _has_cannot_link_violation_csr_numba(
                    labels_in,
                    cannot_link_indptr,
                    cannot_link_indices,
                    int(noise_label),
                )
            )
        else:
            has_violation = True

    # ---- Path 2: pair iterator if available ----
    elif merge_constraint.iter_cannot_link_pairs is not None:
        violations = find_cannot_link_violations(labels_in, merge_constraint=merge_constraint, noise_label=int(noise_label))
        has_violation = bool(violations.shape[0] > 0)

    # ---- Path 3: cannot cheaply check -> conservatively split ----
    else:
        has_violation = True

    if not has_violation:
        return labels_in  # noop

    return split_clusters_to_respect_cannot_link(
        labels_in,
        merge_constraint=merge_constraint,
        distances=distances,
        noise_label=int(noise_label),
    )


@numba.njit(cache=True)
def _dsu_find_numba(parent: np.ndarray, x: int) -> int:
    """
    DSU find with path compression (Numba).
    """
    root = x
    while parent[root] != root:
        root = parent[root]

    while parent[x] != x:
        px = parent[x]
        parent[x] = root
        x = px

    return root


@numba.njit(cache=True)
def _csr_row_contains_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    row: int,
    value: int,
) -> bool:
    """
    Linear membership test in a CSR row (Numba-safe, does not assume sorting).
    """
    start = int(indptr[row])
    end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == value:
            return True
    return False


@numba.njit(cache=True)
def _component_has_conflict_with_root_numba(
    root_small: int,
    root_large: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
) -> bool:
    """
    Check whether merging component(root_small) into component(root_large) would
    violate any cannot-link constraints.

    For each node i in the smaller component:
        for each forbidden neighbor j in cannot_link[i]:
            if find(j) == root_large -> conflict
    """
    node = head[root_small]
    while node != -1:
        start = int(cannot_link_indptr[node])
        end = int(cannot_link_indptr[node + 1])
        for k in range(start, end):
            j = int(cannot_link_indices[k])
            if _dsu_find_numba(parent, j) == root_large:
                return True
        node = next_node[node]
    return False


@numba.njit(cache=True)
def _constrained_kruskal_mst_csr_strict_sorted_numba(
    u_sorted: np.ndarray,
    v_sorted: np.ndarray,
    w_sorted: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Strict constrained Kruskal (Numba) for CSR cannot-link adjacency.

    Inputs u_sorted, v_sorted, w_sorted MUST be sorted by increasing w.

    Returns:
        mst_edges (np.ndarray):
            Shape (<=N-1, 3), float64. Columns: [u, v, w]
            This may be a forest if constraints prevent full connectivity.
    """
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)

    # Per-component membership as a linked list stored in arrays.
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)

    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1

    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    for idx_edge in range(u_sorted.shape[0]):
        a = int(u_sorted[idx_edge])
        b = int(v_sorted[idx_edge])
        ww = float(w_sorted[idx_edge])

        # --- REAL FIX (robustness): always reject a directly-forbidden edge ---
        # This makes strict=True correct even if a user accidentally provides a
        # one-directional CSR adjacency (e.g. upper-triangular constraints).
        if _csr_row_contains_numba(cannot_link_indptr, cannot_link_indices, a, b) or _csr_row_contains_numba(
            cannot_link_indptr, cannot_link_indices, b, a
        ):
            continue

        ra = _dsu_find_numba(parent, a)
        rb = _dsu_find_numba(parent, b)
        if ra == rb:
            continue

        # Union-by-size, but only after constraint check.
        if size[ra] < size[rb]:
            root_small = ra
            root_large = rb
        else:
            root_small = rb
            root_large = ra

        if _component_has_conflict_with_root_numba(
            root_small,
            root_large,
            parent,
            head,
            next_node,
            cannot_link_indptr,
            cannot_link_indices,
        ):
            continue

        # Union: attach small -> large
        parent[root_small] = root_large
        size[root_large] = size[root_large] + size[root_small]

        # Concatenate member lists: large_tail.next = small_head
        next_node[tail[root_large]] = head[root_small]
        tail[root_large] = tail[root_small]

        # Emit MST edge
        mst_edges[n_added, 0] = float(a)
        mst_edges[n_added, 1] = float(b)
        mst_edges[n_added, 2] = float(ww)
        n_added += 1

        if n_added == n_points - 1:
            break

    return mst_edges[:n_added]


def _ensure_csr_distance_matrix(distances: Union[np.ndarray, CSR]) -> CSR:
    """
    Validate and return distances as CSR.

    IMPORTANT:
        - Do NOT call eliminate_zeros(): explicit off-diagonal zeros are meaningful edges.
        - Remove diagonal entries structurally so they do not affect core distances.
    """
    if sp.isspmatrix_csr(distances):
        distance_matrix_csr = distances.copy().astype(np.float64)
    elif isinstance(distances, np.ndarray):
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("Precomputed distance matrix must be square (N,N).")
        distance_matrix_csr = sp.csr_matrix(distances.astype(np.float64, copy=False))
    else:
        raise TypeError("distances must be a numpy array or scipy.sparse.csr_matrix.")

    n_points = int(distance_matrix_csr.shape[0])
    if n_points == 0 or distance_matrix_csr.shape[1] != n_points:
        raise ValueError("Precomputed distance matrix must be non-empty and square (N,N).")

    # ---- Remove diagonal structurally (preserves off-diagonal explicit zeros) ----
    coo = distance_matrix_csr.tocoo()
    mask_offdiag = coo.row != coo.col
    distance_matrix_csr = sp.coo_matrix(
        (coo.data[mask_offdiag], (coo.row[mask_offdiag], coo.col[mask_offdiag])),
        shape=coo.shape,
    ).tocsr()

    # Keep explicit zeros; just canonicalize representation.
    distance_matrix_csr.sum_duplicates()
    distance_matrix_csr.sort_indices()

    if distance_matrix_csr.data.size and np.any(distance_matrix_csr.data < 0):
        raise ValueError("Distances must be non-negative.")

    return distance_matrix_csr


def _symmetrize_min_keep_present(distance_matrix_csr: CSR) -> CSR:
    """
    Symmetrize a sparse distance matrix by taking:
        - min(A_ij, A_ji) when both exist
        - the existing value when only one direction exists

    IMPORTANT:
        - Preserve explicit off-diagonal zeros.
        - Do not use eliminate_zeros() or bool-mask multiply patterns (they often drop zeros).
    """
    A = distance_matrix_csr.tocsr(copy=True)
    A.sum_duplicates()
    A.sort_indices()

    B = A.T.tocsr(copy=True)
    B.sum_duplicates()
    B.sort_indices()

    n_points = int(A.shape[0])
    if n_points == 0:
        return A

    coo_a = A.tocoo()
    coo_b = B.tocoo()

    rows = np.concatenate([coo_a.row, coo_b.row]).astype(np.int64, copy=False)
    cols = np.concatenate([coo_a.col, coo_b.col]).astype(np.int64, copy=False)
    data = np.concatenate([coo_a.data, coo_b.data]).astype(np.float64, copy=False)

    # Drop diagonal structurally.
    mask_offdiag = rows != cols
    rows = rows[mask_offdiag]
    cols = cols[mask_offdiag]
    data = data[mask_offdiag]

    if data.size == 0:
        return sp.csr_matrix(A.shape, dtype=np.float64)

    # Reduce duplicates by min over identical (row, col).
    key = rows * np.int64(n_points) + cols
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    group_starts = np.empty(key.shape[0], dtype=bool)
    group_starts[0] = True
    group_starts[1:] = key[1:] != key[:-1]
    idx_start = np.flatnonzero(group_starts)

    data_min = np.minimum.reduceat(data, idx_start)
    rows_u = rows[idx_start]
    cols_u = cols[idx_start]

    sym = sp.coo_matrix((data_min, (rows_u, cols_u)), shape=A.shape).tocsr()
    sym.sum_duplicates()
    sym.sort_indices()
    return sym


def _core_distances_from_sparse_rows(
    distance_matrix_csr: CSR,
    *,
    min_samples: int,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute HDBSCAN core distances from a sparse precomputed distance graph.
    """
    n_points = int(distance_matrix_csr.shape[0])
    core_distances = np.empty(n_points, dtype=np.float64)

    if sample_weights is None:
        for idx_point in range(n_points):
            start = int(distance_matrix_csr.indptr[idx_point])
            end = int(distance_matrix_csr.indptr[idx_point + 1])
            degree = int(end - start)

            if degree == 0:
                core_distances[idx_point] = np.inf
                continue

            neighbor_distances_sorted = np.sort(distance_matrix_csr.data[start:end], kind="mergesort")
            if degree >= int(min_samples):
                core_distances[idx_point] = float(neighbor_distances_sorted[int(min_samples) - 1])
            else:
                core_distances[idx_point] = float(neighbor_distances_sorted[-1])
        return core_distances

    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    if sample_weights.shape != (n_points,):
        raise ValueError("sample_weights must have shape (N,)")

    for idx_point in range(n_points):
        start = int(distance_matrix_csr.indptr[idx_point])
        end = int(distance_matrix_csr.indptr[idx_point + 1])

        neighbor_indices = distance_matrix_csr.indices[start:end]
        neighbor_distances = distance_matrix_csr.data[start:end]
        if neighbor_indices.size == 0:
            core_distances[idx_point] = np.inf
            continue

        order = np.argsort(neighbor_distances, kind="mergesort")
        cumulative_weight = 0.0
        target_weight = float(min_samples)

        reached = False
        for idx_order in order.tolist():
            cumulative_weight += float(sample_weights[int(neighbor_indices[int(idx_order)])])
            if cumulative_weight >= target_weight:
                core_distances[idx_point] = float(neighbor_distances[int(idx_order)])
                reached = True
                break

        if not reached:
            core_distances[idx_point] = float(neighbor_distances[int(order[-1])])

    return core_distances


def _mutual_reachability_edges_upper_triangle(
    distance_matrix_csr: CSR,
    *,
    core_distances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an undirected edge list (u, v, w) from the upper triangle of a sparse
    distance matrix, with mutual reachability weights.
    """
    coo = sp.triu(distance_matrix_csr, k=1).tocoo()
    if coo.nnz == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    u = coo.row.astype(np.int64, copy=False)
    v = coo.col.astype(np.int64, copy=False)
    d = coo.data.astype(np.float64, copy=False)

    w = np.maximum.reduce([d, core_distances[u], core_distances[v]])
    return u, v, w


def _kruskal_mst_unconstrained(
    *,
    n_points: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Standard Kruskal MST on an undirected weighted graph.
    """
    order = np.argsort(w, kind="mergesort")
    u_sorted = u[order]
    v_sorted = v[order]
    w_sorted = w[order]

    ds = ds_rank_create(int(n_points))
    mst_edges = np.empty((int(n_points) - 1, 3), dtype=np.float64)

    n_added = 0
    for idx_edge in range(u_sorted.shape[0]):
        a = int(u_sorted[idx_edge])
        b = int(v_sorted[idx_edge])
        ww = float(w_sorted[idx_edge])

        root_a = int(ds_find(ds, a))
        root_b = int(ds_find(ds, b))
        if root_a == root_b:
            continue

        ds_union_by_rank(ds, root_a, root_b)
        mst_edges[n_added, 0] = float(a)
        mst_edges[n_added, 1] = float(b)
        mst_edges[n_added, 2] = float(ww)
        n_added += 1

        if n_added == int(n_points) - 1:
            break

    return mst_edges[:n_added]


def _kruskal_mst_constrained_hard(
    *,
    n_points: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    merge_constraint: MergeConstraint,
) -> np.ndarray:
    """
    Hard constrained Kruskal:
        Skip edges whose merge would place any cannot-link pair in one component.

    This returns a spanning forest if constraints prevent full connectivity.
    """
    if merge_constraint.cannot_link_indptr is None or merge_constraint.cannot_link_indices is None:
        raise ValueError(
            "strict=True requires merge_constraint built from a cannot-link matrix "
            "(must provide CSR payload arrays)."
        )

    order = np.argsort(w, kind="mergesort")
    u_sorted = np.asarray(u[order], dtype=np.int32)
    v_sorted = np.asarray(v[order], dtype=np.int32)
    w_sorted = np.asarray(w[order], dtype=np.float64)

    cannot_link_indptr = np.asarray(merge_constraint.cannot_link_indptr, dtype=np.int64)
    cannot_link_indices = np.asarray(merge_constraint.cannot_link_indices, dtype=np.int32)

    if cannot_link_indptr.ndim != 1 or cannot_link_indptr.size != int(n_points) + 1:
        raise ValueError("merge_constraint.cannot_link_indptr must have shape (N+1,) for strict=True.")

    mst_edges = _constrained_kruskal_mst_csr_strict_sorted_numba(
        u_sorted,
        v_sorted,
        w_sorted,
        cannot_link_indptr,
        cannot_link_indices,
        int(n_points),
    )

    return mst_edges


def _choose_large_finite_penalty(
    distances: Union[np.ndarray, CSR],
    *,
    user_penalty: float,
    scale_quantile: float = 99.9,
    factor: float = 1e6,
) -> float:
    """
    Map user_penalty to a large finite value if the user passes +inf.
    """
    if np.isfinite(user_penalty):
        return float(user_penalty)

    if sp.issparse(distances):
        vals = distances.data[np.isfinite(distances.data)]
    else:
        vals = np.asarray(distances, dtype=float)
        vals = vals[np.isfinite(vals)]

    base = 1.0 if vals.size == 0 else float(np.percentile(vals, scale_quantile))
    finite_penalty = float(base * factor + 1.0)
    finite_penalty = float(min(finite_penalty, np.finfo(np.float64).max / 10.0))
    return finite_penalty


def _connect_components_with_penalty_edges(
    *,
    n_points: int,
    mst_edges: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """
    If mst_edges is a forest, connect its components with penalty-weight edges
    so that we return an (N-1, 3) tree.
    """
    n_points = int(n_points)
    if mst_edges.shape[0] >= n_points - 1:
        return mst_edges

    ds = ds_rank_create(n_points)
    for row in mst_edges:
        a = int(row[0])
        b = int(row[1])
        root_a = int(ds_find(ds, a))
        root_b = int(ds_find(ds, b))
        if root_a != root_b:
            ds_union_by_rank(ds, root_a, root_b)

    root_to_rep = {}
    for idx_point in range(n_points):
        root = int(ds_find(ds, idx_point))
        if root not in root_to_rep:
            root_to_rep[root] = int(idx_point)

    reps = list(root_to_rep.values())
    if len(reps) <= 1:
        return mst_edges

    rep_root = int(reps[0])
    extra_edges = np.empty((len(reps) - 1, 3), dtype=np.float64)
    for idx_extra, rep in enumerate(reps[1:]):
        extra_edges[idx_extra, 0] = float(rep_root)
        extra_edges[idx_extra, 1] = float(int(rep))
        extra_edges[idx_extra, 2] = float(penalty)

    return np.vstack([mst_edges, extra_edges])


def _sanitize_mst_edge_weights(
    mst_edges: np.ndarray,
    *,
    finite_penalty: float,
) -> np.ndarray:
    """
    Replace any non-finite edge weights in the MST with a large finite penalty.
    """
    bad = ~np.isfinite(mst_edges[:, 2])
    if np.any(bad):
        mst_edges = mst_edges.copy()
        mst_edges[bad, 2] = float(finite_penalty)
    return mst_edges


def connected_components_from_distance_graph(distances: Union[np.ndarray, CSR]) -> np.ndarray:
    """
    Compute connected components from the support of the precomputed distance graph.
    """
    if isinstance(distances, np.ndarray):
        n_points = int(distances.shape[0])
        return np.zeros(n_points, dtype=np.int64)

    distance_matrix_csr = distances.tocsr()
    n_points = int(distance_matrix_csr.shape[0])

    adjacency = distance_matrix_csr.copy()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    n_components, labels = sp_csgraph.connected_components(csgraph=adjacency, directed=False, return_labels=True)
    _ = n_components
    return labels.astype(np.int64, copy=False)


def find_cannot_link_violations(
    labels: np.ndarray,
    *,
    merge_constraint: MergeConstraint,
    noise_label: int = -1,
) -> np.ndarray:
    """
    Enumerate cannot-link violations in the final cluster labels.
    """
    labels = np.asarray(labels)
    if merge_constraint.iter_cannot_link_pairs is None:
        raise ValueError(
            "Cannot enumerate violations: merge_constraint does not support iterating constraint pairs."
        )

    out_pairs = []
    for i, j in merge_constraint.iter_cannot_link_pairs():
        li = int(labels[int(i)])
        lj = int(labels[int(j)])
        if li == lj and li != int(noise_label):
            out_pairs.append((int(i), int(j)))

    if len(out_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)

    return np.asarray(out_pairs, dtype=np.int64)


def split_clusters_to_respect_cannot_link(
    labels: np.ndarray,
    *,
    merge_constraint: MergeConstraint,
    distances: Optional[Union[np.ndarray, CSR]] = None,
    noise_label: int = -1,
) -> np.ndarray:
    """
    Post-hoc cleanup: split any cluster that violates cannot-link constraints.
    """
    labels_in = np.asarray(labels).astype(np.int64, copy=True)

    if merge_constraint.pair_cannot_link is None:
        raise ValueError(
            "Post-hoc splitting requires merge_constraint.pair_cannot_link for efficient conflict checks."
        )

    # ---- Step 1: split by distance-graph connected components (optional) ----
    if distances is not None and sp.issparse(distances):
        graph_component_labels = connected_components_from_distance_graph(distances)
        next_label = int(labels_in.max()) + 1

        for label_id in np.unique(labels_in).tolist():
            if int(label_id) == int(noise_label):
                continue
            idx_points = np.flatnonzero(labels_in == int(label_id))
            if idx_points.size <= 1:
                continue

            sub_component_ids = graph_component_labels[idx_points]
            unique_sub_components = np.unique(sub_component_ids)
            if unique_sub_components.size <= 1:
                continue

            for idx_sub, comp_id in enumerate(unique_sub_components.tolist()):
                if idx_sub == 0:
                    continue
                mask = sub_component_ids == int(comp_id)
                labels_in[idx_points[mask]] = int(next_label)
                next_label += 1

    # ---- Step 2: greedy split any remaining violations inside each cluster ----
    next_label = int(labels_in.max()) + 1

    for label_id in sorted(np.unique(labels_in).tolist()):
        if int(label_id) == int(noise_label):
            continue

        idx_points = np.flatnonzero(labels_in == int(label_id))
        if idx_points.size <= 1:
            continue

        idx_set = set(int(x) for x in idx_points.tolist())
        _ = idx_set
        adjacency_conflict = {int(i): [] for i in idx_points.tolist()}

        idx_list = idx_points.tolist()
        for a_i in range(len(idx_list)):
            i = int(idx_list[a_i])
            for a_j in range(a_i + 1, len(idx_list)):
                j = int(idx_list[a_j])
                if merge_constraint.pair_cannot_link(i, j) or merge_constraint.pair_cannot_link(j, i):
                    adjacency_conflict[i].append(j)
                    adjacency_conflict[j].append(i)

        has_any_conflict = any(len(v) > 0 for v in adjacency_conflict.values())
        if not has_any_conflict:
            continue

        order_nodes = sorted(
            idx_list,
            key=lambda node: (-len(adjacency_conflict[int(node)]), int(node)),
        )

        color_of_node: dict[int, int] = {}
        for node in order_nodes:
            used_colors = set()
            for nbr in adjacency_conflict[int(node)]:
                if int(nbr) in color_of_node:
                    used_colors.add(int(color_of_node[int(nbr)]))

            color = 0
            while color in used_colors:
                color += 1
            color_of_node[int(node)] = int(color)

        colors_present = sorted(set(color_of_node.values()))
        if len(colors_present) <= 1:
            continue

        for color in colors_present:
            if int(color) == 0:
                continue
            nodes_color = [node for node, c in color_of_node.items() if int(c) == int(color)]
            labels_in[np.asarray(nodes_color, dtype=np.int64)] = int(next_label)
            next_label += 1

    return labels_in


# ---------------------------- public wrappers ----------------------------


def fast_hdbscan_precomputed(
    distances: Union[np.ndarray, CSR],
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]]:
    """
    Run fast_hdbscan on a precomputed distance matrix (dense or sparse CSR).
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)

    n_points = int(distance_matrix_csr.shape[0])
    if min_samples is None:
        min_samples = int(min_cluster_size)

    if int(min_samples) <= 0 or int(min_cluster_size) <= 0:
        raise ValueError("min_samples and min_cluster_size must be positive integers.")

    core_distances = _core_distances_from_sparse_rows(
        distance_matrix_csr,
        min_samples=int(min_samples),
        sample_weights=sample_weights,
    )
    u, v, w = _mutual_reachability_edges_upper_triangle(distance_matrix_csr, core_distances=core_distances)
    mst_edges = _kruskal_mst_unconstrained(n_points=n_points, u=u, v=v, w=w)

    if mst_edges.shape[0] < n_points - 1:
        mst_edges = _connect_components_with_penalty_edges(n_points=n_points, mst_edges=mst_edges, penalty=np.inf)

    finite_penalty = _choose_large_finite_penalty(distances, user_penalty=np.inf)
    mst_edges = _sanitize_mst_edge_weights(mst_edges, finite_penalty=finite_penalty)

    return (
        *clusters_from_spanning_tree(
            mst_edges,
            min_cluster_size=int(min_cluster_size),
            cluster_selection_method=str(cluster_selection_method),
            max_cluster_size=max_cluster_size,
            allow_single_cluster=bool(allow_single_cluster),
            cluster_selection_epsilon=float(cluster_selection_epsilon),
            cluster_selection_persistence=float(cluster_selection_persistence),
            sample_weights=sample_weights,
        ),
    )[: (None if return_trees else 2)]


def fast_hdbscan_precomputed_with_merge_constraint(
    distances: Union[np.ndarray, CSR],
    merge_constraint: MergeConstraint,
    *,
    strict: bool = True,
    penalty: float = np.inf,
    posthoc_cleanup: bool = False,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]]:
    """
    Run fast_hdbscan on a precomputed distance matrix with merge constraints.
    """
    if not np.isfinite(penalty) and not np.isinf(penalty):
        raise ValueError("penalty must be finite or +inf.")

    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)

    n_points = int(distance_matrix_csr.shape[0])
    if min_samples is None:
        min_samples = int(min_cluster_size)

    if int(min_samples) <= 0 or int(min_cluster_size) <= 0:
        raise ValueError("min_samples and min_cluster_size must be positive integers.")

    core_distances = _core_distances_from_sparse_rows(
        distance_matrix_csr,
        min_samples=int(min_samples),
        sample_weights=sample_weights,
    )
    u, v, w = _mutual_reachability_edges_upper_triangle(distance_matrix_csr, core_distances=core_distances)

    if bool(strict):
        mst_edges = _kruskal_mst_constrained_hard(
            n_points=n_points,
            u=u,
            v=v,
            w=w,
            merge_constraint=merge_constraint,
        )
    else:
        if merge_constraint.pair_cannot_link is None:
            raise ValueError("strict=False requires merge_constraint.pair_cannot_link to be available.")

        w_inflated = w.copy()
        for idx_edge in range(u.shape[0]):
            a = int(u[idx_edge])
            b = int(v[idx_edge])
            if merge_constraint.pair_cannot_link(a, b) or merge_constraint.pair_cannot_link(b, a):
                w_inflated[idx_edge] = float(max(float(w_inflated[idx_edge]), float(penalty)))

        mst_edges = _kruskal_mst_unconstrained(n_points=n_points, u=u, v=v, w=w_inflated)

    # Always connect components at +inf so merges occur only at the top of the hierarchy.
    if mst_edges.shape[0] < n_points - 1:
        mst_edges = _connect_components_with_penalty_edges(n_points=n_points, mst_edges=mst_edges, penalty=np.inf)

    finite_penalty = _choose_large_finite_penalty(distances, user_penalty=float(penalty))
    mst_edges = _sanitize_mst_edge_weights(mst_edges, finite_penalty=finite_penalty)

    results = clusters_from_spanning_tree(
        mst_edges,
        min_cluster_size=int(min_cluster_size),
        cluster_selection_method=str(cluster_selection_method),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=bool(allow_single_cluster),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_persistence=float(cluster_selection_persistence),
        sample_weights=sample_weights,
    )

    if bool(posthoc_cleanup):
        labels_clean = _maybe_split_labels_if_cannot_link_violated(
            results[0],
            merge_constraint=merge_constraint,
            distances=distance_matrix_csr,
            noise_label=-1,
        )
        results = (labels_clean, results[1], *results[2:])

    return results[: (None if return_trees else 2)]


def fast_hdbscan_precomputed_with_cannot_link(
    distances: Union[np.ndarray, CSR],
    cannot_link: Union[np.ndarray, CSR],
    *,
    strict: bool = True,
    penalty: float = np.inf,
    posthoc_cleanup: bool = False,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]]:
    """
    Backwards-compatible wrapper: cannot-link constraints via dense/sparse matrix.
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    n_points = int(distance_matrix_csr.shape[0])

    merge_constraint = MergeConstraint.from_cannot_link_matrix(
        cannot_link=cannot_link,
        n_points=n_points,
    )

    return fast_hdbscan_precomputed_with_merge_constraint(
        distances=distances,
        merge_constraint=merge_constraint,
        strict=bool(strict),
        penalty=float(penalty),
        posthoc_cleanup=bool(posthoc_cleanup),
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_method=str(cluster_selection_method),
        allow_single_cluster=bool(allow_single_cluster),
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_persistence=float(cluster_selection_persistence),
        sample_weights=sample_weights,
        return_trees=bool(return_trees),
    )








#############################
###### TESTING SUITE ########
#############################

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as ssd

from sklearn.metrics import adjusted_rand_score

from fast_hdbscan.hdbscan import fast_hdbscan as fast_hdbscan_feature_space
# from fast_hdbscan.precomputed import (
#     MergeConstraint,
#     find_cannot_link_violations,
#     split_clusters_to_respect_cannot_link,
#     fast_hdbscan_precomputed,
#     fast_hdbscan_precomputed_with_cannot_link,
#     fast_hdbscan_precomputed_with_merge_constraint,
# )


def _dense_pairwise_distances_euclidean(data: np.ndarray) -> np.ndarray:
    condensed = ssd.pdist(data, metric="euclidean")
    dense = ssd.squareform(condensed)
    return dense.astype(np.float64, copy=False)


def _dense_to_knn_csr(distances: np.ndarray, *, k: int) -> sp.csr_matrix:
    """
    Keep k nearest neighbors per row (excluding self) in a directed graph.
    """
    n = int(distances.shape[0])
    rows = []
    cols = []
    vals = []
    for i in range(n):
        d = distances[i].copy()
        d[i] = np.inf
        nn = np.argsort(d, kind="mergesort")[: int(k)]
        for j in nn.tolist():
            if np.isfinite(d[int(j)]):
                rows.append(int(i))
                cols.append(int(j))
                vals.append(float(d[int(j)]))
    return sp.csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n))


def test_precomputed_matches_feature_space_on_dense_distances():
    rng = np.random.default_rng(0)

    data_a = rng.normal(loc=-2.0, scale=0.3, size=(40, 2))
    data_b = rng.normal(loc=2.0, scale=0.3, size=(40, 2))
    data = np.vstack([data_a, data_b]).astype(np.float64, copy=False)

    distances = _dense_pairwise_distances_euclidean(data)

    labels_feat, probs_feat = fast_hdbscan_feature_space(
        data,
        min_cluster_size=8,
        min_samples=8,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        return_trees=False,
    )
    labels_prec, probs_prec = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=8,
        min_samples=8,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        return_trees=False,
    )

    # Use ARI to ignore label permutations.
    assert adjusted_rand_score(labels_feat, labels_prec) == 1.0
    assert probs_feat.shape == probs_prec.shape
    assert labels_prec.shape[0] == data.shape[0]


def test_empty_cannot_link_dense_is_noop_vs_unconstrained():
    rng = np.random.default_rng(1)

    data = rng.normal(size=(60, 3)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    cannot_link = np.zeros((data.shape[0], data.shape[0]), dtype=bool)

    labels_base, _ = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=6,
        min_samples=6,
        allow_single_cluster=True,
    )
    labels_con, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link=cannot_link,
        strict=True,
        posthoc_cleanup=False,
        min_cluster_size=6,
        min_samples=6,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_base, labels_con) == 1.0


def test_empty_cannot_link_sparse_is_noop_vs_unconstrained():
    rng = np.random.default_rng(11)

    data = rng.normal(size=(50, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    cannot_link_sparse = sp.csr_matrix((data.shape[0], data.shape[0]), dtype=bool)

    labels_base, _ = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=5,
        min_samples=5,
        allow_single_cluster=True,
    )
    labels_con, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link=cannot_link_sparse,
        strict=True,
        posthoc_cleanup=False,
        min_cluster_size=5,
        min_samples=5,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_base, labels_con) == 1.0


def test_strict_true_requires_matrix_payload_for_callable():
    rng = np.random.default_rng(12)
    data = rng.normal(size=(12, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    def pair_cannot_link(i: int, j: int) -> bool:
        _ = i
        _ = j
        return False

    merge_constraint_callable = MergeConstraint.from_pair_cannot_link(pair_cannot_link)

    did_raise = False
    try:
        _ = fast_hdbscan_precomputed_with_merge_constraint(
            distances,
            merge_constraint=merge_constraint_callable,
            strict=True,
            min_cluster_size=3,
            min_samples=3,
            allow_single_cluster=True,
        )
    except ValueError:
        did_raise = True

    assert did_raise, "strict=True should require a matrix-backed MergeConstraint (CSR payload arrays)."


def test_soft_mode_callable_matches_dense_matrix_behavior():
    rng = np.random.default_rng(2)

    data = rng.normal(size=(35, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    # Create a small random cannot-link set (dense matrix).
    n = int(data.shape[0])
    cannot_link_dense = np.zeros((n, n), dtype=bool)
    pairs = rng.choice(n * n, size=40, replace=False)
    for p in pairs.tolist():
        i = int(p // n)
        j = int(p % n)
        if i == j:
            continue
        cannot_link_dense[i, j] = True
        cannot_link_dense[j, i] = True

    merge_from_dense = MergeConstraint.from_cannot_link_matrix(cannot_link_dense, n_points=n)

    def pair_cannot_link(i: int, j: int) -> bool:
        return bool(cannot_link_dense[int(i), int(j)])

    merge_from_callable = MergeConstraint.from_pair_cannot_link(pair_cannot_link)

    # Compare strict=False (soft, direct-edge penalization) because strict=True
    # is intentionally matrix-only in the current implementation.
    labels_dense, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_from_dense,
        strict=False,
        penalty=np.inf,
        min_cluster_size=4,
        min_samples=4,
        allow_single_cluster=True,
    )
    labels_callable, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_from_callable,
        strict=False,
        penalty=np.inf,
        min_cluster_size=4,
        min_samples=4,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_dense, labels_callable) == 1.0


def test_posthoc_cleanup_removes_violation_small_graph():
    distances = np.array(
        [
            [0.0, 0.1, 10.0, 10.0],
            [0.1, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )

    cannot_link = np.zeros((4, 4), dtype=bool)
    cannot_link[0, 1] = True
    cannot_link[1, 0] = True

    merge_constraint = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=4)

    labels, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_constraint,
        strict=True,
        posthoc_cleanup=True,
        min_cluster_size=2,
        min_samples=1,
        allow_single_cluster=True,
        return_trees=False,
    )

    violations = find_cannot_link_violations(labels, merge_constraint=merge_constraint)
    assert violations.shape[0] == 0


def test_merge_constraint_sanitizes_symmetry_and_diagonal_dense():
    n = 5
    cannot_link = np.zeros((n, n), dtype=bool)
    cannot_link[0, 0] = True
    cannot_link[0, 1] = True
    cannot_link[1, 0] = False  # intentionally asymmetric

    mc = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)

    # Diagonal must be forced off.
    assert mc.pair_cannot_link is not None
    assert mc.pair_cannot_link(0, 0) is False

    # Symmetry must be enforced.
    assert mc.pair_cannot_link(0, 1) is True
    assert mc.pair_cannot_link(1, 0) is True

    # Iteration should contain (0,1) exactly once.
    assert mc.iter_cannot_link_pairs is not None
    pairs = set(tuple(sorted((int(i), int(j)))) for (i, j) in mc.iter_cannot_link_pairs())
    assert (0, 1) in pairs
    assert (0, 0) not in pairs


def test_merge_constraint_dense_vs_sparse_equivalent_pairs_and_checks():
    rng = np.random.default_rng(21)
    n = 25

    cannot_link = np.zeros((n, n), dtype=bool)
    # Add some random pairs; enforce symmetry manually here as well.
    for _ in range(60):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        cannot_link[i, j] = True
        cannot_link[j, i] = True

    mc_dense = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)
    mc_sparse = MergeConstraint.from_cannot_link_matrix(sp.csr_matrix(cannot_link), n_points=n)

    assert mc_dense.pair_cannot_link is not None
    assert mc_sparse.pair_cannot_link is not None
    assert mc_dense.iter_cannot_link_pairs is not None
    assert mc_sparse.iter_cannot_link_pairs is not None

    pairs_dense = set(tuple(sorted((int(i), int(j)))) for (i, j) in mc_dense.iter_cannot_link_pairs())
    pairs_sparse = set(tuple(sorted((int(i), int(j)))) for (i, j) in mc_sparse.iter_cannot_link_pairs())
    assert pairs_dense == pairs_sparse

    # Spot-check pair queries.
    for _ in range(100):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        assert bool(mc_dense.pair_cannot_link(i, j)) == bool(mc_sparse.pair_cannot_link(i, j))


def test_precomputed_handles_sparse_distance_graph_with_isolated_node():
    # Sparse distance graph with one isolated node (degree 0) -> core distance inf.
    n = 5
    rows = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    cols = np.array([1, 2, 3, 0, 3, 0, 1, 2], dtype=np.int64)
    vals = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    distances_csr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    labels, probs = fast_hdbscan_precomputed(
        distances_csr,
        min_cluster_size=2,
        min_samples=1,
        allow_single_cluster=True,
    )

    assert labels.shape == (n,)
    assert probs.shape == (n,)
    assert np.all(np.isfinite(probs))


def test_posthoc_split_fixes_cross_component_violation():
    # Two disconnected components in the distance graph (CSR).
    # Start with a "bad" clustering that merges everything into one cluster,
    # then ensure posthoc splitting separates components and removes violations.
    n = 6
    rows = []
    cols = []
    vals = []

    # component 1: {0,1,2} complete graph with distance 1
    comp1 = [0, 1, 2]
    for i in comp1:
        for j in comp1:
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    # component 2: {3,4,5} complete graph with distance 1
    comp2 = [3, 4, 5]
    for i in comp2:
        for j in comp2:
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    distances_csr = sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )

    # Cannot-link one point across components.
    cannot_link = np.zeros((n, n), dtype=bool)
    cannot_link[0, 3] = True
    cannot_link[3, 0] = True

    merge_constraint = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)

    labels_bad = np.zeros(n, dtype=np.int64)  # everything in one cluster
    violations_before = find_cannot_link_violations(labels_bad, merge_constraint=merge_constraint)
    assert violations_before.shape[0] >= 1

    labels_fixed = split_clusters_to_respect_cannot_link(
        labels_bad,
        merge_constraint=merge_constraint,
        distances=distances_csr,
    )
    violations_after = find_cannot_link_violations(labels_fixed, merge_constraint=merge_constraint)
    assert violations_after.shape[0] == 0

    # And ensure the two components are not forced into the same final label.
    assert labels_fixed[0] != labels_fixed[3]


def test_find_violations_requires_iterable_pairs():
    def pair_cannot_link(i: int, j: int) -> bool:
        return (int(i) == 0 and int(j) == 1) or (int(i) == 1 and int(j) == 0)

    mc = MergeConstraint.from_pair_cannot_link(pair_cannot_link)
    labels = np.zeros(3, dtype=np.int64)

    did_raise = False
    try:
        _ = find_cannot_link_violations(labels, merge_constraint=mc)
    except ValueError:
        did_raise = True

    assert did_raise


if __name__ == "__main__":
    test_precomputed_matches_feature_space_on_dense_distances()
    test_empty_cannot_link_dense_is_noop_vs_unconstrained()
    test_empty_cannot_link_sparse_is_noop_vs_unconstrained()
    test_strict_true_requires_matrix_payload_for_callable()
    test_soft_mode_callable_matches_dense_matrix_behavior()
    test_posthoc_cleanup_removes_violation_small_graph()
    test_merge_constraint_sanitizes_symmetry_and_diagonal_dense()
    test_merge_constraint_dense_vs_sparse_equivalent_pairs_and_checks()
    test_precomputed_handles_sparse_distance_graph_with_isolated_node()
    test_posthoc_split_fixes_cross_component_violation()
    test_find_violations_requires_iterable_pairs()
    print("All tests passed.")