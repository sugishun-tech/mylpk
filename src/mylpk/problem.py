"""Validated matrix problem, solve results, and strict solver options."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np
from scipy import sparse


def vector(value, length: int, name: str, *, finite=False, dtype=float):
    a = np.asarray(value, dtype=dtype)
    if a.ndim == 0:
        a = np.full(length, a.item(), dtype=dtype)
    if a.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {a.shape}.")
    a = np.array(a, dtype=dtype, order="C", copy=True)
    if np.issubdtype(a.dtype, np.floating):
        if np.isnan(a).any() or (finite and not np.isfinite(a).all()):
            raise ValueError(f"{name} contains non-finite values.")
    return a


@dataclass
class LinearProblem:
    """min/max c @ x + offset, row_lower <= A @ x <= row_upper.

    Integrality: 0 continuous, 1 integer, 2 semi-continuous, 3 semi-integer.
    Semi variables require 0 <= lower <= upper < infinity and also admit zero.
    Input arrays are copied. Treat a compiled problem as an immutable snapshot.
    """
    c: Any
    A: Any = None
    row_lower: Any = -np.inf
    row_upper: Any = np.inf
    lower: Any = 0.0
    upper: Any = np.inf
    integrality: Any = 0
    sense: str = "min"
    offset: float = 0.0
    var_names: list[str] | None = None
    row_names: list[str] | None = None
    name: str = "problem"

    def __post_init__(self):
        self.c = np.array(self.c, dtype=float, order="C", copy=True)
        if self.c.ndim != 1 or not np.isfinite(self.c).all():
            raise ValueError("c must be a finite one-dimensional vector.")
        n = self.c.size
        self.A = sparse.csc_matrix((0, n)) if self.A is None else sparse.csc_matrix(self.A, dtype=float, copy=True)
        if self.A.ndim != 2 or self.A.shape[1] != n or not np.isfinite(self.A.data).all():
            raise ValueError("A has invalid shape or non-finite entries.")
        self.A.check_format(full_check=True)
        self.A.sum_duplicates()
        self.A.eliminate_zeros()
        self.A.sort_indices()
        m = self.A.shape[0]
        self.row_lower = vector(self.row_lower, m, "row_lower")
        self.row_upper = vector(self.row_upper, m, "row_upper")
        self.lower = vector(self.lower, n, "lower")
        self.upper = vector(self.upper, n, "upper")
        # Validate before converting: 0.5 must not silently become continuous.
        integ = vector(self.integrality, n, "integrality", finite=True)
        if np.any(~np.isin(integ, [0, 1, 2, 3])):
            raise ValueError("integrality must contain only 0, 1, 2, 3.")
        self.integrality = integ.astype(np.int8)
        if np.isposinf(self.lower).any() or np.isneginf(self.upper).any() or np.isposinf(self.row_lower).any() or np.isneginf(self.row_upper).any():
            raise ValueError("A lower bound cannot be +inf; an upper bound cannot be -inf.")
        semi = self.integrality >= 2
        if np.any(self.lower[semi] < 0) or not np.isfinite(self.upper[semi]).all():
            raise ValueError("Semi variables require nonnegative lower and finite upper bounds.")
        if self.sense not in ("min", "max") or not np.isfinite(self.offset):
            raise ValueError("sense must be min/max; offset must be finite.")
        self.offset = float(self.offset)
        self.var_names = list(self.var_names) if self.var_names is not None else [f"x{i}" for i in range(n)]
        self.row_names = list(self.row_names) if self.row_names is not None else [f"r{i}" for i in range(m)]
        if len(self.var_names) != n or len(set(self.var_names)) != n or not all(isinstance(v,str) for v in self.var_names):
            raise ValueError("Variable names must be unique strings of the correct length.")
        if len(self.row_names) != m or len(set(self.row_names)) != m or not all(isinstance(v,str) for v in self.row_names):
            raise ValueError("Row names must be unique strings of the correct length.")

    @property
    def n(self): return self.c.size
    @property
    def m(self): return self.A.shape[0]
    @property
    def sign(self): return 1.0 if self.sense == "min" else -1.0
    @property
    def is_mip(self): return bool(np.any(self.integrality))
    def replace(self, **kwargs): return replace(self, **kwargs)
    def relax(self):
        lo = self.lower.copy()
        lo[self.integrality >= 2] = 0
        return self.replace(integrality=0, lower=lo)
    def solve(self, solver="mylpk", **kwargs):
        from .solvers import solve
        return solve(self, solver=solver, **kwargs)
    def write(self, path):
        from .io import write_problem
        return write_problem(self, path)
    def stats(self):
        a = np.abs(self.A.data)
        return {"variables": self.n, "rows": self.m, "nonzeros": self.A.nnz,
                "density": self.A.nnz / max(1, self.n*self.m),
                "integer_variables": int(np.count_nonzero(self.integrality)),
                "coefficient_min_abs": float(a.min()) if a.size else 0.,
                "coefficient_max_abs": float(a.max()) if a.size else 0.}


@dataclass(frozen=True)
class Basis:
    signature: str
    indices: np.ndarray
    scaling: bool


@dataclass
class SolveResult:
    status: str
    x: np.ndarray | None = None
    objective: float | None = None
    message: str = ""
    solver: str = "mylpk"
    iterations: int = 0
    nodes: int = 0
    bound: float | None = None
    gap: float | None = None
    elapsed: float = 0.0
    dual: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    row_lower_dual: np.ndarray | None = None
    row_upper_dual: np.ndarray | None = None
    lower_dual: np.ndarray | None = None
    upper_dual: np.ndarray | None = None
    activity: np.ndarray | None = None
    max_violation: float | None = None
    basis: Basis | None = None
    ray: np.ndarray | None = None
    certificate: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)
    model: Any = field(default=None, repr=False)

    @property
    def success(self): return self.status == "optimal"
    @property
    def fun(self): return self.objective
    @property
    def feasible(self): return self.x is not None and self.raw.get("primal_feasible", False)
    def require_optimal(self):
        if not self.success:
            raise RuntimeError(f"Optimization terminated with {self.status}: {self.message}")
        return self
    def __getitem__(self, item):
        if self.x is None:
            raise RuntimeError("This result has no primal solution.")
        if hasattr(item, "owner") and hasattr(item, "index"):
            if self.model is not item.owner:
                raise ValueError("Variable does not belong to this result's model.")
            return float(self.x[item.index])
        if hasattr(item, "_vars"):
            return {k: self[v] for k, v in item._vars.items()}
        return self.x[item]
    def to_dict(self):
        fields = ("status", "objective", "solver", "iterations", "nodes", "bound", "gap", "elapsed", "max_violation", "message")
        d = {k: getattr(self, k) for k in fields}
        for k in ("x", "dual", "reduced_costs", "row_lower_dual", "row_upper_dual", "lower_dual", "upper_dual", "activity", "ray"):
            a = getattr(self, k)
            d[k] = a.tolist() if a is not None else None
        d["feasible"] = self.feasible
        return d


@dataclass(frozen=True)
class Options:
    max_iter: int = 100000
    time_limit: float = np.inf
    feasibility_tol: float = 1e-7
    dual_tol: float = 1e-9
    integrality_tol: float = 1e-7
    pivot_tol: float = 1e-12
    refactor_interval: int = 48
    pivot_rule: str = "dantzig"
    scaling: bool = True
    presolve: bool = True
    max_basis_mb: float = 512.0
    node_limit: int = 100000
    mip_rel_gap: float = 1e-6
    mip_abs_gap: float = 1e-8
    cuts: bool = True
    heuristic: bool = True

    def __post_init__(self):
        for k in ("max_iter", "refactor_interval", "node_limit"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"{k} must be a positive integer.")
        if self.refactor_interval > 1024:
            raise ValueError("refactor_interval must be at most 1024.")
        for k in ("feasibility_tol", "dual_tol", "integrality_tol", "pivot_tol", "max_basis_mb"):
            v = getattr(self, k)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{k} must be finite and positive.")
        if self.time_limit <= 0 or np.isnan(self.time_limit):
            raise ValueError("time_limit must be positive.")
        for k in ("mip_rel_gap", "mip_abs_gap"):
            if not np.isfinite(getattr(self,k)) or getattr(self,k) < 0:
                raise ValueError(f"{k} must be finite and nonnegative.")
        if self.pivot_rule not in ("dantzig", "bland"):
            raise ValueError("pivot_rule must be dantzig/bland.")
        if self.integrality_tol >= 0.5:
            raise ValueError("integrality_tol must be less than 0.5.")
