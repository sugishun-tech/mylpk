"""Reversible conversion to equality form, with explicit row provenance."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import sparse
from .problem import LinearProblem, Options

class InfeasibleProblem(Exception): pass

@dataclass
class StandardForm:
    A: sparse.csc_matrix
    b: np.ndarray
    c: np.ndarray
    phase_one_cost: np.ndarray
    initial_basis: np.ndarray
    real_columns: int
    transform: sparse.csc_matrix
    shift: np.ndarray
    row_meta: list[tuple[str, int]]
    row_factor: np.ndarray
    cost_scale: float
    signature: str
    scaling: bool

    @property
    def structural_columns(self): return self.transform.shape[1]
    def restore(self, z): return self.shift + np.asarray(self.transform @ z[:self.structural_columns]).ravel()
    def restore_ray(self, z): return np.asarray(self.transform @ z[:self.structural_columns]).ravel()


def standardize(p: LinearProblem, opts: Options) -> StandardForm:
    if p.is_mip:
        raise ValueError("Standard-form conversion expects a continuous relaxation.")
    if np.any(p.lower > p.upper) or np.any(p.row_lower > p.row_upper):
        raise InfeasibleProblem("Contradictory lower/upper bounds.")
    shift = np.zeros(p.n)
    ti, tj, tv, upper_cols, upper_vals, upper_orig = [], [], [], [], [], []
    nz = 0
    for j in range(p.n):
        lo, hi = p.lower[j], p.upper[j]
        if np.isfinite(lo):
            shift[j] = lo
            if lo == hi:
                continue
            ti.append(j); tj.append(nz); tv.append(1.)
            if np.isfinite(hi):
                upper_cols.append(nz); upper_vals.append(hi-lo); upper_orig.append(j)
            nz += 1
        elif np.isfinite(hi):
            shift[j] = hi
            ti.append(j); tj.append(nz); tv.append(-1.)
            nz += 1
        else:
            ti.extend([j,j]); tj.extend([nz,nz+1]); tv.extend([1.,-1.])
            nz += 2
    T = sparse.csc_matrix((tv,(ti,tj)), shape=(p.n,nz))
    At = (p.A @ T).tocsc()
    a0 = np.asarray(p.A @ shift).ravel()
    eq = np.flatnonzero(np.isfinite(p.row_lower) & (p.row_lower == p.row_upper))
    upper = np.flatnonzero(np.isfinite(p.row_upper) & (p.row_lower != p.row_upper))
    lower = np.flatnonzero(np.isfinite(p.row_lower) & (p.row_lower != p.row_upper))
    mats = [At[eq], At[upper], -At[lower]]
    rhs = [p.row_upper[eq]-a0[eq], p.row_upper[upper]-a0[upper], -p.row_lower[lower]+a0[lower]]
    meta = [("equal",int(i)) for i in eq] + [("row_upper",int(i)) for i in upper] + [("row_lower",int(i)) for i in lower]
    inequalities = [False]*len(eq)+[True]*(len(upper)+len(lower))
    if upper_cols:
        mats.append(sparse.csc_matrix((np.ones(len(upper_cols)), (np.arange(len(upper_cols)),upper_cols)),shape=(len(upper_cols),nz)))
        rhs.append(np.array(upper_vals))
        meta += [("upper",j) for j in upper_orig]
        inequalities += [True]*len(upper_cols)
    M = sparse.vstack(mats, format='csr') if mats else sparse.csr_matrix((0,nz))
    b = np.concatenate(rhs) if rhs else np.empty(0)
    iq = np.array(inequalities, dtype=bool)
    if not np.isfinite(b).all():
        raise ValueError("Overflow during bound shifting; rescale input data.")
    nonzero_rows = np.diff(M.indptr) != 0
    for i in np.flatnonzero(~nonzero_rows):
        if (iq[i] and b[i] < 0) or (not iq[i] and b[i] != 0):
            raise InfeasibleProblem(f"Constant row {meta[i]} is inconsistent.")
    keep = nonzero_rows if opts.presolve else np.ones(len(b),dtype=bool)
    M = M[keep].tocsr(); b = b[keep]; iq = iq[keep]
    meta = [v for v,k in zip(meta,keep) if k]
    m = len(b)
    need_mb = 8*(m*m + opts.refactor_interval*m + 12*m)/1024**2
    if need_mb > opts.max_basis_mb:
        raise MemoryError(f"Native dense LU workspace needs about {need_mb:.1f} MiB, exceeding max_basis_mb={opts.max_basis_mb}. Use solver='highs' for large sparse bases or raise the explicit limit.")
    norm = np.ones(m)
    if opts.scaling and m and nz:
        norm = np.asarray(abs(M).max(axis=1).toarray()).ravel()
        norm[norm == 0] = 1
    flip = np.where(b < 0, -1., 1.)
    factor = flip/norm
    M = M.multiply(factor[:,None]).tocsc()
    b = b*factor
    rows_iq = np.flatnonzero(iq)
    S = sparse.csc_matrix((flip[rows_iq],(rows_iq,np.arange(len(rows_iq)))),shape=(m,len(rows_iq)))
    art_rows = np.flatnonzero(~iq | (flip < 0))
    real_n = nz+len(rows_iq)
    R = sparse.csc_matrix((np.ones(len(art_rows)),(art_rows,np.arange(len(art_rows)))),shape=(m,len(art_rows)))
    A = sparse.hstack([M,S,R],format='csc')
    A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    basis = np.empty(m,dtype=np.int32)
    basis[rows_iq] = nz+np.arange(len(rows_iq))
    basis[art_rows] = real_n+np.arange(len(art_rows))
    c = np.zeros(A.shape[1])
    c[:nz] = np.asarray(T.T @ (p.sign*p.c)).ravel()
    scale = float(np.max(np.abs(c),initial=0)) if opts.scaling else 1.
    if scale == 0: scale=1.
    c /= scale
    pc = np.zeros_like(c); pc[real_n:]=1
    h = hashlib.sha256()
    h.update(np.asarray(A.shape,dtype=np.int64).tobytes())
    h.update(A.indptr.astype(np.int64).tobytes()); h.update(A.indices.astype(np.int64).tobytes()); h.update(A.data.tobytes())
    h.update(np.asarray([real_n],dtype=np.int64).tobytes())
    return StandardForm(A,b,c,pc,basis,real_n,T,shift,meta,factor,scale,h.hexdigest(),opts.scaling)
