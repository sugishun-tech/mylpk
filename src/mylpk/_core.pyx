# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Sparse-column revised simplex; dense LU base and product-form updates.

This module does not call another optimization solver. LAPACK is used only for
linear algebra. Every externally supplied buffer is validated before nogil code.
"""
from libc.math cimport fabs, isfinite
from scipy.linalg.cython_lapack cimport dgetrf, dgetrs
import numpy as np
import time

cdef class Workspace:
    cdef int m, n, capacity, neta, bland, degeneracy
    cdef bint identity
    cdef object _data, _indices, _indptr, _basis, _basic, _lu, _piv
    cdef object _eta, _positions, _xb, _pi, _rc, _direction, _row, _b, _c, _ray
    cdef double[::1] data, xb, pi, rc, direction, row, b, c, ray
    cdef int[::1] indices, indptr, basis, basic, piv, positions
    cdef double[::1, :] lu
    cdef double[:, ::1] eta
    cdef int iterations, refactorizations

    def __init__(self, data, indices, indptr, int m, basis, int refactor_interval=48):
        if m < 1 or refactor_interval < 1 or refactor_interval > 1024:
            raise ValueError("m must be positive; refactor_interval must be in [1, 1024].")
        dd = np.array(data, dtype=np.float64, order='C', copy=True)
        ii0 = np.asarray(indices)
        pp0 = np.asarray(indptr)
        bb0 = np.asarray(basis)
        if dd.ndim != 1 or ii0.ndim != 1 or pp0.ndim != 1 or bb0.shape != (m,):
            raise ValueError("Invalid CSC/basis dimensions.")
        if len(pp0) < 2 or not np.issubdtype(ii0.dtype, np.integer) or not np.issubdtype(pp0.dtype, np.integer) or not np.issubdtype(bb0.dtype, np.integer):
            raise ValueError("CSC and basis indices must be integer arrays.")
        n = len(pp0)-1
        if n >= 2147483647 or len(dd) >= 2147483647 or m >= 2147483647:
            raise ValueError("The native core uses 32-bit LAPACK/CSC indices.")
        if pp0[0] != 0 or pp0[len(pp0)-1] != len(dd) or len(ii0) != len(dd) or np.any(np.diff(pp0) < 0):
            raise ValueError("Invalid CSC pointers.")
        if not np.isfinite(dd).all() or np.any(ii0 < 0) or np.any(ii0 >= m):
            raise ValueError("Invalid CSC entries.")
        if np.any(bb0 < 0) or np.any(bb0 >= n) or len(np.unique(bb0)) != m:
            raise ValueError("Invalid basis indices.")
        self.m, self.n, self.capacity = m, n, refactor_interval
        self._data, self._indices, self._indptr = dd, np.array(ii0, dtype=np.int32), np.array(pp0, dtype=np.int32)
        self._basis = np.array(bb0, dtype=np.int32, copy=True)
        self._basic = np.zeros(n, dtype=np.int32)
        self._basic[self._basis] = 1
        self._lu = np.empty((m, m), dtype=np.float64, order='F')
        self._piv = np.empty(m, dtype=np.int32)
        self._eta = np.empty((refactor_interval, m), dtype=np.float64)
        self._positions = np.empty(refactor_interval, dtype=np.int32)
        self._xb, self._pi, self._rc = np.zeros(m), np.zeros(m), np.zeros(n)
        self._direction, self._row = np.zeros(m), np.zeros(m)
        self._b, self._c, self._ray = np.zeros(m), np.zeros(n), np.zeros(n)
        self.data, self.indices, self.indptr = self._data, self._indices, self._indptr
        self.basis, self.basic, self.lu, self.piv = self._basis, self._basic, self._lu, self._piv
        self.eta, self.positions = self._eta, self._positions
        self.xb, self.pi, self.rc = self._xb, self._pi, self._rc
        self.direction, self.row = self._direction, self._row
        self.b, self.c, self.ray = self._b, self._c, self._ray
        self.iterations, self.refactorizations, self.bland, self.degeneracy = 0, 0, 0, 0
        cdef int info
        with nogil:
            info = self._factor()
        if info:
            raise ValueError("The supplied basis is singular.")

    cdef int _factor(self) noexcept nogil:
        cdef int i, j, k, col, info=0
        self.identity = True
        for j in range(self.m):
            col = self.basis[j]
            if self.indptr[col+1]-self.indptr[col] != 1:
                self.identity = False
                break
            k = self.indptr[col]
            if self.indices[k] != j or self.data[k] != 1.0:
                self.identity = False
                break
        if not self.identity:
            for j in range(self.m):
                for i in range(self.m):
                    self.lu[i,j] = 0.0
                col = self.basis[j]
                for k in range(self.indptr[col], self.indptr[col+1]):
                    self.lu[self.indices[k], j] += self.data[k]
            dgetrf(&self.m, &self.m, &self.lu[0,0], &self.m, &self.piv[0], &info)
        self.neta = 0
        self.refactorizations += 1
        return info

    cdef void _ftran(self, double[::1] v) noexcept nogil:
        cdef int i, e, p, one=1, info=0
        cdef double t
        cdef char trans = 'N'
        if not self.identity:
            dgetrs(&trans, &self.m, &one, &self.lu[0,0], &self.m, &self.piv[0], &v[0], &self.m, &info)
        for e in range(self.neta):
            p = self.positions[e]
            t = v[p] / self.eta[e,p]
            for i in range(self.m):
                v[i] -= t*self.eta[e,i]
            v[p] = t

    cdef void _btran(self, double[::1] v) noexcept nogil:
        cdef int i, e, p, one=1, info=0
        cdef double t
        cdef char trans = 'T'
        for e in range(self.neta-1, -1, -1):
            p = self.positions[e]
            t = v[p]
            for i in range(self.m):
                if i != p:
                    t -= self.eta[e,i]*v[i]
            v[p] = t / self.eta[e,p]
        if not self.identity:
            dgetrs(&trans, &self.m, &one, &self.lu[0,0], &self.m, &self.piv[0], &v[0], &self.m, &info)

    cdef void _price(self) noexcept nogil:
        cdef int i, j, k
        cdef double t
        for i in range(self.m):
            self.pi[i] = self.c[self.basis[i]]
        self._btran(self.pi)
        for j in range(self.n):
            t = self.c[j]
            for k in range(self.indptr[j], self.indptr[j+1]):
                t -= self.data[k]*self.pi[self.indices[k]]
            self.rc[j] = 0.0 if self.basic[j] else t

    cdef void _column(self, int j) noexcept nogil:
        cdef int i, k
        for i in range(self.m):
            self.direction[i] = 0.0
        for k in range(self.indptr[j], self.indptr[j+1]):
            self.direction[self.indices[k]] += self.data[k]
        self._ftran(self.direction)

    cdef int _pivot(self, int entering, int leaving, double step) noexcept nogil:
        cdef int i, info
        for i in range(self.m):
            self.xb[i] -= step*self.direction[i]
            self.eta[self.neta,i] = self.direction[i]
        self.xb[leaving] = step
        self.positions[self.neta] = leaving
        self.neta += 1
        self.basic[self.basis[leaving]] = 0
        self.basic[entering] = 1
        self.basis[leaving] = entering
        self.iterations += 1
        if self.neta >= self.capacity:
            info = self._factor()
            if info:
                return 4
            for i in range(self.m):
                self.xb[i] = self.b[i]
            self._ftran(self.xb)
        return 0

    cdef int _iterate(self, int steps, int eligible, double ftol, double dtol, double ptol) noexcept nogil:
        cdef int it, i, j, k, enter, leave, code
        cdef double worst, best, ratio, alpha, step, relaxed, magnitude
        cdef bint primal_ok, dual_ok
        for it in range(steps):
            self._price()
            primal_ok = True
            dual_ok = True
            for i in range(self.m):
                if not isfinite(self.xb[i]) or not isfinite(self.pi[i]):
                    return 4
                if self.xb[i] < -ftol:
                    primal_ok = False
            enter = -1
            worst = -dtol
            for j in range(eligible):
                if not isfinite(self.rc[j]):
                    return 4
                if not self.basic[j] and self.rc[j] < -dtol:
                    dual_ok = False
                    if self.bland:
                        if enter < 0:
                            enter = j
                    elif self.rc[j] < worst:
                        worst, enter = self.rc[j], j
            if primal_ok:
                if dual_ok:
                    return 0
                self._column(enter)
                relaxed = 1.0e300
                for i in range(self.m):
                    if self.direction[i] > ptol:
                        ratio = (max(self.xb[i], 0.0)+ftol)/self.direction[i]
                        if ratio < relaxed:
                            relaxed = ratio
                leave, magnitude, step = -1, -1.0, 0.0
                for i in range(self.m):
                    if self.direction[i] > ptol:
                        ratio = max(self.xb[i], 0.0)/self.direction[i]
                        if ratio <= relaxed:
                            if leave < 0 or (self.bland and self.basis[i] < self.basis[leave]) or (not self.bland and self.direction[i] > magnitude):
                                leave, magnitude, step = i, self.direction[i], ratio
                if leave < 0:
                    for j in range(self.n):
                        self.ray[j] = 0.0
                    self.ray[enter] = 1.0
                    for i in range(self.m):
                        self.ray[self.basis[i]] = -self.direction[i]
                    return 3
            elif dual_ok:
                leave, worst = -1, -ftol
                for i in range(self.m):
                    if self.xb[i] < -ftol:
                        if leave < 0 or (self.bland and self.basis[i] < self.basis[leave]) or (not self.bland and self.xb[i] < worst):
                            leave, worst = i, self.xb[i]
                for i in range(self.m):
                    self.row[i] = 0.0
                self.row[leave] = 1.0
                self._btran(self.row)
                enter, best, magnitude = -1, 1.0e300, 0.0
                for j in range(eligible):
                    if self.basic[j]:
                        continue
                    alpha = 0.0
                    for k in range(self.indptr[j], self.indptr[j+1]):
                        alpha += self.data[k]*self.row[self.indices[k]]
                    if alpha < -ptol:
                        ratio = max(self.rc[j], 0.0)/(-alpha)
                        if ratio < best-1e-12 or (fabs(ratio-best) <= 1e-12 and ((self.bland and (enter < 0 or j < enter)) or (not self.bland and -alpha > magnitude))):
                            best, enter, magnitude = ratio, j, -alpha
                if enter < 0:
                    return 2
                self._column(enter)
                if fabs(self.direction[leave]) <= ptol:
                    return 4
                step = self.xb[leave]/self.direction[leave]
            else:
                return 5  # Neither primal nor dual feasible: restart phase I.
            if step <= ftol:
                self.degeneracy += 1
                if self.degeneracy >= 12:
                    self.bland = 1
            else:
                self.degeneracy = 0
            code = self._pivot(enter, leave, step)
            if code:
                return code
        return 6

    def run(self, cost, rhs, int eligible, int max_iter=100000, double feasibility_tol=1e-8,
            double dual_tol=1e-9, double pivot_tol=1e-12, double time_limit=float('inf'),
            pivot_rule="dantzig"):
        cc = np.asarray(cost, dtype=np.float64)
        bb = np.asarray(rhs, dtype=np.float64)
        if cc.shape != (self.n,) or bb.shape != (self.m,):
            raise ValueError("Incorrect cost or RHS length.")
        if not np.isfinite(cc).all() or not np.isfinite(bb).all():
            raise ValueError("Cost and RHS must be finite.")
        if eligible < 0 or eligible > self.n or max_iter < 1:
            raise ValueError("Invalid eligible columns or iteration limit.")
        if not np.isfinite([feasibility_tol, dual_tol, pivot_tol]).all() or min(feasibility_tol,dual_tol,pivot_tol) <= 0 or time_limit <= 0 or np.isnan(time_limit):
            raise ValueError("Tolerances/time limit must be positive.")
        if pivot_rule not in ("dantzig", "bland"):
            raise ValueError("pivot_rule must be 'dantzig' or 'bland'.")
        self._c[:] = cc
        self._b[:] = bb
        self._xb[:] = bb
        self._ray[:] = 0
        if pivot_rule == "bland":
            self.bland = 1
        cdef int code=6, old=self.iterations, steps
        with nogil:
            self._ftran(self.xb)
        start = time.perf_counter()
        while code == 6 and self.iterations-old < max_iter:
            if time.perf_counter()-start >= time_limit:
                code = 7
                break
            steps = min(128, max_iter-(self.iterations-old))
            with nogil:
                code = self._iterate(steps, eligible, feasibility_tol, dual_tol, pivot_tol)
        if code == 6:
            code = 1
        with nogil:
            self._price()
        x = np.zeros(self.n)
        x[self._basis] = self._xb
        return {"code": code, "x": x, "dual": self._pi.copy(), "reduced_costs": self._rc.copy(),
                "basis": self._basis.copy(), "iterations": self.iterations-old,
                "refactorizations": self.refactorizations, "ray": self._ray.copy(),
                "farkas": self._row.copy() if code == 2 else None}

    def purge_artificials(self, int eligible, double tol=1e-8, double pivot_tol=1e-12, int max_iter=100000):
        """Pivot zero artificial basics out; retain only redundant zero rows."""
        if eligible < 0 or eligible > self.n or not np.isfinite([tol, pivot_tol]).all() or tol <= 0 or pivot_tol <= 0 or max_iter < 0:
            raise ValueError("Invalid artificial-column boundary/tolerance/iteration limit.")
        cdef int i, j, k, enter, code=0, old=self.iterations
        cdef double a, best, step
        with nogil:
            for i in range(self.m):
                if self.basis[i] < eligible:
                    continue
                if fabs(self.xb[i]) > tol:
                    code = 4
                    break
                for k in range(self.m):
                    self.row[k] = 0
                self.row[i] = 1
                self._btran(self.row)
                best, enter = pivot_tol, -1
                for j in range(eligible):
                    if self.basic[j]:
                        continue
                    a = 0
                    for k in range(self.indptr[j], self.indptr[j+1]):
                        a += self.data[k]*self.row[self.indices[k]]
                    if fabs(a) > best:
                        best, enter = fabs(a), j
                if enter >= 0:
                    if self.iterations-old >= max_iter:
                        code = 1
                        break
                    self._column(enter)
                    step = self.xb[i]/self.direction[i]
                    code = self._pivot(enter, i, step)
                    if code:
                        break
        return code

    @property
    def basis_indices(self):
        return self._basis.copy()

    @property
    def iteration_count(self):
        return self.iterations
