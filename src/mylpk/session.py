"""Persistent native LP workspace for objective/RHS reoptimization."""
from __future__ import annotations
import copy
from dataclasses import replace
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .problem import LinearProblem,Options,SolveResult,vector
from .lp import solve_lp,postsolve
from ._core import Workspace
from .solvers import solve


class LPSession:
    """Prepare and solve once, then retain sparse columns, LU and eta factors.

    resolve() changes objective coefficients, objective sense/offset, and row
    bound VALUES. Variable bounds and matrix structure remain fixed. Simultaneous
    primal/dual infeasibility triggers an explicit native phase-I restart.
    One session is serialized by its lock; independent sessions may run in threads.
    """
    def __init__(self,problem,*,options=None,**kwargs):
        if not isinstance(problem,LinearProblem) or problem.is_mip: raise ValueError('LPSession requires a continuous LinearProblem.')
        self.options=Options(**kwargs) if options is None else (Options(**(options|kwargs)) if isinstance(options,dict) else replace(options,**kwargs))
        self.problem=problem.replace(); self._lock=threading.RLock(); self._workspace=None; self._standard=None
        self.last_result=solve_lp(self.problem,self.options)
        self._adopt(self.last_result)
    def _adopt(self,result):
        self.last_result=result
        s=result.raw.get('standard')
        if result.status=='optimal' and result.basis is not None and s is not None:
            self._standard=s
            self._workspace=Workspace(s.A.data,s.A.indices,s.A.indptr,s.A.shape[0],result.basis.indices,self.options.refactor_interval)
        else: self._workspace=None; self._standard=None
    def resolve(self,*,c=None,row_lower=None,row_upper=None,offset=None,sense=None):
        with self._lock:
            start=time.perf_counter(); p=copy.copy(self.problem)
            if c is not None: p.c=vector(c,p.n,'c',finite=True)
            if row_lower is not None: p.row_lower=vector(row_lower,p.m,'row_lower')
            if row_upper is not None: p.row_upper=vector(row_upper,p.m,'row_upper')
            if offset is not None:
                if not np.isfinite(offset): raise ValueError('offset must be finite.')
                p.offset=float(offset)
            if sense is not None:
                if sense not in ('min','max'): raise ValueError('sense must be min/max.')
                p.sense=sense
            if np.isposinf(p.row_lower).any() or np.isneginf(p.row_upper).any(): raise ValueError('Invalid infinite row bound orientation.')
            if not np.array_equal(np.isfinite(p.row_lower),np.isfinite(self.problem.row_lower)) or not np.array_equal(np.isfinite(p.row_upper),np.isfinite(self.problem.row_upper)) or not np.array_equal(p.row_lower==p.row_upper,self.problem.row_lower==self.problem.row_upper):
                raise ValueError('An LPSession cannot change row bound topology; construct a new session.')
            s=self._standard; cold=self._workspace is None or np.any(p.row_lower>p.row_upper)
            if not cold:
                rhs=s.b.copy(); shift_activity=np.asarray(p.A @ s.shift).ravel()
                represented=set()
                for k,(kind,i) in enumerate(s.row_meta):
                    if kind in ('equal','row_upper'):
                        rhs[k]=(p.row_upper[i]-shift_activity[i])*s.row_factor[k]; represented.add(i)
                    elif kind=='row_lower':
                        rhs[k]=(-p.row_lower[i]+shift_activity[i])*s.row_factor[k]; represented.add(i)
                # Constant rows eliminated during presolve can become inconsistent.
                for i in range(p.m):
                    if i not in represented and not (p.row_lower[i]<=shift_activity[i]<=p.row_upper[i]): cold=True; break
                cost=np.zeros(s.A.shape[1]); cost[:s.structural_columns]=np.asarray(s.transform.T @ (p.sign*p.c)).ravel()
                scale=float(np.max(abs(cost),initial=0)) if self.options.scaling else 1.
                if scale==0: scale=1.
                cost/=scale
                ss=replace(s,b=rhs,c=cost,cost_scale=scale)
                if not cold:
                    o=self.options
                    d=self._workspace.run(cost,rhs,s.real_columns,max_iter=o.max_iter,feasibility_tol=o.feasibility_tol,
                        dual_tol=o.dual_tol,pivot_tol=o.pivot_tol,time_limit=o.time_limit,pivot_rule=o.pivot_rule)
                    if d['code'] in (4,5) or np.max(abs(d['x'][s.real_columns:]),initial=0)>o.feasibility_tol:
                        cold=True
                    else:
                        result=postsolve(p,ss,d,o,d['iterations'],True)
                        if result.status=='numerical_error': cold=True
                        else:
                            result.raw['persistent_workspace']=True; result.raw['cold_restart']=False
                            self._standard=ss
            if cold:
                remain=self.options.time_limit-(time.perf_counter()-start)
                result=solve_lp(p,replace(self.options,time_limit=remain)) if remain>0 else SolveResult('time_limit')
                result.raw['persistent_workspace']=False; result.raw['cold_restart']=True
                self._adopt(result)
            self.problem=p; self.last_result=result
            result.elapsed=time.perf_counter()-start
            return result


def solve_many(problems,*,workers=None,solver='mylpk',**kwargs):
    """Independent solves in a thread pool; result order matches input order."""
    if workers is not None and (not isinstance(workers,int) or workers<1): raise ValueError('workers must be a positive integer or None.')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p:solve(p,solver=solver,**kwargs),problems))
