"""Two-phase native LP driver and postsolve."""
from __future__ import annotations
import time
import numpy as np
from .problem import LinearProblem, SolveResult, Basis, Options
from .standard import standardize, InfeasibleProblem
from .diagnostics import attach_primal, kkt_report, verify_ray
from ._core import Workspace

_CODES={0:"optimal",1:"iteration_limit",2:"infeasible",3:"unbounded",4:"numerical_error",5:"invalid_basis",7:"time_limit"}


def solve_lp(p: LinearProblem, opts: Options, *, basis: Basis | None=None) -> SolveResult:
    start=time.perf_counter()
    def finish(r):
        r.elapsed=time.perf_counter()-start
        return r
    try:
        s=standardize(p,opts)
    except InfeasibleProblem as e:
        return finish(SolveResult("infeasible",message=str(e)))
    m,n=s.A.shape
    if m==0:
        bad=np.flatnonzero(s.c[:s.real_columns] < -opts.dual_tol)
        if bad.size:
            z=np.zeros(n); z[bad[0]]=1
            r=SolveResult("unbounded",x=s.shift.copy(),ray=s.restore_ray(z),message="A feasible improving ray exists.")
            attach_primal(p,r,opts.feasibility_tol)
            return finish(r)
        r=SolveResult("optimal",x=s.shift.copy(),dual=np.zeros(p.m),row_lower_dual=np.zeros(p.m),row_upper_dual=np.zeros(p.m),reduced_costs=p.c.copy())
        r.lower_dual=np.zeros(p.n); r.upper_dual=np.zeros(p.n)
        for j in range(p.n):
            if np.isfinite(p.lower[j]) and np.isfinite(p.upper[j]) and p.lower[j]==p.upper[j]:
                r.lower_dual[j]=p.sign*max(p.sign*p.c[j],0)
                r.upper_dual[j]=p.sign*min(p.sign*p.c[j],0)
            elif np.isfinite(p.lower[j]): r.lower_dual[j]=p.c[j]
            elif np.isfinite(p.upper[j]): r.upper_dual[j]=p.c[j]
        attach_primal(p,r,opts.feasibility_tol)
        if r.feasible: r.bound=r.objective; r.gap=0.
        else: r.status='numerical_error'; r.message='Closed-form solution failed original-space validation or objective overflowed.'
        return finish(r)
    warm=basis is not None and basis.signature==s.signature and basis.scaling==opts.scaling
    try:
        ws=Workspace(s.A.data,s.A.indices,s.A.indptr,m,basis.indices if warm else s.initial_basis,opts.refactor_interval)
    except ValueError:
        if not warm: raise
        warm=False
        ws=Workspace(s.A.data,s.A.indices,s.A.indptr,m,s.initial_basis,opts.refactor_interval)
    total=0
    def run(c, eligible):
        nonlocal total
        remain=opts.time_limit-(time.perf_counter()-start)
        if remain<=0 or total>=opts.max_iter:
            return None
        d=ws.run(c,s.b,eligible,max_iter=opts.max_iter-total,feasibility_tol=opts.feasibility_tol,
                 dual_tol=opts.dual_tol,pivot_tol=opts.pivot_tol,time_limit=remain,pivot_rule=opts.pivot_rule)
        total+=d['iterations']
        return d
    d=None
    if warm:
        d=run(s.c,s.real_columns)
        if d is not None and (d['code']==5 or np.max(abs(d['x'][s.real_columns:]),initial=0)>opts.feasibility_tol):
            warm=False; d=None
            ws=Workspace(s.A.data,s.A.indices,s.A.indptr,m,s.initial_basis,opts.refactor_interval)
    if d is None and not warm:
        if n>s.real_columns:
            p1=run(s.phase_one_cost,n)
            if p1 is None:
                return finish(SolveResult("time_limit" if time.perf_counter()-start>=opts.time_limit else "iteration_limit",iterations=total))
            if p1['code']!=0:
                return finish(SolveResult(_CODES[p1['code']],iterations=total,message="Phase I did not complete.",raw={"phase":1}))
            if np.sum(p1['x'][s.real_columns:])>opts.feasibility_tol*max(1,m):
                y=-p1['dual']
                return finish(SolveResult("infeasible",iterations=total,message="Positive phase-I artificial objective.",
                    certificate={"kind":"farkas_standard_form", "y":y, "A_transpose_y_min":float(np.min(s.A[:,:s.real_columns].T @ y,initial=0)), "b_dot_y":float(s.b @ y), "signature":s.signature}))
            before=ws.iteration_count
            code=ws.purge_artificials(s.real_columns,opts.feasibility_tol*max(1,m),opts.pivot_tol,max(0,opts.max_iter-total))
            total+=ws.iteration_count-before
            if code:
                return finish(SolveResult(_CODES.get(code,'numerical_error'),iterations=total,message="Artificial-basis cleanup did not complete."))
        d=run(s.c,s.real_columns)
    if d is None:
        return finish(SolveResult("time_limit" if time.perf_counter()-start>=opts.time_limit else "iteration_limit",iterations=total))
    return finish(postsolve(p,s,d,opts,total,warm))


def postsolve(p,s,d,opts,total,warm):
    m=s.A.shape[0]
    status=_CODES[d['code']]
    x=s.restore(d['x']) if status in ("optimal","iteration_limit","time_limit","unbounded") else None
    r=SolveResult(status,x=x,iterations=total,basis=Basis(s.signature,d['basis'],opts.scaling),
                  raw={"phase":2,"warm_start_used":warm,"refactorizations":d['refactorizations'],"standard":s,"standard_solution":d['x'],"standard_dual":d['dual']})
    if status=="unbounded":
        r.ray=s.restore_ray(d['ray'])
    if status=="infeasible" and d['farkas'] is not None:
        r.certificate={"kind":"farkas_standard_form","y":d['farkas'],"signature":s.signature}
    if status=="optimal":
        dl=np.zeros(p.m); du=np.zeros(p.m); xu=np.zeros(p.n)
        for (kind,i),v in zip(s.row_meta,d['dual']*s.row_factor*s.cost_scale*p.sign):
            if kind=="equal": dl[i]+=v
            elif kind=="row_upper": du[i]+=v
            elif kind=="row_lower": dl[i]-=v
            elif kind=="upper": xu[i]+=v
        r.dual=dl+du; r.row_lower_dual=dl; r.row_upper_dual=du
        r.reduced_costs=p.c-np.asarray(p.A.T @ r.dual).ravel()
        xl=np.zeros(p.n)
        for j in range(p.n):
            if np.isfinite(p.lower[j]) and p.lower[j]==p.upper[j]:
                xl[j]=p.sign*max(p.sign*r.reduced_costs[j],0)
                xu[j]=p.sign*min(p.sign*r.reduced_costs[j],0)
            elif np.isfinite(p.lower[j]): xl[j]=r.reduced_costs[j]-xu[j]
            elif np.isfinite(p.upper[j]): xu[j]=r.reduced_costs[j]
        r.lower_dual=xl; r.upper_dual=xu
    attach_primal(p,r,opts.feasibility_tol)
    if status=='unbounded':
        check=verify_ray(p,r.x,r.ray,opts.feasibility_tol)
        r.raw['ray_check']=check
        if not check['valid']:
            r.status='numerical_error'; r.message='Unbounded ray failed original-space verification.'
    if status=="optimal":
        report=kkt_report(p,r)
        r.raw['kkt']=report
        dual_scale=1+float(np.max(abs(p.c),initial=0))
        obj_scale=1+abs(r.objective)
        if (not r.feasible or not np.isfinite([report['stationarity_max'],report['dual_sign_violation'],report['duality_gap']]).all() or report['stationarity_max']>20*opts.dual_tol*dual_scale or report['dual_sign_violation']>20*opts.dual_tol*dual_scale or report['duality_gap']>20*opts.feasibility_tol*obj_scale or np.max(abs(d['x'][s.real_columns:]),initial=0)>opts.feasibility_tol*max(1,m)):
            r.status="numerical_error"
            r.message="The candidate failed an original-space primal/dual postsolve check."
        else:
            r.bound=r.objective; r.gap=0.
    return r
