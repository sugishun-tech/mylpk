"""Explicit backend selection. There is no silent fallback between solvers."""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import time
import threading
import numpy as np
from scipy import sparse
from scipy import optimize
from .problem import LinearProblem, Options, SolveResult
from .lp import solve_lp
from .mip import solve_mip, expand_semivariables
from .diagnostics import attach_primal

_GLPK_LOCK=threading.RLock()
_REGISTRY={}

def register_solver(name, function):
    """Register function(problem, options, backend_options) -> SolveResult."""
    if not isinstance(name,str) or not name or name in ('mylpk','native','highs','highs-ds','highs-ipm','glpk'):
        raise ValueError('Invalid/reserved backend name.')
    if not callable(function): raise TypeError('Backend must be callable.')
    _REGISTRY[name]=function


def available_solvers():
    return {'mylpk':True,'highs':True,'highs-ds':True,'highs-ipm':True,
            'glpk':importlib.util.find_spec('swiglpk') is not None,
            **{name:True for name in _REGISTRY}}


def solve(p, solver='mylpk', *, options=None, basis=None, x0=None, callback=None, backend_options=None, **kwargs):
    if not isinstance(p,LinearProblem): raise TypeError('Expected LinearProblem.')
    if options is None: opts=Options(**kwargs)
    elif isinstance(options,dict): opts=Options(**(options|kwargs))
    elif isinstance(options,Options): opts=replace(options,**kwargs)
    else: raise TypeError('options must be Options or a dictionary.')
    if not isinstance(backend_options,(dict,type(None))): raise TypeError('backend_options must be a dictionary.')
    if solver in ('mylpk','native'):
        if backend_options: raise ValueError('Use native Options, not backend_options, for mylpk.')
        if p.is_mip:
            if basis is not None: raise ValueError('Use x0 for MILP; LP bases belong to individual relaxation nodes.')
            return solve_mip(p,opts,x0=x0,callback=callback)
        if x0 is not None or callback is not None:
            raise ValueError('Native LP accepts basis warm starts; x0/callback are MILP options.')
        return solve_lp(p,opts,basis=basis)
    if basis is not None or x0 is not None or callback is not None:
        raise ValueError('This adapter does not expose basis/x0/callback. They are never silently ignored.')
    if np.any(p.lower>p.upper) or np.any(p.row_lower>p.row_upper):
        return SolveResult('infeasible',solver=solver,message='Contradictory bounds.')
    if p.n==0:
        r=solve_lp(p.relax(),opts); r.solver=solver; return r
    if solver in ('highs','highs-ds','highs-ipm'): return _highs(p,opts,solver,backend_options or {})
    if solver=='glpk': return _glpk(p,opts,backend_options or {})
    if solver.startswith('pyomo:'): return _pyomo(p,opts,solver[6:],backend_options or {})
    if solver in _REGISTRY: return _REGISTRY[solver](p,opts,backend_options or {})
    raise ValueError(f'Unknown solver {solver!r}. Available: {available_solvers()}')


def _highs(p,opts,method,backend):
    start=time.perf_counter()
    common={'presolve':opts.presolve}
    if np.isfinite(opts.time_limit): common['time_limit']=opts.time_limit
    if p.is_mip:
        if method!='highs': raise ValueError('Use highs, not an LP-only method, for MILP.')
        allowed={'disp','node_limit','presolve','time_limit','mip_rel_gap'}
        if set(backend)-allowed: raise ValueError(f'Unsupported SciPy MILP options: {set(backend)-allowed}')
        common.update(node_limit=opts.node_limit,mip_rel_gap=opts.mip_rel_gap)
        common.update(backend)
        q=optimize.milp(p.sign*p.c,integrality=p.integrality,
            bounds=optimize.Bounds(p.lower,p.upper),constraints=optimize.LinearConstraint(p.A,p.row_lower,p.row_upper),options=common)
        r=SolveResult({0:'optimal',1:'limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(q.status,'solver_error'),x=q.x,
             solver='highs',message=str(q.message),nodes=int(getattr(q,'mip_node_count',0) or 0),
             bound=None if getattr(q,'mip_dual_bound',None) is None else p.sign*q.mip_dual_bound+p.offset,
             gap=getattr(q,'mip_gap',None))
        # Native and HiGHS use different gap denominators; publish adapter convention.
        r.raw['gap_convention']='HiGHS/SciPy reported gap (excludes objective offset)'
    else:
        allowed={'disp','presolve','time_limit','maxiter','dual_feasibility_tolerance','primal_feasibility_tolerance','ipm_optimality_tolerance','simplex_dual_edge_weight_strategy'}
        if set(backend)-allowed: raise ValueError(f'Unsupported SciPy LP options: {set(backend)-allowed}')
        eq=np.flatnonzero(np.isfinite(p.row_lower)&(p.row_lower==p.row_upper))
        up=np.flatnonzero(np.isfinite(p.row_upper)&(p.row_lower!=p.row_upper))
        lo=np.flatnonzero(np.isfinite(p.row_lower)&(p.row_lower!=p.row_upper))
        A=sparse.vstack([p.A[up],-p.A[lo]],format='csc')
        b=np.r_[p.row_upper[up],-p.row_lower[lo]]
        common.update(maxiter=opts.max_iter,primal_feasibility_tolerance=max(1e-10,opts.feasibility_tol),dual_feasibility_tolerance=max(1e-10,opts.dual_tol))
        common.update(backend)
        q=optimize.linprog(p.sign*p.c,A_ub=A if len(b) else None,b_ub=b if len(b) else None,
            A_eq=p.A[eq] if len(eq) else None,b_eq=p.row_lower[eq] if len(eq) else None,
            bounds=np.column_stack([p.lower,p.upper]),method=method,options=common)
        r=SolveResult({0:'optimal',1:'limit',2:'infeasible',3:'unbounded',4:'numerical_error'}.get(q.status,'solver_error'),
            x=q.x,solver=method,message=str(q.message),iterations=int(q.nit or 0))
        if q.success:
            dl=np.zeros(p.m); du=np.zeros(p.m)
            dl[eq]=p.sign*q.eqlin.marginals
            du[up]=p.sign*q.ineqlin.marginals[:len(up)]
            dl[lo]=-p.sign*q.ineqlin.marginals[len(up):]
            r.row_lower_dual=dl; r.row_upper_dual=du; r.dual=dl+du
            r.lower_dual=p.sign*q.lower.marginals; r.upper_dual=p.sign*q.upper.marginals
            r.reduced_costs=r.lower_dual+r.upper_dual
    attach_primal(p,r,opts.feasibility_tol)
    if r.success and r.x is not None:
        if not r.feasible: r.status='numerical_error'; r.message+=' Original-space primal check failed.'
        elif not p.is_mip: r.bound=r.objective; r.gap=0.
    r.elapsed=time.perf_counter()-start
    return r


def _glpk(original,opts,backend):
    try: import swiglpk as g
    except ImportError as e: raise ImportError("GLPK adapter requires: python -m pip install swiglpk (or install '.[glpk]' from the project directory).") from e
    allowed={'msg_lev','presolve','meth','br_tech','bt_tech','gmi_cuts','mir_cuts','cov_cuts','clq_cuts'}
    if set(backend)-allowed: raise ValueError(f'Unsupported GLPK options: {set(backend)-allowed}')
    allowed_values={
        'msg_lev':{g.GLP_MSG_OFF,g.GLP_MSG_ERR,g.GLP_MSG_ON,g.GLP_MSG_ALL,g.GLP_MSG_DBG},
        'meth':{g.GLP_PRIMAL,g.GLP_DUALP,g.GLP_DUAL},
        'br_tech':{g.GLP_BR_FFV,g.GLP_BR_LFV,g.GLP_BR_MFV,g.GLP_BR_DTH,g.GLP_BR_PCH},
        'bt_tech':{g.GLP_BT_DFS,g.GLP_BT_BFS,g.GLP_BT_BLB,g.GLP_BT_BPH},
    }
    for key,value in backend.items():
        valid=allowed_values.get(key,{g.GLP_ON,g.GLP_OFF})
        if not isinstance(value,(int,np.integer)) or value not in valid:
            raise ValueError(f'Invalid GLPK enum for {key}; allowed values: {sorted(valid)}.')
    defaults=Options()
    if opts.node_limit!=defaults.node_limit or opts.mip_abs_gap!=defaults.mip_abs_gap:
        raise ValueError('This GLPK adapter does not expose node_limit or mip_abs_gap; use native/HiGHS options supported by that backend.')
    p,_=expand_semivariables(original)
    # GLPK requires integer variable bounds to be integral.
    if p.is_mip:
        lo=p.lower.copy(); hi=p.upper.copy(); ids=p.integrality==1
        lo[ids]=np.ceil(lo[ids]); hi[ids]=np.floor(hi[ids]); p=p.replace(lower=lo,upper=hi)
        if np.any(lo>hi): return SolveResult('infeasible',solver='glpk')
    start=time.perf_counter()
    def btype(lo,hi):
        if np.isfinite(lo) and np.isfinite(hi): return g.GLP_FX if lo==hi else g.GLP_DB
        if np.isfinite(lo): return g.GLP_LO
        if np.isfinite(hi): return g.GLP_UP
        return g.GLP_FR
    with _GLPK_LOCK:
        prob=g.glp_create_prob()
        try:
            g.glp_set_obj_dir(prob,g.GLP_MIN if p.sense=='min' else g.GLP_MAX)
            if p.m: g.glp_add_rows(prob,p.m)
            if p.n: g.glp_add_cols(prob,p.n)
            for i in range(p.m):
                g.glp_set_row_bnds(prob,i+1,btype(p.row_lower[i],p.row_upper[i]),float(p.row_lower[i]) if np.isfinite(p.row_lower[i]) else 0.,float(p.row_upper[i]) if np.isfinite(p.row_upper[i]) else 0.)
            for j in range(p.n):
                g.glp_set_col_bnds(prob,j+1,btype(p.lower[j],p.upper[j]),float(p.lower[j]) if np.isfinite(p.lower[j]) else 0.,float(p.upper[j]) if np.isfinite(p.upper[j]) else 0.)
                g.glp_set_obj_coef(prob,j+1,float(p.c[j]))
                if p.integrality[j]: g.glp_set_col_kind(prob,j+1,g.GLP_IV)
            g.glp_set_obj_coef(prob,0,p.offset)
            coo=p.A.tocoo(); size=p.A.nnz
            ia=g.intArray(size+1); ja=g.intArray(size+1); ar=g.doubleArray(size+1)
            for k,(i,j,v) in enumerate(zip(coo.row,coo.col,coo.data),1): ia[k]=int(i)+1; ja[k]=int(j)+1; ar[k]=float(v)
            g.glp_load_matrix(prob,size,ia,ja,ar)
            sm=g.glp_smcp(); g.glp_init_smcp(sm)
            sm.msg_lev=g.GLP_MSG_OFF; sm.presolve=g.GLP_ON if opts.presolve else g.GLP_OFF
            sm.it_lim=min(opts.max_iter,2147483647)
            sm.tol_bnd=min(1e-2,max(1e-9,opts.feasibility_tol)); sm.tol_dj=min(1e-2,max(1e-9,opts.dual_tol))
            if np.isfinite(opts.time_limit): sm.tm_lim=min(2147483647,max(1,int(opts.time_limit*1000)))
            for k,v in backend.items():
                if k in ('msg_lev','presolve','meth'): setattr(sm,k,v)
            code=g.glp_simplex(prob,sm)
            status=g.glp_get_status(prob)
            if p.is_mip and status==g.GLP_OPT:
                ip=g.glp_iocp(); g.glp_init_iocp(ip)
                ip.msg_lev=g.GLP_MSG_OFF; ip.presolve=g.GLP_ON if opts.presolve else g.GLP_OFF
                ip.tol_int=opts.integrality_tol; ip.mip_gap=opts.mip_rel_gap
                if np.isfinite(opts.time_limit): ip.tm_lim=min(2147483647,max(1,int((opts.time_limit-(time.perf_counter()-start))*1000)))
                for k,v in backend.items():
                    if k!='meth': setattr(ip,k,v)
                code=g.glp_intopt(prob,ip); status=g.glp_mip_status(prob)
                has=status in (g.GLP_OPT,g.GLP_FEAS)
                x=np.array([g.glp_mip_col_val(prob,j+1) for j in range(p.n)]) if has else None
            else:
                has=not p.is_mip and status in (g.GLP_OPT,g.GLP_FEAS)
                x=np.array([g.glp_get_col_prim(prob,j+1) for j in range(p.n)]) if has else None
            if code==getattr(g,'GLP_ENOPFS',-999): label='infeasible'
            elif status==g.GLP_NOFEAS: label='infeasible'
            elif status==g.GLP_UNBND: label='relaxation_unbounded' if p.is_mip else 'unbounded'
            elif code==getattr(g,'GLP_ENODFS',-999): label='infeasible_or_unbounded'
            elif code==0 and status==g.GLP_OPT: label='optimal'
            elif code==getattr(g,'GLP_ETMLIM',-999): label='time_limit'
            elif code==getattr(g,'GLP_EITLIM',-999): label='iteration_limit'
            elif code==getattr(g,'GLP_EMIPGAP',-999): label='gap_limit'
            else: label='solver_error'
            r=SolveResult(label,x=None if x is None else x[:original.n],solver='glpk',iterations=int(g.glp_get_it_cnt(prob)),
                message=f'GLPK return code={code}, status={status}.',raw={'glpk_code':code,'glpk_status':status,
                'unsupported_common_limits':['node_limit','mip_abs_gap'],'bound_available':label=='optimal'})
            if label=='optimal' and not p.is_mip:
                dual=np.array([g.glp_get_row_dual(prob,i+1) for i in range(p.m)])
                rc=np.array([g.glp_get_col_dual(prob,j+1) for j in range(p.n)])
                dl=np.zeros(p.m); du=np.zeros(p.m)
                eq=np.isfinite(p.row_lower)&(p.row_lower==p.row_upper)
                dl[eq]=dual[eq]
                dl[~eq]=p.sign*np.maximum(p.sign*dual[~eq],0)
                du[~eq]=p.sign*np.minimum(p.sign*dual[~eq],0)
                r.dual=dual; r.row_lower_dual=dl; r.row_upper_dual=du; r.reduced_costs=rc
                r.lower_dual=p.sign*np.maximum(p.sign*rc,0); r.upper_dual=p.sign*np.minimum(p.sign*rc,0)
            attach_primal(original,r,opts.feasibility_tol)
            if label=='optimal' and r.x is not None:
                if not r.feasible: r.status='numerical_error'
                else: r.bound=r.objective; r.gap=0.
            r.elapsed=time.perf_counter()-start
            return r
        finally:
            g.glp_delete_prob(prob)


def _pyomo(original,opts,name,backend):
    try: import pyomo.environ as po
    except ImportError as e: raise ImportError("Install pyomo, or install '.[interop]' from the project directory, for pyomo:<solver> adapters.") from e
    if opts!=Options():
        raise ValueError('pyomo:<solver> only accepts backend_options; native Options are not silently forwarded.')
    p,_=expand_semivariables(original)
    model=po.ConcreteModel(); model.I=po.RangeSet(0,p.n-1)
    def bounds(m,j): return (None if np.isneginf(p.lower[j]) else float(p.lower[j]),None if np.isposinf(p.upper[j]) else float(p.upper[j]))
    model.x=po.Var(model.I,bounds=bounds,domain=lambda m,j:po.Integers if p.integrality[j] else po.Reals)
    model.obj=po.Objective(expr=sum(float(p.c[j])*model.x[j] for j in range(p.n))+p.offset,sense=po.minimize if p.sense=='min' else po.maximize)
    model.rows=po.ConstraintList(); A=p.A.tocsr()
    for i in range(p.m):
        e=sum(float(A.data[k])*model.x[int(A.indices[k])] for k in range(A.indptr[i],A.indptr[i+1]))
        if A.indptr[i]==A.indptr[i+1]:
            if p.row_lower[i]<=0<=p.row_upper[i]: continue
            return SolveResult('infeasible',solver=f'pyomo:{name}')
        model.rows.add((None if np.isneginf(p.row_lower[i]) else float(p.row_lower[i]),e,None if np.isposinf(p.row_upper[i]) else float(p.row_upper[i])))
    engine=po.SolverFactory(name)
    if not engine.available(exception_flag=False): raise RuntimeError(f'Pyomo solver {name!r} is not installed/available.')
    start=time.perf_counter(); q=engine.solve(model,options=backend,load_solutions=False)
    term=str(q.solver.termination_condition)
    labels={'optimal':'optimal','infeasible':'infeasible','unbounded':'unbounded','infeasibleOrUnbounded':'infeasible_or_unbounded','maxTimeLimit':'time_limit','maxIterations':'iteration_limit'}
    x=None
    if len(q.solution):
        model.solutions.load_from(q)
        x=np.array([po.value(model.x[j]) for j in range(original.n)])
    r=SolveResult(labels.get(term,'solver_error'),x=x,solver=f'pyomo:{name}',message=term,elapsed=time.perf_counter()-start,
                  raw={'options_note':'Only backend_options are forwarded to the external Pyomo solver.'})
    attach_primal(original,r,opts.feasibility_tol)
    if r.success:
        if not r.feasible: r.status='numerical_error'; r.message+=' Original-space primal check failed.'
        else: r.bound=r.objective; r.gap=0.
    return r
