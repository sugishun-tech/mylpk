"""Dual construction, basis sensitivity, irreducible infeasibility diagnosis."""
from __future__ import annotations
import numpy as np
from scipy import sparse,linalg
from .problem import LinearProblem,Options
from .standard import standardize
from .solvers import solve


def dual_problem(p: LinearProblem):
    if p.is_mip: raise ValueError('Construct an LP relaxation before taking an LP dual.')
    eq=np.flatnonzero(np.isfinite(p.row_lower)&(p.row_lower==p.row_upper))
    lo=np.flatnonzero(np.isfinite(p.row_lower)&(p.row_lower!=p.row_upper))
    up=np.flatnonzero(np.isfinite(p.row_upper)&(p.row_lower!=p.row_upper))
    vl=np.flatnonzero(np.isfinite(p.lower)); vu=np.flatnonzero(np.isfinite(p.upper))
    eye=sparse.eye(p.n,format='csc')
    G=sparse.vstack([p.A[lo],-p.A[up],eye[vl],-eye[vu]],format='csc')
    h=np.r_[p.row_lower[lo],-p.row_upper[up],p.lower[vl],-p.upper[vu]]
    E=p.A[eq]; f=p.row_lower[eq]
    A=sparse.hstack([G.T,E.T],format='csc')
    names=[f'row_lower_{i}' for i in lo]+[f'row_upper_{i}' for i in up]+[f'bound_lower_{j}' for j in vl]+[f'bound_upper_{j}' for j in vu]+[f'equality_{i}' for i in eq]
    return LinearProblem(p.sign*np.r_[h,f],A,p.sign*p.c,p.sign*p.c,
        np.r_[np.zeros(len(h)),np.full(len(f),-np.inf)],np.inf,0,
        'max' if p.sense=='min' else 'min',p.offset,names,[f'stationarity_{j}' for j in range(p.n)],p.name+'_dual')


def _nonnegative_interval(value,direction,tol=1e-10):
    lower=-np.inf; upper=np.inf
    for x,d in zip(value,direction):
        if d>tol: lower=max(lower,-x/d)
        elif d<-tol: upper=min(upper,-x/d)
    return [lower,upper]


def basis_sensitivity(p,result,*,max_columns=2000):
    """One-coordinate allowable deltas in SCALED STANDARD-FORM coordinates.

    Bounds apply with the same basis and all other coefficients fixed. They
    are neither simultaneous uncertainty regions nor original-coordinate ranges.
    """
    if p.is_mip or result.status!='optimal' or result.basis is None:
        raise ValueError('A native optimal LP basis is required.')
    s=result.raw.get('standard')
    if s is None: raise ValueError('Standard-form postsolve data is unavailable.')
    current=standardize(p,Options(scaling=result.basis.scaling))
    if current.signature!=s.signature or not np.array_equal(current.b,s.b) or not np.array_equal(current.c,s.c):
        raise ValueError('The problem has changed since this basis was solved.')
    Bidx=result.basis.indices
    if np.any(Bidx>=s.real_columns): raise ValueError('Redundant-row artificial basics must be removed before basis sensitivity; this rank-deficient case is not supported.')
    if s.real_columns>max_columns: raise MemoryError('Sensitivity is dense; raise max_columns explicitly for a larger analysis.')
    B=s.A[:,Bidx].toarray(); inv=linalg.solve(B,np.eye(B.shape[0]),assume_a='gen')
    xb=inv @ s.b
    rhs=np.array([_nonnegative_interval(xb,inv[:,j]) for j in range(len(s.b))])
    mask=np.ones(s.real_columns,dtype=bool); mask[Bidx]=False; nonbasic=np.flatnonzero(mask)
    D=inv @ s.A[:,nonbasic].toarray()
    rc=s.c[nonbasic]-s.A[:,nonbasic].T @ linalg.solve(B.T,s.c[Bidx])
    cost=np.tile([-np.inf,np.inf],(s.real_columns,1))
    for k,j in enumerate(Bidx): cost[j]=_nonnegative_interval(rc,-D[k])
    for k,j in enumerate(nonbasic): cost[j]=[-max(0.,rc[k]),np.inf]
    return {'coordinate_system':'scaled_standard_form','rhs_delta':rhs,'cost_delta':cost,
        'basis':Bidx.copy(),'basic_values':xb,'row_provenance':s.row_meta,
        'row_factor':s.row_factor.copy(),'objective_scale':s.cost_scale,
        'structural_columns':s.structural_columns}


def find_iis(p,*,solver='mylpk',include_bounds=True,relax_integrality=True,max_checks=1000,**solver_options):
    """Deletion filter for an inclusion-minimal infeasible subsystem.

    Each ranged/equality row is a single group. This does NOT minimize the
    cardinality. Limits/unknown solver outcomes stop the proof of irreducibility.
    """
    if not isinstance(max_checks,int) or max_checks<1: raise ValueError('max_checks must be positive.')
    if not relax_integrality and np.any(p.integrality>=2):
        raise ValueError('IIS with semi-variable integrality is not supported; use the LP relaxation or an explicit binary reformulation.')
    q=p.relax() if relax_integrality else p
    first=solve(q,solver=solver,**solver_options); calls=1
    if first.status!='infeasible':
        return {'status':'not_proven_infeasible','solver_status':first.status,'irreducible':False,'checks':calls,'members':[]}
    members=[('row',i) for i in range(q.m)]
    if include_bounds:
        members += [('lower',int(j)) for j in np.flatnonzero(np.isfinite(q.lower))]
        members += [('upper',int(j)) for j in np.flatnonzero(np.isfinite(q.upper))]
    active=set(members)
    def subsystem(groups):
        rl=q.row_lower.copy(); ru=q.row_upper.copy(); lb=q.lower.copy(); ub=q.upper.copy()
        for kind,i in members:
            if (kind,i) not in groups:
                if kind=='row': rl[i]=-np.inf; ru[i]=np.inf
                elif kind=='lower': lb[i]=-np.inf
                else: ub[i]=np.inf
        return q.replace(row_lower=rl,row_upper=ru,lower=lb,upper=ub)
    complete=True; stop=None
    for item in members:
        if calls>=max_checks: complete=False; stop='check_limit'; break
        trial=active-{item}; r=solve(subsystem(trial),solver=solver,**solver_options); calls+=1
        if r.status=='infeasible': active=trial
        elif r.status not in ('optimal','unbounded'):
            complete=False; stop=r.status; break
    result=[]
    for kind,i in members:
        if (kind,i) in active: result.append({'kind':kind,'index':i,'name':q.row_names[i] if kind=='row' else q.var_names[i]})
    return {'status':'irreducible' if complete else 'partial','irreducible':complete,'checks':calls,'members':result,
        'relaxed_integrality':relax_integrality,'grouping':'whole rows and individual variable bounds','stop_reason':stop}
