"""Convenient matrix APIs with SciPy-style LP input names."""
import numpy as np
from scipy import sparse
from .problem import LinearProblem
from .solvers import solve

def linprog(c,A_ub=None,b_ub=None,A_eq=None,b_eq=None,bounds=(0,None),*,sense='min',offset=0.,solver='mylpk',**kwargs):
    c=np.asarray(c,dtype=float)
    if c.ndim!=1: raise ValueError('c must be a vector.')
    n=len(c); mats=[]; lo=[]; hi=[]
    for A,b,equal in ((A_ub,b_ub,False),(A_eq,b_eq,True)):
        if A is None:
            if b is not None: raise ValueError('RHS was supplied without a constraint matrix.')
            continue
        if b is None: raise ValueError('A constraint matrix requires a RHS.')
        A=sparse.csc_matrix(A,dtype=float); b=np.asarray(b,dtype=float)
        if A.shape[1]!=n or b.shape!=(A.shape[0],): raise ValueError('Constraint shape mismatch.')
        mats.append(A); lo.append(b if equal else np.full(len(b),-np.inf)); hi.append(b)
    if bounds is None: bounds=(0,None)
    arr=np.asarray(bounds,dtype=object)
    if arr.shape==(2,) and all(v is None or np.isscalar(v) for v in arr): arr=np.tile(arr,(n,1))
    if arr.shape!=(n,2): raise ValueError('bounds must be a pair or n bound pairs.')
    lower=np.array([-np.inf if v is None else v for v in arr[:,0]],float)
    upper=np.array([np.inf if v is None else v for v in arr[:,1]],float)
    p=LinearProblem(c,sparse.vstack(mats,format='csc') if mats else None,
        np.concatenate(lo) if lo else -np.inf,np.concatenate(hi) if hi else np.inf,lower,upper,sense=sense,offset=offset)
    return solve(p,solver=solver,**kwargs)


def milp(c,*,A=None,row_lower=-np.inf,row_upper=np.inf,lower=0.,upper=np.inf,integrality=1,sense='min',offset=0.,solver='mylpk',**kwargs):
    return solve(LinearProblem(c,A,row_lower,row_upper,lower,upper,integrality,sense,offset),solver=solver,**kwargs)
