"""Explicit local SciPy adapter for continuous QP/QCQP models.

No convexity or global optimality certificate is claimed for this adapter.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse,optimize
from mylpk.problem import SolveResult
from mylpk.diagnostics import attach_primal
from .expression import scalar_value,bound_value


def coefficients(expr,n):
    c=np.zeros(n); rr=[]; cc=[]; vv=[]
    for j,v in expr.linear.items(): c[j]=scalar_value(v)
    for (i,j),v in expr.quadratic.items():
        value=scalar_value(v)
        if i==j: rr.append(i);cc.append(j);vv.append(2*value)
        else: rr.extend([i,j]);cc.extend([j,i]);vv.extend([value,value])
    return sparse.csc_matrix((vv,(rr,cc)),shape=(n,n)),c,scalar_value(expr.constant)

@dataclass
class QuadraticProblem:
    linear: object
    Q: sparse.csc_matrix
    constraints: list

    @classmethod
    def from_expressions(cls,p,objective,constraints):
        Q,_,_=coefficients(objective,p.n)
        cons=[]
        for con in constraints:
            q,c,k=coefficients(con.expr,p.n)
            cons.append((q,c,k,bound_value(con.lower,-np.inf),bound_value(con.upper,np.inf)))
        return cls(p,Q,cons)
    def solve(self,solver='scipy',*,x0=None,max_iter=1000,tol=1e-9,time_limit=np.inf,**kwargs):
        if solver!='scipy': raise ValueError("Quadratic models currently require explicit solver='scipy'. Native mylpk and GLPK only solve LP/MILP.")
        if kwargs: raise TypeError(f'Unsupported quadratic options: {list(kwargs)}')
        p=self.linear
        if p.is_mip: raise ValueError('Mixed-integer quadratic/nonlinear solving is not implemented.')
        if max_iter<1 or tol<=0 or not np.isfinite(tol) or time_limit<=0 or np.isnan(time_limit): raise ValueError('Invalid quadratic solver limits.')
        if np.any(p.lower>p.upper) or np.any(p.row_lower>p.row_upper): return SolveResult('infeasible',solver='scipy-slsqp')
        x=np.zeros(p.n) if x0 is None else np.asarray(x0,dtype=float)
        if x.shape!=(p.n,) or not np.isfinite(x).all(): raise ValueError('Invalid x0.')
        x=np.minimum(np.maximum(x,p.lower),p.upper)
        cons=[]
        if p.m: cons.append(optimize.LinearConstraint(p.A.toarray(),p.row_lower,p.row_upper))
        for Q,c,k,lo,hi in self.constraints:
            cons.append(optimize.NonlinearConstraint(lambda x,Q=Q,c=c,k=k: .5*x @ (Q @ x)+c @ x+k,lo,hi,
                jac=lambda x,Q=Q,c=c: np.asarray(Q @ x+c)))
        start=time.perf_counter(); last=x.copy()
        class Stop(Exception): pass
        def callback(v):
            nonlocal last
            last=np.array(v,copy=True)
            if time.perf_counter()-start>=time_limit: raise Stop()
        try:
            q=optimize.minimize(lambda x:p.sign*(.5*x @ (self.Q @ x)+p.c @ x+p.offset),x,
                jac=lambda x:p.sign*np.asarray(self.Q @ x+p.c),method='SLSQP',bounds=optimize.Bounds(p.lower,p.upper),
                constraints=cons,callback=callback,options={'maxiter':max_iter,'ftol':tol})
            r=SolveResult('locally_optimal' if q.success else 'local_solver_failure',x=q.x,solver='scipy-slsqp',message=str(q.message),iterations=int(q.nit))
        except Stop:
            r=SolveResult('time_limit',x=last,solver='scipy-slsqp',message='Stopped after a completed SLSQP iteration.')
        attach_primal(p,r,max(1e-7,10*tol))
        nonlinear_violation=0.
        for Q,c,k,lo,hi in self.constraints:
            value=.5*r.x @ (Q @ r.x)+c @ r.x+k
            nonlinear_violation=max(nonlinear_violation,lo-value,value-hi,0.)
        r.max_violation=max(r.max_violation,nonlinear_violation)
        r.raw['primal_feasible']=r.feasible and nonlinear_violation<=max(1e-7,10*tol)
        r.raw['global_optimality_certified']=False
        r.objective=float(.5*r.x @ (self.Q @ r.x)+p.c @ r.x+p.offset)
        r.elapsed=time.perf_counter()-start
        if r.status=='locally_optimal' and not r.feasible: r.status='numerical_error'
        return r
    def write(self,path): raise ValueError('QP/QCQP interchange is not implemented; JSON/LP/MPS writers currently accept linear models only.')
