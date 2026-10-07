"""Lexicographic solves, binary-pattern enumeration and parameter scenarios."""
import numpy as np
from .expression import as_expr,quicksum


def solve_lexicographic(self,objectives,*,solver='mylpk',atol=1e-8,rtol=0.,**kwargs):
    if atol<0 or rtol<0 or not np.isfinite([atol,rtol]).all(): raise ValueError('Tolerances must be finite and nonnegative.')
    objectives=list(objectives)
    if not objectives: raise ValueError('At least one objective is required.')
    saved=(self._objective,self._sense,list(self._constraints),self._row_names.copy())
    result=None; values=[]
    try:
        for i,item in enumerate(objectives):
            expr,sense=item if isinstance(item,tuple) else (item,'min')
            if sense not in ('min','max'): raise ValueError('Objective sense must be min/max.')
            expr=as_expr(expr)
            (self.minimize if sense=='min' else self.maximize)(expr)
            result=self.solve(solver=solver,**kwargs)
            if result.status!='optimal': break
            val=expr.evaluate(result.x); values.append(val)
            if i<len(objectives)-1:
                tol=atol+rtol*abs(val)
                self.add(expr<=val+tol if sense=='min' else expr>=val-tol)
    finally:
        self._objective,self._sense,self._constraints,self._row_names=saved
        self._touch()
        if result is not None:
            result.raw['lexicographic_values']=values
            # The returned snapshot belongs to the staged objective, not the restored model.
            self._last_result=None; self._solution_revision=-1
    return result


def enumerate_solutions(self,limit=10,*,variables=None,solver='mylpk',**kwargs):
    if not isinstance(limit,int) or limit<1: raise ValueError('limit must be positive.')
    vs=list(variables) if variables is not None else [v for v in self._variables if v.kind=='binary']
    if not vs or any(v.owner is not self or v.kind!='binary' for v in vs): raise ValueError('Choose at least one binary variable from this model.')
    saved=list(self._constraints),self._row_names.copy()
    results=[]; termination='solution_limit'
    try:
        for _ in range(limit):
            r=self.solve(solver=solver,**kwargs)
            if r.status!='optimal': termination=r.status; break
            results.append(r)
            self.add(quicksum((1-v) if r[v]>=.5 else v for v in vs)>=1)
    finally:
        self._constraints,self._row_names=saved; self._touch()
        if results:
            results[-1].raw['enumeration_termination']=termination
            self._last_result=None; self._solution_revision=-1
    return results


def solve_scenarios(self,scenarios,*,solver='mylpk',**kwargs):
    originals={name:p.value for name,p in self._parameters.items()}; results={}
    try:
        for label,values in scenarios.items():
            # Each scenario starts from the original parameter values.
            for name,v in originals.items(): self._parameters[name].value=v
            for name,v in values.items():
                if name not in self._parameters: raise KeyError(f'Unknown parameter {name!r}.')
                self._parameters[name].value=v
            r=self.solve(solver=solver,**kwargs)
            r.raw['scenario']=dict(values); results[label]=r
    finally:
        for name,v in originals.items(): self._parameters[name].value=v
    return results


def install(Model):
    for fn in (solve_lexicographic,enumerate_solutions,solve_scenarios): setattr(Model,fn.__name__,fn)
