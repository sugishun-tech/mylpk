"""The public modeling interface: direct expressions, indexed variables, bulk matrices."""
from __future__ import annotations
import copy
import itertools
import numpy as np
from scipy import sparse
from mylpk import LinearProblem
from .expression import Expr,Variable,Parameter,Scalar,Constraint,as_expr,scalar_value,bound_value,quicksum,dot,between,_owner

_KINDS={'continuous':0,'integer':1,'binary':1,'semicontinuous':2,'semiinteger':3}


class IndexedVars:
    def __init__(self,variables): self._vars=variables
    def __getitem__(self,key): return self._vars[key]
    def __iter__(self): return iter(self._vars.values())
    def __len__(self): return len(self._vars)
    def keys(self): return self._vars.keys()
    def items(self): return self._vars.items()
    def values(self): return self._vars.values()
    @property
    def solution(self): return {k:v.value for k,v in self._vars.items()}
    def sum(self,*pattern):
        if not pattern: return quicksum(self._vars.values())
        out=[]
        for key,v in self._vars.items():
            k=key if isinstance(key,tuple) else (key,)
            if len(k)!=len(pattern): raise ValueError('Sum pattern dimension mismatch.')
            if all(p=='*' or p==j for p,j in zip(pattern,k)): out.append(v)
        return quicksum(out)
    def dot(self,coefficients): return dot(coefficients,self)


class MatrixBlock:
    def __init__(self,A,variables,lb,ub,name):
        self.A=sparse.coo_matrix(A,dtype=float)
        self.variables=list(variables); self.lb=lb; self.ub=ub; self.name=name; self.row_slice=None; self.model=None
        if self.A.shape[1]!=len(self.variables): raise ValueError('Matrix column/variable count mismatch.')
    @property
    def dual(self):
        r=self.model.last_result
        if r is None or r.dual is None or self.row_slice is None: raise RuntimeError('No current LP duals.')
        return r.dual[self.row_slice].copy()


class Model:
    def __init__(self,name='model'):
        self.name=str(name); self._variables=[]; self._constraints=[]; self._blocks=[]; self._parameters={}
        self._objective=Expr(); self._sense='min'; self._revision=0; self._solution_revision=-1; self._last_result=None
        self._var_names=set(); self._row_names=set(); self._aux=0
    def _touch(self): self._revision+=1
    @property
    def last_result(self): return self._last_result if self._solution_revision==self._revision else None
    def _solution_values(self):
        r=self.last_result
        if r is None or r.x is None or not r.feasible: raise RuntimeError('No feasible solution for the current model revision. Call solve() and inspect its status.')
        return r.x
    def _aux_name(self,prefix):
        while True:
            self._aux+=1; name=f'_{prefix}_{self._aux}'
            if name not in self._var_names and name not in self._row_names: return name
    def var(self,name=None,*,lb=0.,ub=None,kind='continuous'):
        if kind not in _KINDS: raise ValueError(f'kind must be one of {list(_KINDS)}.')
        name=self._aux_name('x') if name is None else str(name)
        if name in self._var_names: raise ValueError(f'Duplicate variable name {name!r}.')
        _owner(self,getattr(lb,'owner',None)); _owner(self,getattr(ub,'owner',None))
        v=Variable(self,len(self._variables),name,lb,ub,kind)
        self._variables.append(v); self._var_names.add(name); self._touch()
        return v
    def vars(self,name,*sets,lb=0.,ub=None,kind='continuous'):
        if not sets: raise ValueError('Provide an index set, length, or Cartesian-product index sets.')
        axes=[range(s) if isinstance(s,int) else list(s) for s in sets]
        def at(spec,key):
            if callable(spec) and not isinstance(spec,Scalar): return spec(*(key if isinstance(key,tuple) else (key,)))
            if isinstance(spec,dict): return spec[key]
            if isinstance(spec,(list,np.ndarray)): return spec[key]
            return spec
        result={}
        for tup in itertools.product(*axes):
            key=tup[0] if len(tup)==1 else tup
            if key in result: raise ValueError('Duplicate variable index.')
            label=','.join(str(t) for t in tup)
            result[key]=self.var(f'{name}[{label}]',lb=at(lb,key),ub=at(ub,key),kind=kind)
        return IndexedVars(result)
    def param(self,name,value):
        if name in self._parameters: raise ValueError(f'Duplicate parameter {name!r}.')
        p=Parameter(self,name,value); self._parameters[name]=p; self._touch(); return p
    def add(self,constraint,name=None):
        if not isinstance(constraint,Constraint):
            if isinstance(constraint,(bool,np.bool_,str,Expr)):
                raise TypeError('Expected a symbolic constraint; a Python boolean is not a model constraint.')
            try: iterator=iter(constraint)
            except TypeError as e: raise TypeError('Expected a Constraint or iterable of constraints.') from e
            return [self.add(c,None if name is None else f'{name}[{i}]') for i,c in enumerate(iterator)]
        _owner(self,constraint.expr.owner)
        _owner(self,getattr(constraint.lower,'owner',None)); _owner(self,getattr(constraint.upper,'owner',None))
        if constraint.model is not None: raise ValueError('A Constraint instance can only be added once.')
        cname=name or constraint.name or self._aux_name('row')
        if cname in self._row_names: raise ValueError(f'Duplicate constraint name {cname!r}.')
        constraint.name=cname; constraint.model=self
        self._constraints.append(constraint); self._row_names.add(cname); self._touch(); return constraint
    def __iadd__(self,constraint): self.add(constraint); return self
    def add_matrix(self,A,variables=None,*,lb=-np.inf,ub=np.inf,name=None):
        vs=self._variables if variables is None else variables
        block=MatrixBlock(A,vs,lb,ub,name or self._aux_name('matrix'))
        for v in block.variables:
            if not isinstance(v,Variable) or v.owner is not self: raise ValueError('Matrix variables must belong to this model.')
        block.model=self; self._blocks.append(block); self._touch(); return block
    def minimize(self,expr):
        e=as_expr(expr); _owner(self,e.owner); self._objective=e; self._sense='min'; self._touch(); return self
    def maximize(self,expr):
        e=as_expr(expr); _owner(self,e.owner); self._objective=e; self._sense='max'; self._touch(); return self
    def compile(self):
        n=len(self._variables); c=np.zeros(n)
        for j,a in self._objective.linear.items(): c[j]=scalar_value(a)
        lower=[]; upper=[]; integ=[]
        for v in self._variables:
            a,b=v.effective_bounds(); lower.append(a); upper.append(b); integ.append(_KINDS[v.kind])
        rr=[]; cc=[]; vv=[]; rl=[]; ru=[]; names=[]; nonlinear=[]
        for con in self._constraints:
            con.row_index=None
            if con.expr.quadratic:
                nonlinear.append(con); continue
            row=len(rl); con.row_index=row
            for j,a in con.expr.linear.items():
                value=scalar_value(a)
                if value: rr.append(row); cc.append(j); vv.append(value)
            constant=scalar_value(con.expr.constant)
            rl.append(bound_value(con.lower,-np.inf)-constant); ru.append(bound_value(con.upper,np.inf)-constant); names.append(con.name)
        A=sparse.csc_matrix((vv,(rr,cc)),shape=(len(rl),n))
        mats=[A]
        for block in self._blocks:
            coo=block.A; ids=np.array([v.index for v in block.variables],dtype=int)
            mats.append(sparse.csc_matrix((coo.data,(coo.row,ids[coo.col])),shape=(coo.shape[0],n)))
            block.row_slice=slice(len(rl),len(rl)+coo.shape[0])
            def bound_array(spec,default):
                if spec is None or isinstance(spec,(Scalar,int,float,np.number)):
                    return [bound_value(spec,default)]*coo.shape[0]
                a=list(spec)
                if len(a)!=coo.shape[0]: raise ValueError('Matrix row bound length mismatch.')
                return [bound_value(v,default) for v in a]
            rl.extend(bound_array(block.lb,-np.inf)); ru.extend(bound_array(block.ub,np.inf))
            names.extend(f'{block.name}[{i}]' for i in range(coo.shape[0]))
        p=LinearProblem(c,sparse.vstack(mats,format='csc'),rl,ru,lower,upper,integ,self._sense,scalar_value(self._objective.constant),
            [v.name for v in self._variables],names,self.name)
        if self._objective.quadratic or nonlinear:
            from .quadratic import QuadraticProblem
            return QuadraticProblem.from_expressions(p,self._objective,nonlinear)
        return p
    def solve(self,solver='mylpk',*,warm_start=False,**kwargs):
        p=self.compile()
        if warm_start:
            if not isinstance(p,LinearProblem) or p.is_mip or solver not in ('mylpk','native'):
                raise ValueError('warm_start=True is for native continuous LP bases only.')
            if 'basis' in kwargs: raise ValueError('Use either warm_start=True or an explicit basis.')
            kwargs['basis']=None if self._last_result is None else self._last_result.basis
        r=p.solve(solver=solver,**kwargs)
        r.model=self; self._last_result=r; self._solution_revision=self._revision
        return r
    def write(self,path): return self.compile().write(path)
    def clone(self):
        # Independent matrix snapshot: mutable parameters are deliberately frozen.
        p=self.compile()
        if not isinstance(p,LinearProblem): raise ValueError('clone() currently snapshots linear models only.')
        return type(self).from_problem(p)
    @classmethod
    def from_problem(cls,p):
        if not isinstance(p,LinearProblem): raise TypeError('Expected LinearProblem.')
        m=cls(p.name)
        kinds=['continuous','integer','semicontinuous','semiinteger']
        vs=[m.var(name,lb=p.lower[j],ub=p.upper[j],kind='binary' if p.integrality[j]==1 and p.lower[j]==0 and p.upper[j]==1 else kinds[p.integrality[j]]) for j,name in enumerate(p.var_names)]
        A=p.A.tocsr()
        for i in range(p.m):
            expr=quicksum(A.data[k]*vs[A.indices[k]] for k in range(A.indptr[i],A.indptr[i+1]))
            m.add(between(p.row_lower[i],expr,p.row_upper[i]),name=p.row_names[i])
        obj=dot(p.c,vs)+p.offset
        (m.minimize if p.sense=='min' else m.maximize)(obj)
        return m
    @classmethod
    def read(cls,path):
        from mylpk.io import read_problem
        return cls.from_problem(read_problem(path))
    def __repr__(self):
        return f'Model({self.name!r}, variables={len(self._variables)}, constraints={len(self._constraints)}, blocks={len(self._blocks)})'
