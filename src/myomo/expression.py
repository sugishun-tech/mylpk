"""Sparse degree-two algebra with lazy scalar parameters and explicit relations."""
from __future__ import annotations
import numbers
import operator
import math
import numpy as np
from ._accumulate import accumulate


def _owner(a,b):
    if a is not None and b is not None and a is not b:
        raise ValueError('Expressions from different models cannot be combined.')
    return a if a is not None else b


def scalar_value(x):
    value=x.evaluate() if isinstance(x,Scalar) else float(x)
    if not np.isfinite(value): raise ValueError('A symbolic coefficient evaluated to NaN or infinity.')
    return float(value)


def bound_value(x, default):
    if x is None: return default
    value=x.evaluate() if isinstance(x,Scalar) else float(x)
    if np.isnan(value): raise ValueError('A bound evaluated to NaN.')
    return float(value)


class Scalar:
    """A lazily evaluated scalar. It never silently snapshots a Parameter."""
    __array_priority__=10000
    def __init__(self,fn,owner=None): self._fn=fn; self.owner=owner
    def evaluate(self): return float(self._fn())
    def _op(self,other,fn,reverse=False):
        if isinstance(other,Expr): return NotImplemented
        if not isinstance(other,(Scalar,numbers.Real)): return NotImplemented
        owner=_owner(self.owner,getattr(other,'owner',None))
        return Scalar(lambda:fn(scalar_value(other),scalar_value(self)) if reverse else fn(scalar_value(self),scalar_value(other)),owner)
    def __add__(self,o): return self._op(o,operator.add)
    def __radd__(self,o): return self._op(o,operator.add,True)
    def __sub__(self,o): return self._op(o,operator.sub)
    def __rsub__(self,o): return self._op(o,operator.sub,True)
    def __mul__(self,o): return self._op(o,operator.mul)
    def __rmul__(self,o): return self._op(o,operator.mul,True)
    def __truediv__(self,o): return self._op(o,operator.truediv)
    def __rtruediv__(self,o): return self._op(o,operator.truediv,True)
    def __neg__(self): return Scalar(lambda:-scalar_value(self),self.owner)
    def __pow__(self,o): return self._op(o,operator.pow)
    def __float__(self): return scalar_value(self)
    def __bool__(self): raise TypeError('A symbolic scalar is not a boolean; use .value explicitly.')
    def __repr__(self): return f'Scalar({self.evaluate():g})'


class Parameter(Scalar):
    def __init__(self,owner,name,value):
        self.name=name; self._value=scalar_value(value)
        super().__init__(lambda:self._value,owner)
    @property
    def value(self): return self._value
    @value.setter
    def value(self,v):
        self._value=scalar_value(v)
        if self.owner is not None: self.owner._touch()
    def __repr__(self): return f'Parameter({self.name!r}, value={self.value:g})'


def _nonzero(v): return isinstance(v,Scalar) or v!=0


class Expr:
    __array_priority__=10000
    def __init__(self,owner=None,linear=None,quadratic=None,constant=0.):
        self.owner=owner
        self.linear={} if linear is None else {k:v for k,v in linear.items() if _nonzero(v)}
        self.quadratic={} if quadratic is None else {k:v for k,v in quadratic.items() if _nonzero(v)}
        self.constant=constant
    @property
    def degree(self): return 2 if self.quadratic else (1 if self.linear else 0)
    def __add__(self,other):
        other=as_expr(other); owner=_owner(self.owner,other.owner)
        lin=self.linear.copy(); quad=self.quadratic.copy()
        for k,v in other.linear.items(): lin[k]=lin.get(k,0.)+v
        for k,v in other.quadratic.items(): quad[k]=quad.get(k,0.)+v
        return Expr(owner,lin,quad,self.constant+other.constant)
    __radd__=__add__
    def __neg__(self): return self._scale(-1.)
    def __sub__(self,o): return self+(-as_expr(o))
    def __rsub__(self,o): return as_expr(o)+(-self)
    def _scale(self,k):
        return Expr(_owner(self.owner,getattr(k,'owner',None)),{j:v*k for j,v in self.linear.items()},
                    {j:v*k for j,v in self.quadratic.items()},self.constant*k)
    def __mul__(self,o):
        other=as_expr(o); _owner(self.owner,other.owner)
        if other.degree==0: return self._scale(other.constant)
        if self.degree==0: return other._scale(self.constant)
        if self.degree+other.degree>2:
            raise TypeError('Only polynomials up to degree two are supported. Use an explicit external NLP model for higher degrees.')
        ans=self._scale(other.constant)+other._scale(self.constant)
        ans.constant=self.constant*other.constant
        quad=ans.quadratic.copy()
        for i,a in self.linear.items():
            for j,b in other.linear.items():
                key=(min(i,j),max(i,j)); quad[key]=quad.get(key,0.)+a*b
        return Expr(ans.owner,ans.linear,quad,ans.constant)
    __rmul__=__mul__
    def __truediv__(self,o):
        e=as_expr(o)
        if e.degree: raise TypeError('Division by a decision variable is not polynomial modeling.')
        return self._scale(1./e.constant)
    def __pow__(self,k):
        if k==0: return Expr(constant=1.)
        if k==1: return self
        if k==2: return self*self
        raise TypeError('Only powers 0, 1 and 2 are supported.')
    def __le__(self,o): return Constraint(self-as_expr(o),upper=0.)
    def __ge__(self,o): return Constraint(self-as_expr(o),lower=0.)
    def __eq__(self,o): return Constraint(self-as_expr(o),lower=0.,upper=0.)
    def __lt__(self,o): raise TypeError('Strict inequalities are not supported; use <= or an explicit margin.')
    def __gt__(self,o): raise TypeError('Strict inequalities are not supported; use >= or an explicit margin.')
    def __ne__(self,o): raise TypeError('Use Model.all_different for bounded integer inequalities.')
    def __bool__(self): raise TypeError('A symbolic expression is not a boolean. Use between(lb, expr, ub), not chained comparisons.')
    def evaluate(self,x=None):
        if x is None:
            if self.owner is None: return scalar_value(self.constant)
            x=self.owner._solution_values()
        out=scalar_value(self.constant)
        out+=sum(scalar_value(v)*x[j] for j,v in self.linear.items())
        out+=sum(scalar_value(v)*x[i]*x[j] for (i,j),v in self.quadratic.items())
        return float(out)
    @property
    def value(self): return self.evaluate()
    def interval(self):
        if self.quadratic: raise ValueError('Automatic bounds require an affine expression.')
        lo=hi=scalar_value(self.constant)
        for j,v in self.linear.items():
            a=scalar_value(v)
            if a==0: continue
            x=self.owner._variables[j]; l,u=x.effective_bounds()
            if x.kind in ('semicontinuous','semiinteger'): l=min(0.,l)
            if a>0: lo+=a*l; hi+=a*u
            else: lo+=a*u; hi+=a*l
        return float(lo),float(hi)
    def __repr__(self):
        terms=[]
        for j,a in self.linear.items():
            name=self.owner._variables[j].name if self.owner is not None else f'x{j}'
            terms.append(f'{scalar_value(a):g}*{name}')
        for (i,j),a in self.quadratic.items():
            vi=self.owner._variables[i].name; vj=self.owner._variables[j].name
            terms.append(f'{scalar_value(a):g}*{vi}*{vj}')
        c=scalar_value(self.constant)
        if c or not terms: terms.append(f'{c:g}')
        return ' + '.join(terms)


def as_expr(value):
    if isinstance(value,Expr): return value
    if isinstance(value,Scalar): return Expr(owner=value.owner,constant=value)
    if isinstance(value,numbers.Real): return Expr(constant=float(value))
    raise TypeError(f'Expected a number or symbolic expression, got {type(value).__name__}.')


def quicksum(values):
    owner,lin,quad,c=accumulate(as_expr(v) for v in values)
    return Expr(owner,lin,quad,c)


def dot(coefficients,variables):
    vv=list(variables.values()) if hasattr(variables,'values') and not isinstance(variables,np.ndarray) else list(variables)
    if isinstance(coefficients,dict) and hasattr(variables,'keys'):
        return quicksum(coefficients[k]*variables[k] for k in variables.keys())
    cc=list(coefficients)
    if len(cc)!=len(vv): raise ValueError('dot coefficient and variable lengths differ.')
    return quicksum(a*x for a,x in zip(cc,vv))


class Variable(Expr):
    def __init__(self,owner,index,name,lb,ub,kind):
        super().__init__(owner,{index:1.})
        self.index=index; self.name=name; self._lb=lb; self._ub=ub; self._kind=kind
    def effective_bounds(self):
        lo=bound_value(self._lb,-np.inf); hi=bound_value(self._ub,np.inf)
        if self.kind=='binary': lo=max(0.,lo); hi=min(1.,hi)
        return lo,hi
    @property
    def lb(self): return self._lb
    @lb.setter
    def lb(self,v):
        _owner(self.owner,getattr(v,'owner',None)); bound_value(v,-np.inf)
        self._lb=v; self.owner._touch()
    @property
    def ub(self): return self._ub
    @ub.setter
    def ub(self,v):
        _owner(self.owner,getattr(v,'owner',None)); bound_value(v,np.inf)
        self._ub=v; self.owner._touch()
    @property
    def kind(self): return self._kind
    @kind.setter
    def kind(self,value):
        if value not in ('continuous','integer','binary','semicontinuous','semiinteger'): raise ValueError('Unsupported variable domain.')
        self._kind=value; self.owner._touch()
    def fix(self,value):
        _owner(self.owner,getattr(value,'owner',None)); scalar_value(value)
        self._lb=value; self._ub=value; self.owner._touch(); return self
    @property
    def value(self): return float(self.owner._solution_values()[self.index])
    def __repr__(self): return self.name
    __hash__=object.__hash__


class Constraint:
    def __init__(self,expr,lower=-np.inf,upper=np.inf,name=None):
        self.expr=as_expr(expr); self._lower=lower; self._upper=upper; self.name=name
        self.model=None; self.row_index=None
    @property
    def lower(self): return self._lower
    @lower.setter
    def lower(self,value):
        _owner(self.model or self.expr.owner,getattr(value,'owner',None)); bound_value(value,-np.inf)
        self._lower=value
        if self.model is not None: self.model._touch()
    @property
    def upper(self): return self._upper
    @upper.setter
    def upper(self,value):
        _owner(self.model or self.expr.owner,getattr(value,'owner',None)); bound_value(value,np.inf)
        self._upper=value
        if self.model is not None: self.model._touch()
    def __bool__(self):
        raise TypeError('A constraint is not a bool. Chained comparisons and Python and/or do not build constraints. Use between() or Model.logical_*().')
    @property
    def dual(self):
        r=None if self.model is None else self.model.last_result
        if r is None or r.dual is None or self.row_index is None:
            raise RuntimeError('No current LP dual value is available for this constraint.')
        return float(r.dual[self.row_index])
    @property
    def slack(self):
        v=self.expr.value
        return min(v-bound_value(self.lower,-np.inf),bound_value(self.upper,np.inf)-v)


def between(lower,expr,upper): return Constraint(expr,lower,upper)
