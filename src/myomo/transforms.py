"""Explicit linear reformulations. Big-M values are derived at compile time."""
from __future__ import annotations
import itertools
import numpy as np
from .expression import Expr,Scalar,Variable,Constraint,as_expr,scalar_value,quicksum,between,bound_value


def _affine(m,expr):
    e=as_expr(expr)
    if e.owner not in (None,m): raise ValueError('Expression belongs to another model.')
    if e.quadratic: raise ValueError('This transformation requires an affine expression.')
    return e


def _binary(m,b):
    if not isinstance(b,Variable) or b.owner is not m or b.kind!='binary':
        raise ValueError('Expected a binary variable from this model.')
    return b


def _finite_bounds(e):
    lo,hi=e.interval()
    if not np.isfinite([lo,hi]).all(): raise ValueError('Finite expression bounds are required for this exact reformulation. Supply variable bounds or an explicit valid M.')
    return lo,hi


def _nonnegative_scalar(value,name):
    out=scalar_value(value)
    if out<0: raise ValueError(f'{name} must remain nonnegative.')
    return out


def indicator(self,b,relation,*,active=1,M=None,name=None):
    _binary(self,b)
    if active not in (0,1) or not isinstance(relation,Constraint): raise ValueError('active must be 0/1 and relation must be a constraint.')
    e=_affine(self,relation.expr)
    # Infinite bounds are structurally omitted; parameters themselves must remain finite.
    forms=[]
    if np.isfinite(bound_value(relation.upper,np.inf)): forms.append(e-as_expr(relation.upper))
    if np.isfinite(bound_value(relation.lower,-np.inf)): forms.append(as_expr(relation.lower)-e)
    ans=[]
    for i,g in enumerate(forms):
        if M is None:
            big=Scalar(lambda g=g:max(0.,_finite_bounds(g)[1]),self)
        else:
            if scalar_value(M)<0: raise ValueError('M must be nonnegative and mathematically valid.')
            big=Scalar(lambda M=M:_nonnegative_scalar(M,'M'),getattr(M,'owner',None))
        ans.append(self.add(g<=big*((1-b) if active==1 else b),name=None if name is None else f'{name}[{i}]'))
    return ans


def abs_epigraph(self,expr,name=None):
    e=_affine(self,expr); t=self.var(name or self._aux_name('abs'),lb=0)
    self.add(t>=e); self.add(t>=-e); return t


def absolute(self,expr,*,exact=False,name=None):
    if not exact: return abs_epigraph(self,expr,name)
    e=_affine(self,expr)
    ub=Scalar(lambda:max(abs(v) for v in _finite_bounds(e)),self)
    t=self.var(name or self._aux_name('abs_exact'),lb=0,ub=ub)
    b=self.var(self._aux_name('sign'),kind='binary')
    indicator(self,b,e>=0); indicator(self,b,t==e)
    indicator(self,b,e<=0,active=0); indicator(self,b,t==-e,active=0)
    return t


def max_epigraph(self,expressions,name=None):
    es=[_affine(self,e) for e in expressions]
    if not es: raise ValueError('max_epigraph requires at least one expression.')
    t=self.var(name or self._aux_name('max'),lb=None)
    self.add(t>=e for e in es); return t


def min_hypograph(self,expressions,name=None):
    es=[_affine(self,e) for e in expressions]
    if not es: raise ValueError('min_hypograph requires at least one expression.')
    t=self.var(name or self._aux_name('min'),lb=None)
    self.add(t<=e for e in es); return t


def product(self,b,expr,name=None):
    _binary(self,b); e=_affine(self,expr)
    lo=Scalar(lambda:_finite_bounds(e)[0],self); hi=Scalar(lambda:_finite_bounds(e)[1],self)
    t=self.var(name or self._aux_name('product'),lb=Scalar(lambda:min(0,scalar_value(lo)),self),ub=Scalar(lambda:max(0,scalar_value(hi)),self))
    self.add(t>=lo*b); self.add(t<=hi*b)
    self.add(t>=e-hi*(1-b)); self.add(t<=e-lo*(1-b))
    return t


def piecewise(self,expr,breakpoints,values,name=None):
    e=_affine(self,expr); bp=np.asarray(breakpoints,dtype=float); val=np.asarray(values,dtype=float)
    if bp.ndim!=1 or len(bp)<2 or val.shape!=bp.shape or not np.isfinite(bp).all() or not np.isfinite(val).all() or np.any(np.diff(bp)<=0):
        raise ValueError('Breakpoints must be finite and strictly increasing, with matching finite values.')
    t=self.var(name or self._aux_name('pwl'),lb=float(val.min()),ub=float(val.max()))
    lam=self.vars(self._aux_name('lambda'),len(bp),lb=0,ub=1)
    z=self.vars(self._aux_name('segment'),len(bp)-1,kind='binary')
    self.add(lam.sum()==1); self.add(z.sum()==1)
    self.add(e==lam.dot(bp)); self.add(t==lam.dot(val))
    self.add(lam[0]<=z[0]); self.add(lam[len(bp)-1]<=z[len(bp)-2])
    for i in range(1,len(bp)-1): self.add(lam[i]<=z[i-1]+z[i])
    return t


def all_different(self,variables):
    vs=list(variables)
    if any(not isinstance(v,Variable) or v.owner is not self or v.kind not in ('integer','binary') for v in vs):
        raise ValueError('all_different needs integer/binary variables from this model.')
    for a,b in itertools.combinations(vs,2):
        z=self.var(self._aux_name('different'),kind='binary')
        indicator(self,z,a-b>=1)
        indicator(self,z,b-a>=1,active=0)
    return self


def logical_and(self,variables,name=None):
    vs=[_binary(self,v) for v in variables]; t=self.var(name or self._aux_name('and'),kind='binary')
    self.add(t<=v for v in vs); self.add(t>=quicksum(vs)-len(vs)+1); return t


def logical_or(self,variables,name=None):
    vs=[_binary(self,v) for v in variables]; t=self.var(name or self._aux_name('or'),kind='binary')
    self.add(t>=v for v in vs); self.add(t<=quicksum(vs)); return t


def logical_xor(self,a,b,name=None):
    _binary(self,a); _binary(self,b); t=self.var(name or self._aux_name('xor'),kind='binary')
    self.add(t<=a+b); self.add(t>=a-b); self.add(t>=b-a); self.add(t<=2-a-b); return t


def soft(self,relation,*,penalty=1.,name=None):
    if not isinstance(relation,Constraint): raise TypeError('soft requires a constraint.')
    e=_affine(self,relation.expr)
    if scalar_value(penalty)<0: raise ValueError('penalty must be nonnegative.')
    slacks=[]
    if np.isfinite(bound_value(relation.lower,-np.inf)):
        s=self.var(self._aux_name(name or 'soft_lower'),lb=0); self.add(e+s>=relation.lower); slacks.append(s)
    if np.isfinite(bound_value(relation.upper,np.inf)):
        s=self.var(self._aux_name(name or 'soft_upper'),lb=0); self.add(e-s<=relation.upper); slacks.append(s)
    cost=Scalar(lambda:_nonnegative_scalar(penalty,'penalty'),getattr(penalty,'owner',None))
    return cost*quicksum(slacks)


def sos1(self,variables):
    vs=list(variables); z=self.vars(self._aux_name('sos1'),len(vs),kind='binary')
    for i,v in enumerate(vs):
        e=_affine(self,v)
        lo=Scalar(lambda e=e:_finite_bounds(e)[0],self); hi=Scalar(lambda e=e:_finite_bounds(e)[1],self)
        self.add(e>=lo*z[i]); self.add(e<=hi*z[i])
    self.add(z.sum()<=1); return z


def sos2(self,variables):
    vs=list(variables)
    if len(vs)<2: raise ValueError('sos2 requires at least two ordered variables.')
    z=self.vars(self._aux_name('sos2'),len(vs)-1,kind='binary'); self.add(z.sum()<=1)
    for i,v in enumerate(vs):
        e=_affine(self,v)
        lo=Scalar(lambda e=e:_finite_bounds(e)[0],self); hi=Scalar(lambda e=e:_finite_bounds(e)[1],self)
        gate=z[0] if i==0 else (z[len(vs)-2] if i==len(vs)-1 else z[i-1]+z[i])
        self.add(e>=lo*gate); self.add(e<=hi*gate)
    return z


def install(Model):
    for fn in (indicator,abs_epigraph,absolute,max_epigraph,min_hypograph,product,piecewise,all_different,logical_and,logical_or,logical_xor,soft,sos1,sos2):
        setattr(Model,fn.__name__,fn)
