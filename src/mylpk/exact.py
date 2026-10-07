"""Exact LP certificate CHECKING with Python integers and Fraction.

This is not an exact simplex solver. Input coefficients are interpreted as the
exact rationals represented by their stored decimal float strings. Candidate
primal/dual values are rationally reconstructed, then every condition is checked
without a numerical tolerance. Failed reconstruction is never called a proof.
"""
from fractions import Fraction
import numpy as np


def _input(x): return Fraction(str(float(x)))
def _reconstruct(x,limit): return _input(x).limit_denominator(limit)

def make_exact_certificate(p,result,*,max_denominator=10**9):
    if p.is_mip: raise ValueError('LP dual certificates do not prove integer optimality.')
    if not isinstance(max_denominator,int) or max_denominator<1: raise ValueError('max_denominator must be positive.')
    fields={'x':'x','row_lower_dual':'row_lower_dual','row_upper_dual':'row_upper_dual','lower_dual':'lower_dual','upper_dual':'upper_dual'}
    d={'schema':'mylpk.lp-certificate.v1','input_interpretation':'exact rationals from decimal strings of stored float64 inputs'}
    for k,a in fields.items():
        value=getattr(result,a,None)
        if value is None: raise ValueError(f'{a} is required for an exact LP certificate.')
        d[k]=[str(_reconstruct(v,max_denominator)) for v in value]
    report=verify_exact_certificate(p,d)
    return {'certificate':d,**report}


def verify_exact_certificate(p,certificate):
    if p.is_mip: raise ValueError('Only continuous LP optimality is certified.')
    if certificate.get('schema')!='mylpk.lp-certificate.v1': raise ValueError('Unsupported certificate schema.')
    sizes={'x':p.n,'row_lower_dual':p.m,'row_upper_dual':p.m,'lower_dual':p.n,'upper_dual':p.n}
    a={}
    for k,n in sizes.items():
        if len(certificate.get(k,[]))!=n: raise ValueError(f'Incorrect {k} length.')
        a[k]=[Fraction(v) for v in certificate[k]]
    x=a['x']; dl=a['row_lower_dual']; du=a['row_upper_dual']; xl=a['lower_dual']; xu=a['upper_dual']
    fail=[]; activity=[Fraction(0) for _ in range(p.m)]; station=[_input(v) for v in p.c]
    coo=p.A.tocoo()
    for i,j,v in zip(coo.row,coo.col,coo.data):
        f=_input(v); activity[i]+=f*x[j]; station[j]-=f*(dl[i]+du[i])
    sign=1 if p.sense=='min' else -1
    dual_obj=_input(p.offset)
    for i in range(p.m):
        lo,hi=p.row_lower[i],p.row_upper[i]; eq=np.isfinite(lo) and lo==hi
        if np.isfinite(lo):
            f=_input(lo); dual_obj+=f*dl[i]
            if activity[i]<f: fail.append(f'row_lower[{i}]')
        elif dl[i]: fail.append(f'multiplier_on_absent_row_lower[{i}]')
        if np.isfinite(hi):
            f=_input(hi); dual_obj+=f*du[i]
            if activity[i]>f: fail.append(f'row_upper[{i}]')
        elif du[i]: fail.append(f'multiplier_on_absent_row_upper[{i}]')
        if not eq:
            if sign*dl[i]<0: fail.append(f'row_lower_dual_sign[{i}]')
            if sign*du[i]>0: fail.append(f'row_upper_dual_sign[{i}]')
    for j in range(p.n):
        lo,hi=p.lower[j],p.upper[j]
        if np.isfinite(lo):
            f=_input(lo); dual_obj+=f*xl[j]
            if x[j]<f: fail.append(f'lower[{j}]')
        elif xl[j]: fail.append(f'multiplier_on_absent_lower[{j}]')
        if np.isfinite(hi):
            f=_input(hi); dual_obj+=f*xu[j]
            if x[j]>f: fail.append(f'upper[{j}]')
        elif xu[j]: fail.append(f'multiplier_on_absent_upper[{j}]')
        if sign*xl[j]<0: fail.append(f'lower_dual_sign[{j}]')
        if sign*xu[j]>0: fail.append(f'upper_dual_sign[{j}]')
        station[j]-=xl[j]+xu[j]
        if station[j]: fail.append(f'stationarity[{j}]')
    primal_obj=sum((_input(c)*v for c,v in zip(p.c,x)),_input(p.offset))
    if primal_obj!=dual_obj: fail.append('primal_dual_gap')
    return {'certified':not fail,'failures':fail,'primal_objective':str(primal_obj),'dual_objective':str(dual_obj)}
