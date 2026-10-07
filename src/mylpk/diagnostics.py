"""Independent primal and dual checks on the original, unscaled matrix."""
from __future__ import annotations
import numpy as np
from .problem import LinearProblem, SolveResult


def violations(p: LinearProblem, x, *, integer=True):
    x = np.asarray(x,dtype=float)
    if x.shape != (p.n,) or not np.isfinite(x).all():
        return {"max_violation": np.inf, "scaled_violation": np.inf, "activity": None}
    activity = np.asarray(p.A @ x).ravel()
    row_scale = 1 + np.asarray(abs(p.A) @ np.abs(x)).ravel()
    row_scale = np.maximum(row_scale, np.where(np.isfinite(p.row_lower),abs(p.row_lower),0))
    row_scale = np.maximum(row_scale, np.where(np.isfinite(p.row_upper),abs(p.row_upper),0))
    rl = np.maximum(p.row_lower-activity,0)
    ru = np.maximum(activity-p.row_upper,0)
    vl = np.maximum(p.lower-x,0); vu = np.maximum(x-p.upper,0)
    semi = p.integrality >= 2
    if integer:
        zero = semi & (np.abs(x) <= 1e-12)
        vl[zero] = 0; vu[zero] = 0
    else:
        vl[semi] = np.maximum(-x[semi],0)
    iv = np.zeros(p.n)
    if integer:
        idx = np.isin(p.integrality,[1,3])
        iv[idx] = np.abs(x[idx]-np.rint(x[idx]))
    values = np.concatenate([rl,ru,vl,vu,iv])
    scaled = np.concatenate([rl/row_scale,ru/row_scale,vl/(1+abs(x)),vu/(1+abs(x)),iv])
    return {"max_violation":float(np.max(values,initial=0)), "scaled_violation":float(np.max(scaled,initial=0)),
            "activity":activity, "row_lower":rl, "row_upper":ru, "lower":vl, "upper":vu, "integrality":iv}


def attach_primal(p, result, tol=1e-7):
    if result.x is None:
        return result
    v = violations(p,result.x)
    result.activity=v["activity"]
    result.max_violation=v["max_violation"]
    result.raw["scaled_primal_violation"]=v["scaled_violation"]
    result.raw["primal_feasible"]=v["scaled_violation"] <= tol
    with np.errstate(over='ignore',invalid='ignore'):
        result.objective=float(p.c @ result.x+p.offset)
    if not np.isfinite(result.objective):
        result.raw['primal_feasible']=False
        result.raw['objective_overflow']=True
    return result


def kkt_report(p: LinearProblem, r: SolveResult):
    if p.is_mip:
        raise ValueError("KKT multipliers certify an LP relaxation, not a MILP optimum.")
    if r.x is None or r.dual is None:
        raise ValueError("A primal solution and dual multipliers are required.")
    lo,up=r.lower_dual,r.upper_dual
    stationarity = p.c-p.A.T @ r.dual-lo-up
    sign=p.sign
    eq=np.isfinite(p.row_lower)&(p.row_lower==p.row_upper)
    dual_err=max(float(np.max(-sign*r.row_lower_dual[~eq],initial=0)),
                 float(np.max(sign*r.row_upper_dual[~eq],initial=0)),
                 float(np.max(-sign*lo,initial=0)),float(np.max(sign*up,initial=0)))
    obj=p.offset
    obj+=float(p.row_lower[np.isfinite(p.row_lower)] @ r.row_lower_dual[np.isfinite(p.row_lower)])
    obj+=float(p.row_upper[np.isfinite(p.row_upper)] @ r.row_upper_dual[np.isfinite(p.row_upper)])
    obj+=float(p.lower[np.isfinite(p.lower)] @ lo[np.isfinite(p.lower)])
    obj+=float(p.upper[np.isfinite(p.upper)] @ up[np.isfinite(p.upper)])
    return {"primal":violations(p,r.x),"stationarity_max":float(np.max(abs(stationarity),initial=0)),
            "dual_sign_violation":max(0.,dual_err),"dual_objective":obj,
            "duality_gap":abs(float(p.c @ r.x+p.offset)-obj)}


def verify_ray(p,x,d,tol=1e-7):
    d=np.asarray(d,dtype=float)
    if d.shape!=(p.n,) or not np.isfinite(d).all(): return {'valid':False,'reason':'invalid vector'}
    Ad=np.asarray(p.A @ d).ravel(); scale=1+np.asarray(abs(p.A) @ abs(d)).ravel()
    bad=0.
    idx=np.isfinite(p.row_lower); bad=max(bad,float(np.max(-Ad[idx]/scale[idx],initial=0)))
    idx=np.isfinite(p.row_upper); bad=max(bad,float(np.max(Ad[idx]/scale[idx],initial=0)))
    idx=np.isfinite(p.lower); bad=max(bad,float(np.max(-d[idx]/(1+abs(d[idx])),initial=0)))
    idx=np.isfinite(p.upper); bad=max(bad,float(np.max(d[idx]/(1+abs(d[idx])),initial=0)))
    improve=p.sign*float(p.c @ d)
    feasible=violations(p,x,integer=False)['scaled_violation']<=tol
    return {'valid':feasible and bad<=tol and improve<0,'direction_violation':bad,'signed_objective_slope':improve,'base_point_feasible':feasible}
