"""Best-bound branch-and-bound, pseudocosts, and binary cover separation.

LP relaxations are solved by our Cython core, never delegated to HiGHS/GLPK.
The search scheduler runs in Python; all numerical pivots run in Cython.
"""
from __future__ import annotations
from dataclasses import replace
import heapq
import math
import time
import numpy as np
from scipy import sparse
from .problem import LinearProblem, Options, SolveResult
from .lp import solve_lp
from .diagnostics import attach_primal, violations


def expand_semivariables(p: LinearProblem):
    ids=np.flatnonzero(p.integrality>=2)
    if not len(ids): return p, []
    n=p.n; k=len(ids)
    lo=p.lower.copy(); lo[ids]=0
    integ=p.integrality.copy(); integ[ids]=(integ[ids]==3).astype(np.int8)
    rr=[]; cc=[]; vv=[]
    for t,j in enumerate(ids):
        rr += [2*t,2*t,2*t+1,2*t+1]
        cc += [int(j),n+t,int(j),n+t]
        vv += [1.,-p.upper[j],-1.,p.lower[j]]
    extra=sparse.csc_matrix((vv,(rr,cc)),shape=(2*k,n+k))
    A=sparse.vstack([sparse.hstack([p.A,sparse.csc_matrix((p.m,k))]),extra],format='csc')
    def unique(prefix,used):
        name=prefix
        while name in used: name+='_'
        used.add(name); return name
    used=set(p.var_names); vn=p.var_names+[unique(f'_semi_{j}',used) for j in ids]
    used=set(p.row_names); rn=p.row_names+[unique(f'_semi_link_{t}',used) for t in range(2*k)]
    q=LinearProblem(np.r_[p.c,np.zeros(k)],A,np.r_[p.row_lower,np.full(2*k,-np.inf)],
        np.r_[p.row_upper,np.zeros(2*k)],np.r_[lo,np.zeros(k)],np.r_[p.upper,np.ones(k)],
        np.r_[integ,np.ones(k,dtype=np.int8)],p.sense,p.offset,vn,rn,p.name)
    return q, list(ids)


def cover_cuts(p, x, seen, tol=1e-7, limit=32):
    """Valid minimal cover inequalities on binary-only rows, including complements."""
    A=p.A.tocsr(); cuts=[]; bounds=[]
    binary=(p.integrality==1)&(p.lower>=0)&(p.upper<=1)
    for i in range(p.m):
        for direction in (1.,-1.):
            rhs=p.row_upper[i] if direction==1 else -p.row_lower[i]
            if not np.isfinite(rhs): continue
            sl=slice(A.indptr[i],A.indptr[i+1]); js=A.indices[sl]; co=direction*A.data[sl]
            if not len(js) or not binary[js].all(): continue
            positive=co>0; a=abs(co)
            h=rhs-float(np.sum(co[~positive]))
            y=np.where(positive,x[js],1-x[js])
            if h<0: continue
            orders=[np.argsort(-a),np.argsort((1-y)/np.maximum(a,1e-300))]
            for order in orders:
                cover=[]; total=0.
                for t in order:
                    cover.append(int(t)); total+=a[t]
                    if total>h+tol: break
                if total<=h+tol: continue
                for t in sorted(cover,key=lambda t:a[t]):
                    if total-a[t]>h+tol:
                        cover.remove(t); total-=a[t]
                if sum(y[t] for t in cover)<=len(cover)-1+tol: continue
                key=tuple(sorted((int(js[t]),1 if positive[t] else -1) for t in cover))
                if key in seen: continue
                seen.add(key)
                j=[k[0] for k in key]; v=[k[1] for k in key]
                cuts.append(sparse.csc_matrix((v,([0]*len(j),j)),shape=(1,p.n)))
                bounds.append(len(cover)-1-sum(not positive[t] for t in cover))
                if len(cuts)>=limit: return cuts,bounds
    return cuts,bounds


def solve_mip(original: LinearProblem, opts: Options, *, x0=None, callback=None):
    start=time.perf_counter()
    p,semi_ids=expand_semivariables(original)
    lo=p.lower.copy(); hi=p.upper.copy(); integer=p.integrality==1
    lo[integer]=np.ceil(lo[integer]); hi[integer]=np.floor(hi[integer])
    p=p.replace(lower=lo,upper=hi)
    if np.any(lo>hi) or np.any(p.row_lower>p.row_upper):
        return SolveResult('infeasible',message='Integer bounds contain no admissible point.')
    incumbent=None; best=np.inf; lp_calls=0; total_iters=0; warm_hits=0; cut_count=0
    nodes=0; serial=0; status='optimal'; message=''; unseen=-np.inf
    pseudo_up=np.ones(p.n); pseudo_down=np.ones(p.n)
    count_up=np.zeros(p.n); count_down=np.zeros(p.n)
    def remaining(): return opts.time_limit-(time.perf_counter()-start)
    def accept(x):
        nonlocal incumbent,best
        if violations(p,x)['scaled_violation']>opts.feasibility_tol: return False
        # Integrality is checked independently with its own absolute tolerance.
        if np.max(abs(x[integer]-np.rint(x[integer])),initial=0)>opts.integrality_tol: return False
        f=p.sign*float(p.c @ x+p.offset)
        if f<best:
            best=f; incumbent=x.copy()
            return True
        return False
    if x0 is not None:
        x=np.asarray(x0,dtype=float)
        if x.shape!=(original.n,) or not np.isfinite(x).all(): raise ValueError('x0 has invalid shape/values.')
        if semi_ids: x=np.r_[x,[float(x[j]!=0) for j in semi_ids]]
        if not accept(x): raise ValueError('x0 is not a feasible integer incumbent.')
    def lp(q,basis=None):
        nonlocal lp_calls,total_iters,warm_hits
        if remaining()<=0: return SolveResult('time_limit')
        r=solve_lp(q.replace(integrality=0),replace(opts,time_limit=remaining()),basis=basis)
        lp_calls+=1; total_iters+=r.iterations; warm_hits+=bool(r.raw.get('warm_start_used'))
        return r
    def rounded_heuristic(x,node_lo,node_hi):
        a=node_lo.copy(); b=node_hi.copy()
        target=np.rint(x[integer]); target=np.minimum(np.maximum(target,a[integer]),b[integer])
        a[integer]=target; b[integer]=target
        r=lp(p.replace(lower=a,upper=b))
        if r.success:
            candidate=r.x.copy(); candidate[integer]=target
            accept(candidate)
    # Heap item: proven bound, serial, variable lower/upper bounds, parent basis,
    # branching variable, direction, distance, parent's LP bound.
    heap=[(-np.inf,serial,lo,hi,None,-1,0,1.,-np.inf)]
    seen=set()
    while heap:
        if remaining()<=0: status='time_limit'; break
        if nodes>=opts.node_limit: status='node_limit'; break
        bound,_,nl,nu,basis,bj,bd,dist,parent_bound=heapq.heappop(heap)
        unseen=bound
        if bound>=best-opts.mip_abs_gap:
            unseen=np.inf; continue
        r=lp(p.replace(lower=nl,upper=nu),basis)
        nodes+=1
        if r.status=='infeasible': unseen=np.inf; continue
        if r.status=='unbounded':
            status='relaxation_unbounded'
            message='An unbounded LP relaxation alone does not certify integer unboundedness.'
            break
        if not r.success:
            status=r.status; message=r.message or 'An LP node could not be certified.'; break
        f=p.sign*r.objective
        if bj>=0 and np.isfinite(parent_bound):
            improvement=max(0.,f-parent_bound)/max(dist,opts.integrality_tol)
            costs,counts=(pseudo_up,count_up) if bd>0 else (pseudo_down,count_down)
            costs[bj]=(costs[bj]*counts[bj]+improvement)/(counts[bj]+1)
            counts[bj]+=1
        if nodes==1 and opts.cuts:
            for _ in range(4):
                cuts,ubs=cover_cuts(p,r.x,seen,opts.feasibility_tol)
                if not cuts: break
                names=[]; used=set(p.row_names)
                for t in range(len(cuts)):
                    name=f'_cover_{cut_count+t}'
                    while name in used: name+='_' 
                    used.add(name); names.append(name)
                p=p.replace(A=sparse.vstack([p.A,*cuts],format='csc'),
                    row_lower=np.r_[p.row_lower,np.full(len(cuts),-np.inf)],row_upper=np.r_[p.row_upper,ubs],row_names=p.row_names+names)
                cut_count+=len(cuts)
                next_r=lp(p.replace(lower=nl,upper=nu))
                if not next_r.success:
                    if next_r.status=='infeasible':
                        r=next_r
                    else:
                        # Keep the previous valid bound; the newly cut node is unfinished.
                        unseen=f; status=next_r.status; message='Root cut reoptimization did not complete.'
                    break
                r=next_r; f=p.sign*r.objective
            if status!='optimal': break
            if r.status=='infeasible': unseen=np.inf; continue
        if f>=best-opts.mip_abs_gap: unseen=np.inf; continue
        frac=np.abs(r.x-np.rint(r.x))
        candidates=np.flatnonzero(integer&(frac>opts.integrality_tol))
        if not candidates.size:
            z=r.x.copy(); z[integer]=np.rint(z[integer])
            if violations(p,z)['scaled_violation']>opts.feasibility_tol:
                status='numerical_error'; message='The integral candidate failed postsolve validation.'; unseen=f; break
            accept(z)
            unseen=np.inf
        else:
            if opts.heuristic and (nodes==1 or nodes%32==0): rounded_heuristic(r.x,nl,nu)
            if f<best-opts.mip_abs_gap:
                down=r.x[candidates]-np.floor(r.x[candidates]); up=1-down
                a=pseudo_down[candidates]*down; b=pseudo_up[candidates]*up
                score=np.minimum(a,b)+0.1*np.maximum(a,b)
                j=int(candidates[np.argmax(score)])
                floor=math.floor(r.x[j]); ceil=floor+1
                left_hi=nu.copy(); left_hi[j]=min(left_hi[j],floor)
                right_lo=nl.copy(); right_lo[j]=max(right_lo[j],ceil)
                if nl[j]<=left_hi[j]:
                    serial+=1
                    heapq.heappush(heap,(f,serial,nl.copy(),left_hi,r.basis,j,-1,r.x[j]-floor,f))
                if right_lo[j]<=nu[j]:
                    serial+=1
                    heapq.heappush(heap,(f,serial,right_lo,nu.copy(),r.basis,j,1,ceil-r.x[j],f))
            unseen=np.inf
        lower_bound=min(heap[0][0] if heap else best,best)
        if callback is not None:
            event={'nodes':nodes,'incumbent':None if incumbent is None else p.sign*best,
                   'bound':p.sign*lower_bound,'elapsed':time.perf_counter()-start,'cuts':cut_count}
            if callback(event) is False:
                status='user_stop'; message='Stopped by the node callback.'; break
        if incumbent is not None and heap:
            gap_abs=max(0.,best-lower_bound)
            gap_rel=gap_abs/max(1.,abs(best))
            if gap_abs<=opts.mip_abs_gap:
                # These nodes are all prunable within the absolute objective tolerance.
                heap=[]; break
            if gap_rel<=opts.mip_rel_gap:
                status='gap_limit'; message='The requested relative MIP gap was reached.'; break
    if not heap and np.isinf(unseen) and unseen>0 and incumbent is None and status=='optimal': status='infeasible'
    if incumbent is None and not heap and status=='optimal': status='infeasible'
    lower_bound=min(heap[0][0] if heap else best,unseen,best)
    if status=='optimal' and incumbent is not None: lower_bound=best
    result=SolveResult(status,x=None if incumbent is None else incumbent[:original.n].copy(),
        solver='mylpk',iterations=total_iters,nodes=nodes,
        bound=None if not np.isfinite(lower_bound) else p.sign*lower_bound,
        gap=None if incumbent is None or not np.isfinite(lower_bound) else max(0.,best-lower_bound)/max(1.,abs(best)),
        elapsed=time.perf_counter()-start,message=message,
        raw={'lp_calls':lp_calls,'warm_start_hits':warm_hits,'cover_cuts':cut_count,'expanded_semivariables':len(semi_ids),
             'relative_gap_denominator':'max(1, abs(incumbent_objective))'})
    attach_primal(original,result,opts.feasibility_tol)
    if result.x is not None and not result.feasible:
        result.status='numerical_error'; result.message='Final MILP incumbent failed original-space validation.'
    return result
