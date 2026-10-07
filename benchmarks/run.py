"""Reproducible, correctness-checked end-to-end microbenchmarks.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/run.py.
No instance data is selected after inspecting performance. All generated cases
and every repeat are retained, including non-optimal solver outcomes.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import scipy
from scipy import sparse
from mylpk import LinearProblem, LPSession, solve
from myomo import Model, quicksum


def cases():
    rng=np.random.default_rng(20261007)
    for name,m,n,density in [('dense_20x40',20,40,1.),('dense_80x160',80,160,1.),('sparse_120x600',120,600,.06)]:
        A=rng.uniform(.1,2,(m,n))
        if density<1: A*=rng.random((m,n))<density
        # One nonzero in every column guarantees boundedness without bound rows.
        A[np.arange(n)%m,np.arange(n)]+=1.
        yield name,LinearProblem(-rng.uniform(.1,2,n),sparse.csc_matrix(A),row_upper=A@rng.uniform(.1,1,n))
    n=12
    costs=rng.uniform(1,20,(n,n))
    rows=[]; cols=[]
    for i in range(n):
        for j in range(n):
            rows.extend([i,n+j]); cols.extend([i*n+j]*2)
    A=sparse.csc_matrix((np.ones(2*n*n),(rows,cols)),shape=(2*n,n*n))
    rhs=np.full(2*n,10.)
    yield 'transport_12x12',LinearProblem(costs.ravel(),A,rhs,rhs)
    n=18; A=rng.integers(1,15,(3,n)).astype(float)
    yield 'binary_knapsack_18',LinearProblem(-rng.integers(1,40,n),A,row_upper=np.floor(.35*A.sum(axis=1)),upper=1.,integrality=1)


def timed(fn,repeats):
    fn()  # A warm-up for imports, shared-library initialization and allocator state.
    times=[]; results=[]
    for _ in range(repeats):
        start=time.perf_counter(); result=fn(); times.append(time.perf_counter()-start); results.append(result)
    return times,results


def check_pair(p,a,b):
    if a.status!='optimal' or b.status!='optimal': return False
    return bool(a.feasible and b.feasible and abs(a.objective-b.objective)<=1e-6*(1+abs(b.objective)))


def run(repeats=5):
    rows=[]
    for name,p in cases():
        print('solve',name,flush=True)
        opt=dict(time_limit=10.,mip_rel_gap=0.,mip_abs_gap=0.)
        answer={}
        for backend in ('mylpk','highs'):
            durations,results=timed(lambda:solve(p,solver=backend,**opt),repeats)
            answer[backend]=(durations,results)
        a=answer['mylpk'][1][-1];b=answer['highs'][1][-1]
        row={'case':name,'shape':list(p.A.shape),'nnz':p.A.nnz,'integral_variables':int(np.count_nonzero(p.integrality)),
             'objectives_agree':all(check_pair(p,r,b) for _,rs in answer.values() for r in rs)}
        for backend,(durations,results) in answer.items():
            row[backend]={'median_ms':1000*statistics.median(durations),'samples_ms':[1000*v for v in durations],
                          'statuses':[r.status for r in results],'objectives':[r.objective for r in results],
                          'iterations':[r.iterations for r in results],'nodes':[r.nodes for r in results],
                          'feasible':[r.feasible for r in results]}
        row['native_over_highs_time']=row['mylpk']['median_ms']/row['highs']['median_ms']
        rows.append(row)
    print('objective updates',flush=True)
    rng=np.random.default_rng(34177); A=rng.uniform(.1,1.,(40,100)); c=-rng.uniform(.2,1.,100)
    base=LinearProblem(c,A,row_upper=A@np.ones(100)*.4)
    costs=[c+rng.normal(0,.005,100) for _ in range(20)]
    problems=[base.replace(c=v) for v in costs]
    def persistent():
        session=LPSession(base)
        return [session.last_result]+[session.resolve(c=v) for v in costs]
    functions={
        'native_session_21_including_initial':persistent,
        'native_cold_21':lambda:[solve(p) for p in [base]+problems],
        'highs_cold_21':lambda:[solve(p,solver='highs') for p in [base]+problems]}
    batch={}; answers={}
    for name,fn in functions.items():
        durations,results=timed(fn,repeats); answers[name]=results[-1]
        batch[name]={'median_ms':1000*statistics.median(durations),'samples_ms':[1000*t for t in durations],
                     'statuses':[r.status for r in results[-1]],'objectives':[r.objective for r in results[-1]],
                     'persistent_reuses':sum(bool(r.raw.get('persistent_workspace')) for r in results[-1])}
    ref=answers['highs_cold_21']
    batch['all_objectives_agree']=all(check_pair(p,r,ref[i]) for rr in answers.values() for i,(p,r) in enumerate(zip([base]+problems,rr)))
    print('expression accumulation',flush=True)
    expression={}
    for n in (1000,5000):
        m=Model(); x=m.vars('x',n)
        item={'terms':n,'scope':'Expression construction only; variable creation and model compilation excluded.'}
        for name,fn in [('python_sum',lambda:sum(x)),('cython_quicksum',lambda:quicksum(x))]:
            durations,results=timed(fn,repeats)
            assert all(len(r.linear)==n and r.degree==1 for r in results)
            item[name]={'median_ms':1000*statistics.median(durations),'samples_ms':[1000*t for t in durations]}
        expression[str(n)]=item
    try:
        import Cython
        cython_version=Cython.__version__
    except ImportError: cython_version=None
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
    except ImportError: pools=[]
    cpu='unknown'
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),cpu)
    return {'schema':'mylpk.benchmark.v1','seed':20261007,'repeats':repeats,'warmups':1,
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'cython':cython_version,
            'platform':platform.platform(),'processor':cpu,'logical_cpus':os.cpu_count(),
            'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
            'threadpools':pools},
        'notes':['Wall-clock in-process microbenchmarks on one host, not an industrial benchmark suite.',
                 'Compiled LinearProblem inputs; transformation, copies, postsolve and checks ARE timed. Model expression building is separate.',
                 'Cold means no reuse of solver state; module import and process startup are not timed.',
                 'HiGHS is invoked through scipy.optimize; this is an API-path comparison, not a standalone C++ engine comparison.',
                 'Thread environment is recorded; internal HiGHS thread counts are not independently instrumented.',
                 'GLPK and Pyomo modeling performance are NOT measured because their optional Python packages are unavailable.',
                 'Each solver has a 10-second solve limit on the cold cases. All outcomes are retained.'],
        'cold_solves':rows,'objective_updates_40x100':batch,'expression_build':expression}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=Path('reports/benchmark.json'));args=parser.parse_args()
    if args.repeats<1: parser.error('repeats must be positive')
    report=run(args.repeats);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output)
    for row in report['cold_solves']:
        print(row['case'],round(row['mylpk']['median_ms'],3),round(row['highs']['median_ms'],3),row['objectives_agree'])
    print('updates',[(k,round(v['median_ms'],3)) for k,v in report['objective_updates_40x100'].items() if isinstance(v,dict)])
    if not all(row['objectives_agree'] for row in report['cold_solves']) or not report['objective_updates_40x100']['all_objectives_agree']:
        raise SystemExit('At least one benchmark failed correctness or did not establish optimality; inspect every recorded status.')

if __name__=='__main__': main()
