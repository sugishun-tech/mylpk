import numpy as np
import pytest
from scipy import sparse
from mylpk import LinearProblem,solve,linprog,Options,kkt_report,LPSession,solve_many
from mylpk.diagnostics import verify_ray


def assert_same(p,**options):
    a=solve(p,**options); b=solve(p,solver='highs')
    assert a.status==b.status,(a.status,b.status,a.message,a.raw.get('kkt'))
    if a.success:
        assert a.objective==pytest.approx(b.objective,rel=2e-6,abs=2e-6)
        assert a.feasible
        assert a.raw.get('scaled_primal_violation',0)<=2e-7
        k=kkt_report(p,a)
        assert k['stationarity_max']<1e-5*(1+max(abs(p.c),default=0))
        assert k['duality_gap']<1e-5*(1+abs(a.objective))
    return a


@pytest.mark.parametrize('seed',range(160))
def test_random_general_lp(seed):
    rng=np.random.default_rng(seed)
    n=int(rng.integers(3,18)); m=int(rng.integers(5,20))
    x=rng.uniform(-2,2,n); lb=x-rng.uniform(.1,3,n); ub=x+rng.uniform(.1,3,n)
    if seed%3==0: lb[0]=-np.inf
    if seed%4==0: ub[1]=np.inf
    if seed%5==0: lb[2]=ub[2]=x[2]
    A=rng.normal(size=(m,n)); A[rng.random((m,n))<.3]=0
    low=A@x-rng.uniform(.1,3,m); high=A@x+rng.uniform(.1,3,m)
    if seed%2==0: low[0]=high[0]=(A@x)[0]
    low[1]=-np.inf; high[2]=np.inf
    free=np.flatnonzero(~np.isfinite(lb)|~np.isfinite(ub))
    if len(free):
        A=np.vstack([A,np.eye(n)[free]])
        low=np.r_[low,x[free]-5]; high=np.r_[high,x[free]+5]
    if seed%7==0:
        row_scale=10.**rng.uniform(-4,4,len(low))
        A*=row_scale[:,None]; low*=row_scale; high*=row_scale
    p=LinearProblem(rng.normal(size=n),sparse.csr_matrix(A),low,high,lb,ub,
                    sense='max' if seed%3==0 else 'min',offset=3.5)
    assert_same(p,refactor_interval=[1,4,48][seed%3],pivot_rule='bland' if seed%11==0 else 'dantzig',presolve=seed%6!=0)


@pytest.mark.parametrize('scaling',[True,False])
@pytest.mark.parametrize('presolve',[True,False])
def test_redundant_equalities(scaling,presolve):
    p=LinearProblem([1,2],[[1,1],[2,2],[0,0]],row_lower=[3,6,0],row_upper=[3,6,0],lower=-np.inf)
    # Without an additional bound this would be unbounded.
    p=p.replace(lower=[0,0])
    r=assert_same(p,scaling=scaling,presolve=presolve)
    assert r.objective==pytest.approx(3)


@pytest.mark.parametrize('p',[
    LinearProblem([1],lower=2,upper=1),
    LinearProblem([1],[[1],[1]],row_lower=[2,-np.inf],row_upper=[np.inf,1]),
    LinearProblem([1],[[0]],row_lower=1,row_upper=1),
    LinearProblem([1,1],[[1,1],[2,2]],row_lower=[1,3],row_upper=[1,3]),
    LinearProblem([0],[[1]],row_lower=2,upper=1),
])
def test_infeasible(p):
    assert solve(p).status=='infeasible'


@pytest.mark.parametrize('p',[
    LinearProblem([-1]),
    LinearProblem([1],lower=-np.inf,upper=5),
    LinearProblem([1,-1],[[1,-1]],row_upper=1,lower=-np.inf),
    LinearProblem([-1,-2],[[1,-1]],row_upper=1),
])
def test_unbounded_ray(p):
    r=solve(p)
    assert r.status=='unbounded'
    assert verify_ray(p,r.x,r.ray)['valid']


def test_beale_cycling_example():
    p=LinearProblem([-10,57,9,24],[[.5,-5.5,-2.5,9],[.5,-1.5,-.5,1],[1,0,0,0]],row_upper=[0,0,1])
    r=assert_same(p,max_iter=10000)
    assert r.objective==pytest.approx(-1)


def test_klee_minty():
    n=8; A=np.zeros((n,n))
    for i in range(n):
        for j in range(i): A[i,j]=2**(i-j+1)
        A[i,i]=1
    r=assert_same(LinearProblem(-2.**np.arange(n-1,-1,-1),A,row_upper=5.**np.arange(1,n+1)))
    assert r.iterations>0


def test_empty_and_fixed_models():
    r=solve(LinearProblem([],offset=7)); assert r.success and r.objective==7
    r=solve(LinearProblem([],np.empty((1,0)),row_lower=1)); assert r.status=='infeasible'
    p=LinearProblem([2,-3],[[1,1]],row_lower=5,row_upper=5,lower=[2,3],upper=[2,3],sense='max')
    r=assert_same(p); assert r.objective==-5


def test_linprog_api():
    r=linprog([-3,-2],[[1,1]],[4],bounds=[(0,2),(0,3)])
    assert r.objective==pytest.approx(-10)
    r=linprog([1],A_eq=[[1]],b_eq=[-3],bounds=(None,None))
    assert r.objective==-3


def test_warm_start_and_sessions():
    p=LinearProblem([-3,-2],[[1,1],[1,0],[0,1]],row_upper=[4,2,3])
    a=solve(p); b=solve(p.replace(c=[-2,-4]),basis=a.basis)
    assert b.success and b.raw['warm_start_used']
    s=LPSession(p)
    rng=np.random.default_rng(40)
    for _ in range(25):
        c=rng.normal(size=2)
        r=s.resolve(c=c)
        ref=solve(p.replace(c=c),solver='highs')
        assert r.success and r.objective==pytest.approx(ref.objective)
    for rhs in ([1,2,3],[8,2,3],[-1,2,3],[4,2,3]):
        r=s.resolve(row_upper=rhs)
        ref=solve(s.problem,solver='highs')
        assert r.status==ref.status
        if r.success: assert r.objective==pytest.approx(ref.objective)
    with pytest.raises(ValueError): s.resolve(row_upper=[np.inf,2,3])


def test_concurrent_independent_solves():
    ps=[LinearProblem([-i,-1],[[1,1]],row_upper=3,upper=2) for i in range(1,9)]
    rs=solve_many(ps,workers=4)
    for p,r in zip(ps,rs): assert r.objective==pytest.approx(solve(p).objective)


def test_memory_guard_and_limits():
    p=LinearProblem(-np.ones(20),np.eye(20),row_upper=1)
    with pytest.raises(MemoryError): solve(p,max_basis_mb=.0001)
    assert solve(p,max_iter=1).status=='iteration_limit'
    assert solve(p,time_limit=1e-12).status=='time_limit'


@pytest.mark.parametrize('bad',[
    {'c':[np.nan]}, {'c':[1],'A':[[np.inf]]}, {'c':[1],'integrality':[.5]},
    {'c':[1],'lower':np.inf}, {'c':[1],'upper':-np.inf},
    {'c':[1],'row_upper':[1,2]}, {'c':[1,2],'var_names':['a','a']},
    {'c':[1],'integrality':2,'upper':np.inf}, {'c':[1],'integrality':2,'lower':-1,'upper':2}
])
def test_bad_problem_inputs(bad):
    with pytest.raises((ValueError,TypeError)): LinearProblem(**bad)


@pytest.mark.parametrize('bad',[{'dual_tol':0},{'pivot_tol':np.nan},{'node_limit':0},{'max_iter':1.2},
    {'refactor_interval':1025},{'mip_rel_gap':-1},{'integrality_tol':.5},{'time_limit':0},{'pivot_rule':'random'}])
def test_bad_options(bad):
    with pytest.raises(ValueError): Options(**bad)


def test_overflow_is_not_reported_as_an_optimum():
    p=LinearProblem([1e308],lower=2,upper=2)
    r=solve(p)
    assert r.status=='numerical_error' and not r.success


def test_corrupt_scipy_sparse_input_is_rejected():
    A=sparse.csc_matrix(([1.],[5],[0,1]),shape=(1,1))
    with pytest.raises(ValueError): LinearProblem([1],A)
