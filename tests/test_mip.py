import itertools
import numpy as np
import pytest
from mylpk import LinearProblem,solve,milp
from mylpk.diagnostics import violations


@pytest.mark.parametrize('seed',range(80))
def test_binary_mip_against_highs_and_enumeration(seed):
    rng=np.random.default_rng(seed); n=8; m=3
    A=rng.integers(1,20,(m,n)).astype(float)
    b=A.sum(axis=1)*rng.uniform(.2,.75,m)
    c=-rng.integers(1,30,n).astype(float)
    p=LinearProblem(c,A,row_upper=b,upper=1,integrality=1,offset=10.)
    r=solve(p,mip_rel_gap=0,cuts=seed%2==0)
    ref=solve(p,solver='highs',mip_rel_gap=0)
    assert r.status=='optimal',(seed,r)
    assert r.objective==pytest.approx(ref.objective,abs=1e-6)
    points=np.array(list(itertools.product([0,1],repeat=n)),float)
    good=np.all(points@A.T<=b+1e-10,axis=1)
    exact=float(np.min(points[good]@c)+10)
    assert r.objective==pytest.approx(exact)
    assert r.feasible and r.gap==0
    assert r.dual is None


@pytest.mark.parametrize('seed',range(40))
def test_general_mixed_integer(seed):
    rng=np.random.default_rng(seed+400)
    n=7; x=np.r_[rng.integers(-2,4,4),rng.uniform(-2,3,3)]
    A=rng.normal(size=(5,n)); hi=A@x+rng.uniform(.2,4,5)
    p=LinearProblem(rng.normal(size=n),A,row_upper=hi,lower=-2,upper=3,integrality=[1,1,1,1,0,0,0],sense='max' if seed%2 else 'min')
    r=solve(p,mip_rel_gap=0,node_limit=20000)
    ref=solve(p,solver='highs',mip_rel_gap=0)
    assert r.status==ref.status,(seed,r,ref)
    if r.success:
        assert r.objective==pytest.approx(ref.objective,rel=2e-6,abs=2e-6)
        assert violations(p,r.x)['scaled_violation']<1e-7


@pytest.mark.parametrize('itype,lower,upper,cap,expected',[(2,2,4,1.5,0),(2,2,4,3.5,3.5),(3,2.2,4.8,3.5,3),(3,2.2,4.8,2.5,0)])
def test_semi_domains(itype,lower,upper,cap,expected):
    p=LinearProblem([-1],[[1]],row_upper=cap,lower=lower,upper=upper,integrality=itype)
    r=solve(p,mip_rel_gap=0)
    assert r.success and r.x[0]==pytest.approx(expected)
    assert r.objective==pytest.approx(solve(p,solver='highs').objective)


def test_integer_infeasible_bounds():
    p=LinearProblem([1],lower=.1,upper=.9,integrality=1)
    assert solve(p).status=='infeasible'


def test_relaxation_unbounded_is_not_integer_proof():
    p=LinearProblem([0,-1],[[1,0]],row_lower=.5,row_upper=.5,integrality=[1,0])
    assert solve(p).status=='relaxation_unbounded'


def test_limits_callback_and_incumbent():
    p=LinearProblem([-3,-4,-5],[[2,3,4]],row_upper=4.5,upper=1,integrality=1)
    r=solve(p,node_limit=1,cuts=False,heuristic=False,mip_rel_gap=0)
    assert r.status=='node_limit' and r.nodes==1
    assert r.bound<=solve(p,mip_rel_gap=0).objective+1e-7
    events=[]
    r=solve(p,cuts=False,heuristic=False,callback=lambda e:events.append(e) or False)
    assert events and r.status=='user_stop'
    r=solve(p,x0=[0,0,1],node_limit=1,cuts=False,heuristic=False)
    assert r.feasible and r.objective<=-5
    with pytest.raises(ValueError): solve(p,x0=[1,1,1])


def test_cover_complement_validity():
    p=LinearProblem([-5,4,-3],[[5,-4,3]],row_upper=2.5,upper=1,integrality=1)
    a=solve(p,cuts=True,mip_rel_gap=0); b=solve(p,cuts=False,mip_rel_gap=0)
    assert a.objective==pytest.approx(b.objective)


def test_milp_function():
    r=milp([-2,-3],A=[[1,1]],row_upper=2.5,upper=2)
    assert r.status=='optimal' and r.objective==-6
