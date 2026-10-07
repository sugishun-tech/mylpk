import importlib.util
import numpy as np
import pytest
from mylpk import LinearProblem,solve,read_problem


@pytest.mark.skipif(importlib.util.find_spec('swiglpk') is None,reason='swiglpk/GLPK is not installed; no live GLPK validation')
@pytest.mark.parametrize('integer',[False,True])
@pytest.mark.parametrize('sense',['min','max'])
@pytest.mark.parametrize('seed',range(5))
def test_glpk_live(seed,sense,integer):
    rng=np.random.default_rng(seed);A=rng.integers(1,10,(3,6));b=A.sum(axis=1)*.4
    p=LinearProblem(rng.integers(-10,11,6),A,row_upper=b,upper=1,integrality=int(integer),sense=sense,offset=2)
    a=solve(p,solver='glpk',mip_rel_gap=0);b=solve(p,mip_rel_gap=0)
    assert a.status==b.status
    if a.success: assert a.objective==pytest.approx(b.objective,abs=1e-6)


@pytest.mark.skipif(importlib.util.find_spec('highspy') is None,reason='highspy is not installed; optional LP reader not live-validated')
def test_lp_import_live(tmp_path):
    p=LinearProblem([-3,-2],[[1,1]],row_upper=4,upper=[2,3],offset=5)
    path=tmp_path/'p.lp';p.write(path)
    q=read_problem(path);assert solve(q).objective==pytest.approx(solve(p).objective)
