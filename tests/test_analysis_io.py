import json
import numpy as np
import pytest
from mylpk import *
from mylpk.cli import main
from mylpk.io import read_mps,read_json
from mylpk._core import Workspace


@pytest.mark.parametrize('sense',['min','max'])
@pytest.mark.parametrize('seed',range(10))
def test_dual_strong_duality(seed,sense):
    rng=np.random.default_rng(seed)
    A=rng.integers(-4,5,(4,5));x=np.ones(5)
    p=LinearProblem(rng.integers(-5,6,5),A,A@x-2,A@x+3,lower=-2,upper=4,sense=sense,offset=9)
    a=solve(p);b=solve(dual_problem(p))
    assert a.success and b.success and a.objective==pytest.approx(b.objective,abs=1e-6)


def test_exact_certificate_and_tampering():
    p=LinearProblem([5,4],[[3,2]],row_upper=18,upper=[7,8],sense='max',offset=3)
    r=solve(p); c=make_exact_certificate(p,r)
    assert c['certified'] and c['primal_objective']=='115/3'
    d=c['certificate'];d['x'][0]='1'
    assert not verify_exact_certificate(p,d)['certified']


def test_basis_sensitivity_inside_and_outside():
    p=LinearProblem([-3,-2],[[1,1],[1,0],[0,1]],row_upper=[4,2,3])
    r=solve(p);a=basis_sensitivity(p,r)
    assert a['coordinate_system']=='scaled_standard_form'
    assert a['rhs_delta'].shape[1]==2 and a['cost_delta'].shape[1]==2
    assert np.all(a['rhs_delta'][:,0]<=1e-7) and np.all(a['rhs_delta'][:,1]>=-1e-7)
    with pytest.raises(ValueError): basis_sensitivity(p.replace(c=[-2,-2]),r)


def test_iis():
    p=LinearProblem([1,0],[[1,0],[1,0],[0,1]],row_lower=[2,-np.inf,-np.inf],row_upper=[np.inf,1,10],upper=20)
    a=find_iis(p)
    assert a['irreducible'] and len(a['members'])==2
    assert {m['index'] for m in a['members']}=={0,1}
    b=find_iis(p,max_checks=1);assert not b['irreducible']
    assert find_iis(LinearProblem([1]))['status']=='not_proven_infeasible'


@pytest.mark.parametrize('suffix',['.json','.mps'])
@pytest.mark.parametrize('sense',['min','max'])
@pytest.mark.parametrize('integer',[False,True])
def test_interchange_roundtrip(tmp_path,suffix,sense,integer):
    p=LinearProblem([3,-2,0,1],[[1,2,0,0],[0,0,1,1]],[-2,1],[4,1],[-2,0,-np.inf,0],[3,2,np.inf,5],
        [1,1,0,0] if integer else 0,sense,7,['日本語','x y','x-z','zero'],['range','equal'])
    path=tmp_path/('model'+suffix);p.write(path);q=read_problem(path)
    a=solve(p,mip_rel_gap=0);b=solve(q,mip_rel_gap=0)
    assert a.success and b.success and a.objective==pytest.approx(b.objective)
    if suffix=='.json': assert q.var_names==p.var_names


def test_mps_rejects_unsupported_sections(tmp_path):
    path=tmp_path/'bad.mps';path.write_text('NAME P\nQSECTION\nENDATA\n')
    with pytest.raises(ValueError): read_mps(path)


def test_json_rejects_bad_sparse_indices(tmp_path):
    p=LinearProblem([1],[[1]],row_upper=2);f=tmp_path/'bad.json';p.write(f)
    d=json.loads(f.read_text());d['A']['indices']=[7];f.write_text(json.dumps(d))
    with pytest.raises(ValueError): read_json(f)


def test_cli_and_lp_writer(tmp_path,capsys):
    p=LinearProblem([-1],upper=3,offset=2);source=tmp_path/'m.json';out=tmp_path/'r.json';p.write(source)
    assert main(['solve',str(source),'--output',str(out)])==0
    assert json.loads(out.read_text())['objective']==-1
    assert main(['inspect',str(source)])==0
    assert main(['convert',str(source),str(tmp_path/'m.lp')])==0
    assert 'objconst = 1' in (tmp_path/'m.lp').read_text()
    assert main(['solvers'])==0
    assert main(['solve',str(source),'--solver','missing'])==4


@pytest.mark.parametrize('mutation',range(6))
def test_native_buffer_validation(mutation):
    data=np.array([1.]);indices=np.array([0],np.int32);ptr=np.array([0,1],np.int32);basis=np.array([0],np.int32)
    if mutation==0: indices=np.array([-1],np.int32)
    if mutation==1: ptr=np.array([0,2],np.int32)
    if mutation==2: data=np.array([np.nan])
    if mutation==3: basis=np.array([2],np.int32)
    if mutation==4: ptr=np.array([0.,1.])
    if mutation==5: indices=np.array([5],np.int32)
    with pytest.raises((ValueError,TypeError)): Workspace(data,indices,ptr,1,basis)


def test_backend_registration():
    from mylpk.problem import SolveResult
    register_solver('test_backend',lambda p,o,b:SolveResult('user_stop',solver='test_backend'))
    assert solve(LinearProblem([1]),solver='test_backend').status=='user_stop'
    with pytest.raises(ValueError): register_solver('mylpk',lambda p,o,b:None)


@pytest.mark.parametrize('record,expected',[('',1.),(' LO B x 0\n',np.inf),(' UP B x 4\n',4.)])
def test_mps_integer_default_conventions(tmp_path,record,expected):
    path=tmp_path/'integer.mps'
    path.write_text("NAME P\nROWS\n N OBJ\nCOLUMNS\n M0 'MARKER' 'INTORG'\n x OBJ -1\n M1 'MARKER' 'INTEND'\nBOUNDS\n"+record+'ENDATA\n')
    q=read_mps(path);assert q.integrality[0]==1 and q.upper[0]==expected
    if not record: assert read_mps(path,integer_default='unbounded').upper[0]==np.inf


def test_mps_ambiguous_negative_bound_rejected(tmp_path):
    path=tmp_path/'negative.mps'
    path.write_text('NAME P\nROWS\n N OBJ\nCOLUMNS\n x OBJ 1\nBOUNDS\n UP B x -1\nENDATA\n')
    with pytest.raises(ValueError): read_mps(path)
