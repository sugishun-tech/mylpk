import numpy as np
import pytest
from myomo import Model,quicksum,dot,between


def test_production_model():
    m=Model(); x=m.var('x',kind='integer'); y=m.var('y',kind='integer')
    m+=2*x+y<=14; m+=x+2*y<=14; m.maximize(3*x+2*y)
    r=m.solve(mip_rel_gap=0)
    assert r.success and r.objective==23 and r[x]==5 and y.value==4


def test_indexed_transport():
    m=Model(); x=m.vars('x',['a','b'],['c','d'],ub=10)
    m+=x.sum('a','*')<=3; m+=x.sum('b','*')<=4
    m+=x.sum('*','c')>=2; m+=x.sum('*','d')>=4
    m.minimize(x.dot({('a','c'):1,('a','d'):3,('b','c'):2,('b','d'):1}))
    r=m.solve(); assert r.success and r.objective==6
    assert r[x][('a','c')]==2


def test_parameters_in_coefficients_and_bounds():
    m=Model(); capacity=m.param('capacity',5); cost=m.param('cost',2)
    x=m.var('x',ub=capacity); m.minimize(-cost*x)
    assert m.solve().objective==-10
    capacity.value=8
    with pytest.raises(RuntimeError): _=x.value
    assert m.solve().objective==-16
    cost.value=-3
    assert m.solve().objective==0


def test_matrix_fast_path():
    m=Model(); x=m.vars('x',3)
    block=m.add_matrix([[1,1,1],[1,0,0]],x,ub=[5,2])
    m.maximize(x.dot([4,2,1])); r=m.solve()
    assert r.objective==14 and block.dual.shape==(2,)


def test_expression_algebra():
    m=Model(); x=m.var('x'); y=m.var('y'); p=m.param('p',2)
    expr=(x+3*y+2)*(2*x-y+4)-x*x+(p*x+1)/2
    for values in ([0,0],[2,3],[-1,4]):
        a,b=values
        assert expr.evaluate(np.array(values))==pytest.approx((a+3*b+2)*(2*a-b+4)-a*a+(2*a+1)/2)
    p.value=3
    assert expr.evaluate(np.array([1.,2.]))==pytest.approx((1+6+2)*(2-2+4)-1+(3+1)/2)


@pytest.mark.parametrize('what',['chain','bool','strict','nonlinear','division','cross_model','bool_add','duplicate'])
def test_actionable_errors(what):
    m=Model(); x=m.var('x')
    with pytest.raises((ValueError,TypeError)):
        if what=='chain': _=0<=x<=1
        elif what=='bool': bool(x)
        elif what=='strict': _=x<1
        elif what=='nonlinear': _=x**3
        elif what=='division': _=1/(x+1)
        elif what=='cross_model': _=x+Model().var('y')
        elif what=='bool_add': m.add(True)
        elif what=='duplicate': m.var('x')


def test_indicator_recomputes_big_m_after_parameter_change():
    m=Model(); bound=m.param('upper',3); x=m.var('x',ub=bound); b=m.var('b',kind='binary')
    m.indicator(b,x<=1); m+=b==0; m.maximize(x)
    assert m.solve().objective==3
    bound.value=20
    assert m.solve().objective==20
    n=Model(); u=n.var('u'); z=n.var('z',kind='binary'); n.indicator(z,u<=1); n.minimize(u)
    with pytest.raises(ValueError): n.compile()


@pytest.mark.parametrize('active',[0,1])
@pytest.mark.parametrize('bvalue',[0,1])
def test_indicator(active,bvalue):
    m=Model(); x=m.var('x',lb=-5,ub=5); b=m.var('b',kind='binary');m+=b==bvalue
    m.indicator(b,between(-2,x,2),active=active);m.maximize(x)
    assert m.solve().objective==pytest.approx(2 if active==bvalue else 5)


@pytest.mark.parametrize('exact',[False,True])
@pytest.mark.parametrize('value',[-3.,0.,2.])
def test_absolute(exact,value):
    m=Model(); x=m.var('x',lb=-5,ub=5); t=m.absolute(x,exact=exact);m+=x==value;m.minimize(t)
    assert m.solve().objective==pytest.approx(abs(value))


def test_abs_epigraph_is_not_automatically_an_equality():
    m=Model(); x=m.var('x',lb=-5,ub=5); t=m.absolute(x,exact=True);m+=x==-2;m.maximize(t)
    assert m.solve().objective==2


@pytest.mark.parametrize('bvalue',[0,1])
@pytest.mark.parametrize('xvalue',[-3,0,4])
def test_binary_product(bvalue,xvalue):
    m=Model(); x=m.var('x',lb=-5,ub=5);b=m.var('b',kind='binary');t=m.product(b,x)
    m+=b==bvalue;m+=x==xvalue;m.minimize(t)
    assert m.solve().objective==pytest.approx(bvalue*xvalue)


@pytest.mark.parametrize('xvalue,expected',[(0,0),(.5,2),(1,4),(2,3),(3,2)])
def test_piecewise_exact(xvalue,expected):
    m=Model();x=m.var('x',ub=3); y=m.piecewise(x,[0,1,3],[0,4,2]);m+=x==xvalue;m.maximize(y)
    r=m.solve();assert r.success and y.value==pytest.approx(expected)


def test_all_different():
    m=Model(); x=m.vars('x',3,lb=1,ub=3,kind='integer');m.all_different(x);m.minimize(x.sum())
    r=m.solve(mip_rel_gap=0);assert r.success and sorted(round(v.value) for v in x)==[1,2,3]


@pytest.mark.parametrize('values',[(0,0),(0,1),(1,0),(1,1)])
def test_logic(values):
    m=Model();a=m.var('a',kind='binary');b=m.var('b',kind='binary')
    c=m.logical_and([a,b]);d=m.logical_or([a,b]);e=m.logical_xor(a,b)
    m+=a==values[0];m+=b==values[1];m.minimize(c+d+e)
    r=m.solve();assert r.success
    assert round(c.value)==(values[0]&values[1]);assert round(d.value)==(values[0]|values[1]);assert round(e.value)==(values[0]^values[1])


def test_soft_max_min():
    m=Model();x=m.var('x',ub=2);cost=m.soft(x>=5,penalty=10);m.minimize(cost)
    assert m.solve().objective==30
    n=Model();x=n.var('x',lb=-10,ub=10);t=n.max_epigraph([x+1,-x+3]);n.minimize(t)
    assert n.solve().objective==pytest.approx(2)
    n=Model();x=n.var('x',lb=-10,ub=10);t=n.min_hypograph([x+1,-x+3]);n.maximize(t)
    assert n.solve().objective==pytest.approx(2)


def test_sos_reformulations():
    m=Model();x=m.vars('x',4,ub=3);m.sos1(x);m.maximize(x.sum())
    assert m.solve().objective==pytest.approx(3)
    n=Model();x=n.vars('x',4,ub=3);n.sos2(x);n.maximize(x.sum())
    assert n.solve().objective==pytest.approx(6)


def test_lexicographic_enumeration_scenarios_clone():
    m=Model();x=m.var('x',ub=5);y=m.var('y',ub=5);m+=x+y<=5
    r=m.solve_lexicographic([(x+y,'max'),(x,'min')],atol=0)
    assert r.success and r[x]==pytest.approx(0) and r[y]==pytest.approx(5)
    n=Model();a=n.var('a',kind='binary');b=n.var('b',kind='binary');n+=a+b>=1;n.minimize(a+b)
    rs=n.enumerate_solutions(10)
    assert len(rs)==3 and len({(round(r[a]),round(r[b])) for r in rs})==3
    n=Model();p=n.param('p',2);x=n.var('x',ub=p);n.maximize(x)
    rs=n.solve_scenarios({'a':{'p':3},'b':{'p':6}})
    assert rs['a'].objective==3 and rs['b'].objective==6 and p.value==2
    clone=n.clone();p.value=10
    assert clone.solve().objective==2 and n.solve().objective==10


def test_quadratic_and_qcqp_are_explicit_local_results():
    m=Model();x=m.var('x',lb=-5,ub=5);y=m.var('y',lb=-5,ub=5);m.minimize((x-2)**2+3*(y+1)**2)
    with pytest.raises(ValueError): m.solve()
    r=m.solve(solver='scipy');assert r.status=='locally_optimal' and r.feasible
    assert r.objective==pytest.approx(0,abs=1e-7)
    assert r.raw['global_optimality_certified'] is False
    n=Model();x=n.var('x',lb=-2,ub=2);y=n.var('y',lb=-2,ub=2);n+=x*x+y*y<=1;n.maximize(x+y)
    r=n.solve(solver='scipy');assert r.status=='locally_optimal' and r.objective==pytest.approx(np.sqrt(2),abs=1e-6)


@pytest.mark.parametrize('field',['lb','ub','kind','lower','upper'])
def test_mutations_invalidate_current_solution(field):
    m=Model();x=m.var('x',ub=3);row=m.add(x<=2);m.maximize(x);m.solve()
    if field=='lb': x.lb=1
    elif field=='ub': x.ub=1
    elif field=='kind': x.kind='integer'
    elif field=='lower': row.lower=-1
    else: row.upper=-1  # expression is x-2, so this means x <= 1
    assert m.last_result is None
    with pytest.raises(RuntimeError): _=x.value
    assert m.solve().success


def test_mutable_bounds_reject_foreign_parameters():
    m=Model();x=m.var('x',ub=3);row=m.add(x<=2);foreign=Model().param('p',1)
    with pytest.raises(ValueError): x.ub=foreign
    with pytest.raises(ValueError): row.lower=foreign
    with pytest.raises(ValueError): x.fix(foreign)


def test_workflow_restoration_does_not_leave_a_false_current_optimum():
    m=Model();x=m.var('x',ub=3);m.maximize(x)
    r=m.solve_lexicographic([(x,'min')]);assert r[x]==0 and m.last_result is None
    n=Model();b=n.var('b',kind='binary');n.minimize(b)
    rs=n.enumerate_solutions(2);assert len(rs)==2 and n.last_result is None


@pytest.mark.parametrize('feature',['indicator','soft'])
def test_sign_restricted_parameters_are_revalidated(feature):
    m=Model();x=m.var('x',ub=3);p=m.param('p',3)
    if feature=='indicator':
        b=m.var('b',kind='binary');m.indicator(b,x<=1,M=p);m.minimize(x)
    else: m.minimize(m.soft(x>=5,penalty=p))
    p.value=-1
    with pytest.raises(ValueError): m.compile()
