"""Lexicographic objectives, binary pattern enumeration and parameter scenarios."""
from myomo import Model


def main():
    m=Model('lexicographic');x=m.var('x',ub=5);y=m.var('y',ub=5);m+=x+y<=5
    r=m.solve_lexicographic([(x+y,'max'),(x,'min')],atol=0)
    r.require_optimal();print('lexicographic:',r[x],r[y],r.raw['lexicographic_values'])
    assert r[x]==0 and r[y]==5
    # These results are snapshots: temporary constraints/objectives are restored.
    n=Model();b=n.vars('b',3,kind='binary');n+=b.sum()>=1;n.minimize(b.sum())
    results=n.enumerate_solutions(10)
    print('binary patterns:',[r[b] for r in results]);assert len(results)==7
    s=Model();capacity=s.param('capacity',2);z=s.var('z',ub=capacity);s.maximize(z)
    results=s.solve_scenarios({'small':{'capacity':3},'large':{'capacity':8}})
    print('scenarios:',{k:r.objective for k,r in results.items()});assert capacity.value==2

if __name__=='__main__': main()
