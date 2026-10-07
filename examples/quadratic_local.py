"""Continuous QCQP through explicitly selected local SLSQP, not native mylpk."""
import numpy as np
from myomo import Model


def main():
    m=Model('unit_disk');x=m.var('x',lb=-2,ub=2);y=m.var('y',lb=-2,ub=2)
    m+=x*x+y*y<=1
    m.maximize(x+y)
    r=m.solve(solver='scipy')
    print(r.status,r.objective,r[x],r[y])
    assert r.status=='locally_optimal' and r.feasible
    assert np.isclose(r.objective,np.sqrt(2))
    assert not r.success and not r.raw['global_optimality_certified']

if __name__=='__main__': main()
