"""LP duality, exact certificate checking, sensitivity and an IIS."""
import numpy as np
from mylpk import LinearProblem,solve,dual_problem,basis_sensitivity,find_iis,make_exact_certificate,kkt_report


def main():
    p=LinearProblem([5,4],[[3,2]],row_upper=18,upper=[7,8],sense='max',offset=3)
    r=solve(p).require_optimal()
    dual=solve(dual_problem(p)).require_optimal()
    assert abs(dual.objective-r.objective)<1e-8
    proof=make_exact_certificate(p,r)
    print('exact:',proof['certified'],proof['primal_objective'])
    assert proof['certified'] and proof['primal_objective']=='115/3'
    print('KKT gap:',kkt_report(p,r)['duality_gap'])
    sensitivity=basis_sensitivity(p,r)
    print('sensitivity coordinates:',sensitivity['coordinate_system'])
    print('one-RHS-coordinate delta ranges:',sensitivity['rhs_delta'])
    bad=LinearProblem([1],[[1],[1]],row_lower=[2,-np.inf],row_upper=[np.inf,1],row_names=['at_least_2','at_most_1'])
    iis=find_iis(bad)
    assert iis['irreducible']
    print('IIS:',iis['members'])

if __name__=='__main__': main()
