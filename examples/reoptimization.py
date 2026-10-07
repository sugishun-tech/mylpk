"""Reuse the native LU/eta workspace for cost and RHS updates."""
import numpy as np
from mylpk import LinearProblem,LPSession,solve


def main():
    p=LinearProblem([-3,-2],[[1,1],[1,0],[0,1]],row_upper=[4,2,3])
    session=LPSession(p)
    print('initial:',session.last_result.objective)
    for c in ([-2,-4],[-5,-1],[-1,-1]):
        r=session.resolve(c=c).require_optimal()
        ref=solve(session.problem,solver='highs').require_optimal()
        assert np.isclose(r.objective,ref.objective)
        print(c,r.objective,'reused=',r.raw['persistent_workspace'])
    print('RHS update:',session.resolve(row_upper=[3,2,3]).require_optimal().objective)

if __name__=='__main__': main()
