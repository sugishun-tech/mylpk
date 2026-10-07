"""Small integer production model; --solver glpk requires swiglpk."""
import argparse
from myomo import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--solver',default='mylpk');args=parser.parse_args()
    m=Model('production')
    capacity=m.param('capacity',14)
    x=m.var('x',kind='integer')
    y=m.var('y',kind='integer')
    m+=2*x+y<=capacity
    m+=x+2*y<=14
    m.maximize(3*x+2*y)
    r=m.solve(solver=args.solver,mip_rel_gap=0).require_optimal()
    print(f'{r.status}: x={r[x]:g}, y={r[y]:g}, objective={r.objective:g}')
    assert r.objective==23
    capacity.value=18
    r2=m.solve(solver=args.solver,mip_rel_gap=0).require_optimal()
    print('updated capacity:',r2.objective, r2[x],r2[y])

if __name__=='__main__': main()
