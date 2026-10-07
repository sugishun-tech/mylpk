"""Named Cartesian indices, summation patterns and dictionary costs."""
from myomo import Model


def main():
    m=Model('transport')
    x=m.vars('ship',['a','b'],['c','d'],ub=10)
    m+=x.sum('a','*')<=3
    m+=x.sum('b','*')<=4
    m+=x.sum('*','c')>=2
    m+=x.sum('*','d')>=4
    m.minimize(x.dot({('a','c'):1,('a','d'):3,('b','c'):2,('b','d'):1}))
    r=m.solve().require_optimal()
    print(r.objective,r[x]);assert r.objective==6

if __name__=='__main__': main()
