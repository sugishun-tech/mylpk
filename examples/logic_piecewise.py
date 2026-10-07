"""Binary indicator, binary-continuous product, exact PWL and absolute value."""
from myomo import Model


def main():
    m=Model('logic_and_pwl')
    on=m.var('on',kind='binary')
    amount=m.var('amount',ub=3)
    m.indicator(on,amount==0,active=0)
    m.indicator(on,amount>=1)
    revenue=m.piecewise(amount,[0,1,3],[0,4,6])
    active_amount=m.product(on,amount)
    distance=m.absolute(amount-2,exact=True)
    m.maximize(revenue-0.5*active_amount-distance-on)
    r=m.solve(mip_rel_gap=0).require_optimal()
    print('objective',r.objective,'amount',r[amount],'on',r[on])
    assert abs(r[amount]-2)<1e-7 and abs(r.objective-3)<1e-7

if __name__=='__main__': main()
