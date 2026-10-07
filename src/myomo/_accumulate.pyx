# cython: language_level=3
"""One-pass dictionary accumulation for symbolic sparse polynomial sums."""
def accumulate(expressions):
    cdef dict linear = {}, quadratic = {}
    cdef object expr, k, v, old, owner = None, constant = 0.0
    for expr in expressions:
        if expr.owner is not None:
            if owner is not None and owner is not expr.owner:
                raise ValueError("Expressions from different models cannot be combined.")
            owner = expr.owner
        constant = constant + expr.constant
        for k, v in expr.linear.items():
            linear[k] = linear.get(k, 0.0) + v
        for k, v in expr.quadratic.items():
            quadratic[k] = quadratic.get(k, 0.0) + v
    return owner, linear, quadratic, constant
