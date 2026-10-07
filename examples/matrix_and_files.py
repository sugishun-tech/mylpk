"""Bulk sparse modeling and portable matrix-file round trips in a temporary folder."""
from pathlib import Path
import tempfile
from scipy import sparse
from myomo import Model
from mylpk import solve,read_problem


def main():
    m=Model('matrix');x=m.vars('x',3)
    m.add_matrix(sparse.csr_matrix([[1,1,1],[1,0,0]]),x,ub=[5,2])
    m.maximize(x.dot([4,2,1]))
    r=m.solve().require_optimal();assert r.objective==14
    with tempfile.TemporaryDirectory() as directory:
        for suffix in ('.json','.mps'):
            path=Path(directory)/('model'+suffix);m.write(path)
            check=solve(read_problem(path)).require_optimal()
            assert check.objective==r.objective
            print(suffix,'round trip:',check.objective)
        m.write(Path(directory)/'model.lp')  # LP import additionally requires highspy.

if __name__=='__main__': main()
