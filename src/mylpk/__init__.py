"""mylpk: native LP/MILP optimization, with optional explicit external backends."""
from .problem import LinearProblem, SolveResult, Options, Basis
from .lp import solve_lp
from .diagnostics import violations, kkt_report
__version__ = "0.1.0"
from .solvers import solve, available_solvers, register_solver
from .api import linprog, milp
from .io import read_problem, write_problem
from .analysis import dual_problem, basis_sensitivity, find_iis
from .exact import make_exact_certificate, verify_exact_certificate
from .session import LPSession, solve_many
