"""Command-line matrix-file optimizer. Never executes a model file as Python."""
import argparse
import json
import sys
from pathlib import Path
from .io import read_problem,write_problem,json_safe
from .solvers import solve,available_solvers


def main(argv=None):
    parser=argparse.ArgumentParser(prog='mylpk',description='Solve, inspect or convert linear optimization models.')
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('solve'); s.add_argument('input'); s.add_argument('--solver',default='mylpk')
    s.add_argument('--output'); s.add_argument('--time-limit',type=float,default=float('inf'))
    s.add_argument('--node-limit',type=int,default=100000); s.add_argument('--mip-gap',type=float,default=1e-6)
    i=sub.add_parser('inspect'); i.add_argument('input')
    c=sub.add_parser('convert'); c.add_argument('input'); c.add_argument('output')
    sub.add_parser('solvers')
    args=parser.parse_args(argv)
    try:
        if args.command=='solvers': print(json.dumps(available_solvers(),indent=2)); return 0
        p=read_problem(args.input)
        if args.command=='inspect': print(json.dumps(p.stats(),indent=2)); return 0
        if args.command=='convert': write_problem(p,args.output); return 0
        r=solve(p,solver=args.solver,time_limit=args.time_limit,node_limit=args.node_limit,mip_rel_gap=args.mip_gap)
        text=json.dumps(json_safe(r.to_dict()),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output: Path(args.output).write_text(text,encoding='utf-8')
        else: print(text,end='')
        return 0 if r.status=='optimal' else (2 if r.status in ('infeasible','unbounded') else 3)
    except (ValueError,TypeError,ImportError,RuntimeError,OSError,MemoryError) as e:
        print(f'mylpk: {e}',file=sys.stderr); return 4

if __name__=='__main__': raise SystemExit(main())
