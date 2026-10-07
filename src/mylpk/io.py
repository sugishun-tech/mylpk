"""Strict JSON, free-format MPS, and LP export.

JSON preserves user names and domains. Text exchange uses stable xN/rN symbols.
Semi variables are explicitly reformulated when exporting LP/MPS.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .problem import LinearProblem
from .mip import expand_semivariables


def json_safe(value):
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)):
        if np.isnan(value): return None
        if np.isposinf(value): return 'inf'
        if np.isneginf(value): return '-inf'
        return float(value)
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value


def write_json(p,path):
    A=p.A
    d={'schema':'mylpk.linear.v1','name':p.name,'sense':p.sense,'offset':p.offset,
        'c':p.c,'A':{'shape':list(A.shape),'data':A.data,'indices':A.indices,'indptr':A.indptr},
        'row_lower':p.row_lower,'row_upper':p.row_upper,'lower':p.lower,'upper':p.upper,
        'integrality':p.integrality,'var_names':p.var_names,'row_names':p.row_names}
    Path(path).write_text(json.dumps(json_safe(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return Path(path)


def read_json(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if d.get('schema')!='mylpk.linear.v1': raise ValueError('Unsupported JSON problem schema.')
    a=d['A']; shape=tuple(a['shape'])
    if len(shape)!=2 or any(not isinstance(k,int) or k<0 for k in shape): raise ValueError('Invalid sparse shape.')
    data=np.array(a['data'],dtype=float); ids=np.array(a['indices']); ptr=np.array(a['indptr'])
    if ids.ndim!=1 or ptr.ndim!=1 or data.ndim!=1 or len(data)!=len(ids) or len(ptr)!=shape[1]+1:
        raise ValueError('Invalid sparse arrays.')
    if not np.issubdtype(ids.dtype,np.integer) and len(ids): raise ValueError('Sparse indices must be integers.')
    if not np.issubdtype(ptr.dtype,np.integer): raise ValueError('Sparse pointers must be integers.')
    if ptr[0]!=0 or ptr[-1]!=len(data) or np.any(np.diff(ptr)<0) or np.any(ids<0) or np.any(ids>=shape[0]): raise ValueError('Invalid CSC structure.')
    A=sparse.csc_matrix((data,ids.astype(np.int64),ptr.astype(np.int64)),shape=shape)
    keys=('row_lower','row_upper','lower','upper','integrality','sense','offset','var_names','row_names','name')
    return LinearProblem(d['c'],A,**{k:d[k] for k in keys})


def write_lp(original,path):
    p,_=expand_semivariables(original)
    if not p.n: raise ValueError('Text LP export requires at least one variable; JSON supports empty models.')
    if np.any(p.lower>p.upper) or np.any(p.row_lower>p.row_upper): raise ValueError('Cannot export contradictory bound pairs to LP.')
    def expression(items):
        terms=[]
        for j,a in items:
            if a: terms.append(f"{'+' if a>=0 else '-'} {abs(float(a)):.17g} x{j}")
        return ' '.join(terms) or '0 x0'
    lines=['\\ mylpk LP export. Original names are preserved only in JSON.', 'Minimize' if p.sense=='min' else 'Maximize']
    obj=expression(enumerate(p.c))
    if p.offset: obj+=f" {'+' if p.offset>=0 else '-'} {abs(p.offset):.17g} objconst"
    lines+=[' obj: '+obj,'Subject To']
    A=p.A.tocsr()
    for i in range(p.m):
        sl=slice(A.indptr[i],A.indptr[i+1]); e=expression(zip(A.indices[sl],A.data[sl]))
        lo,hi=p.row_lower[i],p.row_upper[i]
        if np.isfinite(lo) and lo==hi: lines.append(f' r{i}: {e} = {lo:.17g}')
        else:
            if np.isfinite(lo): lines.append(f' r{i}_lo: {e} >= {lo:.17g}')
            if np.isfinite(hi): lines.append(f' r{i}_up: {e} <= {hi:.17g}')
    if len(lines)==4: lines.append(' empty_row: 0 x0 <= 0')
    lines.append('Bounds')
    for j,(lo,hi) in enumerate(zip(p.lower,p.upper)):
        if lo==hi: lines.append(f' x{j} = {lo:.17g}')
        elif np.isneginf(lo) and np.isposinf(hi): lines.append(f' x{j} free')
        elif np.isneginf(lo): lines.append(f' -inf <= x{j} <= {hi:.17g}')
        elif np.isposinf(hi): lines.append(f' {lo:.17g} <= x{j}')
        else: lines.append(f' {lo:.17g} <= x{j} <= {hi:.17g}')
    if p.offset: lines.append(' objconst = 1')
    integer=np.flatnonzero(p.integrality==1)
    if len(integer):
        lines.append('Generals')
        lines.extend(f' x{j}' for j in integer)
    lines.append('End')
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8'); return Path(path)


def write_mps(original,path):
    p,_=expand_semivariables(original)
    if not p.n: raise ValueError('Text MPS export requires at least one variable; JSON supports empty models.')
    if np.any(p.lower>p.upper) or np.any(p.row_lower>p.row_upper): raise ValueError('Cannot export contradictory bound pairs to MPS.')
    lines=['NAME          MYLPK','OBJSENSE',' MIN' if p.sense=='min' else ' MAX','ROWS',' N  OBJ']
    rhs=[]; ranges=[]
    for i,(lo,hi) in enumerate(zip(p.row_lower,p.row_upper)):
        if np.isfinite(lo) and lo==hi: kind='E'; rhs.append((i,hi))
        elif np.isfinite(hi):
            kind='L'; rhs.append((i,hi))
            if np.isfinite(lo): ranges.append((i,hi-lo))
        elif np.isfinite(lo): kind='G'; rhs.append((i,lo))
        else: kind='N'
        lines.append(f' {kind}  r{i}')
    lines.append('COLUMNS'); in_integer=False; mark=0
    A=p.A
    for j in range(p.n):
        integer=bool(p.integrality[j])
        if integer!=in_integer:
            lines.append(f"    MARK{mark}  'MARKER'  '{'INTORG' if integer else 'INTEND'}'"); mark+=1; in_integer=integer
        lines.append(f'    x{j}  OBJ  {p.c[j]:.17g}')
        for k in range(A.indptr[j],A.indptr[j+1]): lines.append(f'    x{j}  r{A.indices[k]}  {A.data[k]:.17g}')
    if in_integer: lines.append(f"    MARK{mark}  'MARKER'  'INTEND'")
    lines.append('RHS')
    if p.offset: lines.append(f'    RHS1  OBJ  {-p.offset:.17g}')
    lines.extend(f'    RHS1  r{i}  {v:.17g}' for i,v in rhs)
    if ranges:
        lines.append('RANGES'); lines.extend(f'    RNG1  r{i}  {v:.17g}' for i,v in ranges)
    lines.append('BOUNDS')
    for j,(lo,hi) in enumerate(zip(p.lower,p.upper)):
        if p.integrality[j] and lo==0 and hi==1:
            lines.append(f' BV BND1  x{j}'); continue
        if lo==hi: lines.append(f' FX BND1  x{j}  {lo:.17g}'); continue
        if np.isneginf(lo) and np.isposinf(hi): lines.append(f' FR BND1  x{j}'); continue
        if np.isneginf(lo): lines.append(f' MI BND1  x{j}')
        else: lines.append(f' LO BND1  x{j}  {lo:.17g}')
        if np.isfinite(hi): lines.append(f' UP BND1  x{j}  {hi:.17g}')
        else: lines.append(f' PL BND1  x{j}')
    lines.append('ENDATA')
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8'); return Path(path)


def read_mps(path,*,rhs_name=None,ranges_name=None,bounds_name=None,integer_default="binary"):
    """Read linear free-format MPS (also whitespace-tokenizable fixed MPS).

    Selects the first RHS/range/bound set unless explicitly named. Rejects
    quadratic, SOS, indicator and other unsupported sections instead of dropping them.
    """
    if integer_default not in ('binary','unbounded'): raise ValueError('integer_default must be binary/unbounded.')
    section=None; sense='min'; name='problem'; obj=None; objname=None
    rows={}; columns={}; integer=set(); rhs_sets={}; range_sets={}; bound_sets={}; inside=False
    last_col=None
    headers={'NAME','OBJSENSE','OBJNAME','ROWS','COLUMNS','RHS','RANGES','BOUNDS','ENDATA'}
    def number(v): return float(v.replace('D','E').replace('d','e'))
    for lineno,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip() or line.lstrip().startswith('*'): continue
        tokens=line.split()
        if not line[0].isspace():
            head=tokens[0].upper()
            if head not in headers: raise ValueError(f'Unsupported MPS section {head!r} at line {lineno}.')
            section=head
            if head=='ENDATA': break
            if head=='NAME' and len(tokens)>1: name=tokens[1]
            if head=='OBJSENSE' and len(tokens)>1: sense=tokens[1].lower()
            if head=='OBJNAME' and len(tokens)>1: objname=tokens[1]
            continue
        try:
            if section=='OBJSENSE': sense=tokens[0].lower()
            elif section=='OBJNAME': objname=tokens[0]
            elif section=='ROWS':
                kind,rname=tokens
                if kind not in ('N','E','L','G') or rname in rows: raise ValueError('Invalid/duplicate row.')
                rows[rname]=kind
                if kind=='N' and obj is None: obj=rname
            elif section=='COLUMNS':
                if any(t.strip("'\"")=='MARKER' for t in tokens):
                    if tokens[-1].strip("'\"") not in ('INTORG','INTEND'): raise ValueError('Invalid integer marker.')
                    inside=tokens[-1].strip("'\"")=='INTORG'; continue
                if len(tokens)%2==1: col=tokens[0]; values=tokens[1:]; last_col=col
                else:
                    if last_col is None: raise ValueError('Column continuation without a column.')
                    col=last_col; values=tokens
                entries=columns.setdefault(col,{})
                if inside: integer.add(col)
                for k in range(0,len(values),2):
                    row,val=values[k],number(values[k+1])
                    if row not in rows: raise ValueError(f'Unknown row {row}.')
                    entries[row]=entries.get(row,0.)+val
            elif section in ('RHS','RANGES'):
                if len(tokens)%2!=1: raise ValueError('Named RHS/RANGES records are required.')
                group=tokens[0]; values=tokens[1:]
                target=(rhs_sets if section=='RHS' else range_sets).setdefault(group,{})
                for k in range(0,len(values),2):
                    if values[k] not in rows: raise ValueError('Unknown RHS/RANGES row.')
                    target[values[k]]=number(values[k+1])
            elif section=='BOUNDS':
                if len(tokens) not in (3,4): raise ValueError('Malformed BOUNDS record.')
                kind,group,col=tokens[:3]
                if kind not in ('LO','UP','FX','FR','MI','PL','BV','LI','UI','SC','SI'): raise ValueError(f'Unsupported bound type {kind}.')
                if kind in ('LO','UP','FX','LI','UI','SC','SI') and len(tokens)!=4: raise ValueError('Bound value required.')
                columns.setdefault(col,{})
                bound_sets.setdefault(group,[]).append((kind,col,number(tokens[3]) if len(tokens)==4 else None))
            else: raise ValueError('Unexpected data record.')
        except (ValueError,IndexError) as e: raise ValueError(f'MPS line {lineno}: {e}') from e
    obj=objname or obj
    if obj is None or obj not in rows or rows[obj]!='N': raise ValueError('No valid MPS objective row.')
    if sense not in ('min','max','minimize','maximize'): raise ValueError('Invalid objective sense.')
    sense='max' if sense.startswith('max') else 'min'
    def selected(groups,choice):
        if choice is not None:
            if choice not in groups: raise ValueError(f'Unknown set {choice!r}.')
            return groups[choice]
        return next(iter(groups.values()),{})
    rh=selected(rhs_sets,rhs_name); ra=selected(range_sets,ranges_name); bo=selected(bound_sets,bounds_name)
    cn=list(columns); rn=[r for r in rows if r!=obj]; ci={v:i for i,v in enumerate(cn)}; ri={v:i for i,v in enumerate(rn)}
    c=np.zeros(len(cn)); rr=[]; cc=[]; vv=[]
    for col,entries in columns.items():
        j=ci[col]; c[j]=entries.get(obj,0.)
        for row,val in entries.items():
            if row!=obj and val: rr.append(ri[row]); cc.append(j); vv.append(val)
    lo=np.full(len(rn),-np.inf); hi=np.full(len(rn),np.inf)
    for r,i in ri.items():
        typ=rows[r]; b=rh.get(r,0.)
        if typ=='E': lo[i]=hi[i]=b
        elif typ=='L': hi[i]=b
        elif typ=='G': lo[i]=b
        if r in ra:
            v=ra[r]
            if typ=='L': lo[i]=b-abs(v)
            elif typ=='G': hi[i]=b+abs(v)
            elif typ=='E':
                if v>=0: hi[i]=b+v
                else: lo[i]=b+v
            else: raise ValueError('RANGES cannot be applied to a free row.')
    lb=np.zeros(len(cn)); ub=np.full(len(cn),np.inf); it=np.array([int(v in integer) for v in cn])
    bounded_columns={col for _,col,_ in bo}
    if integer_default=='binary':
        for col in integer-bounded_columns: ub[ci[col]]=1.
    explicit_lo=set()
    for kind,col,val in bo:
        j=ci[col]
        if kind in ('LO','LI'): lb[j]=val; explicit_lo.add(j)
        if kind in ('UP','UI'): ub[j]=val
        if kind in ('LI','UI'): it[j]=1
        if kind=='FX': lb[j]=ub[j]=val; explicit_lo.add(j)
        if kind=='FR': lb[j]=-np.inf; ub[j]=np.inf; explicit_lo.add(j)
        if kind=='MI': lb[j]=-np.inf; explicit_lo.add(j)
        if kind=='PL': ub[j]=np.inf
        if kind=='BV': lb[j]=0; ub[j]=1; it[j]=1; explicit_lo.add(j)
        if kind in ('SC','SI'):
            ub[j]=val; it[j]=2 if kind=='SC' else 3
            if j not in explicit_lo: lb[j]=1.
    for j in range(len(cn)):
        if ub[j]<0 and j not in explicit_lo:
            raise ValueError('A negative MPS upper bound without an explicit lower bound is dialect-dependent; add MI or LO explicitly.')
    A=sparse.csc_matrix((vv,(rr,cc)),shape=(len(rn),len(cn)))
    return LinearProblem(c,A,lo,hi,lb,ub,it,sense,-rh.get(obj,0.),cn,rn,name)


def _read_lp_highspy(path):
    try: import highspy as h
    except ImportError as e: raise ImportError('LP import requires highspy: python -m pip install highspy. JSON/MPS import has no extra dependency.') from e
    engine=h.Highs(); engine.setOptionValue('output_flag',False)
    status=engine.readModel(str(path))
    if status!=h.HighsStatus.kOk: raise ValueError(f'HiGHS could not parse {path}.')
    model=engine.getModel()
    if len(model.hessian_.value_):
        raise ValueError('The LP file contains a quadratic objective; linear import cannot discard it.')
    lp=engine.getLp(); a=lp.a_matrix_
    if a.format_==h.MatrixFormat.kColwise:
        A=sparse.csc_matrix((a.value_,a.index_,a.start_),shape=(lp.num_row_,lp.num_col_))
    else: A=sparse.csr_matrix((a.value_,a.index_,a.start_),shape=(lp.num_row_,lp.num_col_)).tocsc()
    integer=[int(v) for v in lp.integrality_] if len(lp.integrality_) else 0
    if not np.isin(integer,[0,1,2,3]).all(): raise ValueError('Unsupported HiGHS variable domain.')
    return LinearProblem(lp.col_cost_,A,lp.row_lower_,lp.row_upper_,lp.col_lower_,lp.col_upper_,integer,
        'min' if lp.sense_==h.ObjSense.kMinimize else 'max',lp.offset_,list(lp.col_names_) or None,list(lp.row_names_) or None,Path(path).stem)


def write_problem(p,path):
    suffix=Path(path).suffix.lower()
    if suffix=='.json': return write_json(p,path)
    if suffix=='.mps': return write_mps(p,path)
    if suffix=='.lp': return write_lp(p,path)
    raise ValueError('Supported output extensions are .json, .mps and .lp.')


def read_problem(path):
    suffix=Path(path).suffix.lower()
    if suffix=='.json': return read_json(path)
    if suffix=='.mps': return read_mps(path)
    if suffix=='.lp': return _read_lp_highspy(path)
    raise ValueError('Supported input extensions are .json, .mps and .lp.')
