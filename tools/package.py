"""Create a source ZIP, excluding compiled artifacts, caches and hidden files."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile

EXCLUDED_DIRS={'build','dist','venv','env','__pycache__','node_modules'}
EXCLUDED_SUFFIXES={'.so','.pyd','.dll','.dylib','.pyc','.pyo','.o','.obj','.a','.c','.html','.whl','.zip'}


def source_files(root: Path):
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        rel=path.relative_to(root)
        if any(part.startswith('.') or part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in rel.parts): continue
        if path.name.upper().startswith(('LICENSE','LICENCE')): continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES: continue
        yield path,rel


def package(root: Path, output: Path):
    root=root.resolve();output=output.resolve()
    if not (root/'pyproject.toml').is_file(): raise ValueError('Source root must contain pyproject.toml.')
    output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path,rel in source_files(root):
            if path.resolve()==output: continue
            archive.write(path,Path('mylpk')/rel);count+=1
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad: raise RuntimeError(f'ZIP CRC check failed for {bad}.')
    return count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('dist/mylpk-0.1.0.zip'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    n=package(args.root,args.output)
    print(f'{args.output.resolve()} ({n} files, {args.output.stat().st_size} bytes)')

if __name__=='__main__':main()
