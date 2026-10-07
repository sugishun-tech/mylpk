"""Build a fresh source ZIP and test its installed wheel in a separate target.

Uses the active interpreter and preinstalled third-party build/runtime/test
dependencies. This is not a cross-platform or completely fresh-OS test.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from package import package


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=Path('reports/wheel_validation.txt'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=args.report.resolve();report.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    with report.open('w',encoding='utf-8') as log,tempfile.TemporaryDirectory(prefix='mylpk-wheel-check-') as temporary:
        base=Path(temporary);archive=base/'source.zip';package(root,archive)
        with zipfile.ZipFile(archive) as z:z.extractall(base/'extract')
        source=base/'extract/mylpk';wheel_dir=base/'wheels';target=base/'installed';work=base/'run';work.mkdir()
        def run(cmd,cwd,run_env=env):
            print('RUN:', ' '.join(map(str,cmd)),flush=True)
            log.write('\n$ '+' '.join(map(str,cmd))+'\n');log.flush()
            q=subprocess.run(list(map(str,cmd)),cwd=cwd,env=run_env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if q.returncode: raise RuntimeError(f'Validation failed ({q.returncode}); inspect {report}')
        # No generated C or extension binaries are present in the source ZIP.
        assert not list(source.rglob('*.so')) and not list(source.rglob('*.c'))
        run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',wheel_dir,'.'],source)
        wheels=list(wheel_dir.glob('mylpk-*.whl'))
        if len(wheels)!=1: raise RuntimeError('Expected exactly one newly built wheel.')
        run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',target,wheels[0]],work)
        installed_env=env|{'PYTHONPATH':str(target)}
        check="import pathlib,mylpk,myomo; print(mylpk.__file__); print(myomo.__file__); assert pathlib.Path(mylpk.__file__).is_relative_to(pathlib.Path("+repr(str(target))+"))"
        run([sys.executable,'-c',check],work,installed_env)
        shutil.copytree(source/'tests',work/'tests')
        run([sys.executable,'-m','pytest','-q',work/'tests','--junitxml='+str(root/'reports/installed_wheel_pytest.xml')],work,installed_env)
        shutil.copytree(source/'examples',work/'examples')
        for example in sorted((work/'examples').glob('*.py')):
            run([sys.executable,example],work,installed_env)
        run([sys.executable,'-m','mylpk','solve',work/'examples/production.json','--solver','highs'],work,installed_env)
        log.write('\nSUCCESS: fresh Cython source build, installed-wheel tests and all examples passed.\n')
    print('SUCCESS:',report)

if __name__=='__main__':main()
