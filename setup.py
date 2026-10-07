"""Portable C build by default; opt in to host-specific instructions explicitly."""
import os
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

flags = ["/O2"] if sys.platform == "win32" else ["-O3", "-fno-math-errno"]
if os.environ.get("MYLPK_NATIVE") == "1":
    if sys.platform == "win32":
        raise RuntimeError("MYLPK_NATIVE is only supported with GCC/Clang.")
    flags += ["-march=native", "-mtune=native"]
# Deliberately no -ffast-math: NaN, infinity and numerical tests must stay valid.
exts = [Extension("mylpk._core", ["src/mylpk/_core.pyx"], language="c", extra_compile_args=flags),
        Extension("myomo._accumulate", ["src/myomo/_accumulate.pyx"], language="c", extra_compile_args=flags)]
setup(ext_modules=cythonize(exts, compiler_directives={"language_level": 3,
    "boundscheck": False, "wraparound": False, "initializedcheck": False,
    "cdivision": True}, annotate=os.environ.get("MYLPK_ANNOTATE") == "1"))
