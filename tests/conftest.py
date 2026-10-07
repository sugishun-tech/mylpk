import sys
from pathlib import Path
# Source-checkout tests work after build_ext --inplace; installed-wheel tests
# use the installed package when this src directory is not present.
source=Path(__file__).resolve().parents[1]/'src'
if source.is_dir(): sys.path.insert(0,str(source))
