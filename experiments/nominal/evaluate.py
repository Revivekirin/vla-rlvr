"""Original LIBERO Long/Spatial evaluation. Goal is preserved in Stage 0."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.libero_plus.suite_common import cli

if __name__=='__main__':
    try: sys.exit(cli('nominal'))
    except (ValueError, OSError, KeyError) as e:
        print('Suite evaluation failed: '+str(e),file=sys.stderr)
        sys.exit(2)
