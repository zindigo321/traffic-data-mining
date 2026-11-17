import argparse
from pathlib import Path
import sys
from v3_io import record

p=argparse.ArgumentParser()
p.add_argument("--root",type=Path,required=True)
p.add_argument("--name",required=True)
p.add_argument("command",nargs=argparse.REMAINDER)
a=p.parse_args()
sys.exit(record(a.root,a.name,a.command[1:] if a.command[0]=="--" else a.command,extra={"GT3_ROOT":str(a.root.resolve())}))

