"""Per-invocation paths; source code and past runs never overwritten by the driver."""
import os,json
from pathlib import Path
CODE_ROOT=Path(__file__).resolve().parent
WORK_ROOT=Path(os.environ.get('PROJECTB_RUN_ROOT',str(CODE_ROOT))).resolve()
CONFIG_PATH=Path(os.environ.get('PROJECTB_CONFIG',str(CODE_ROOT/'config.json'))).resolve()
CONFIG=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
