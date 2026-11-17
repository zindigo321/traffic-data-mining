"""Portable-only file guard inherited by every Python child; never installed globally."""
import os
from pathlib import Path
import sys

binding=os.environ.get('PROJECTB_PACKAGE_ROOT')
if binding:
    package=Path(binding).resolve()
    output=Path(os.environ.get('GTFULL_ROOT',binding)).resolve()
    prefix=Path(sys.prefix).resolve()
    def portable_guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]))
        if not path.is_absolute():return
        resolved=path.resolve()
        if any(resolved.is_relative_to(p) for p in (package,output,prefix)):return
        text=resolved.as_posix().lower()
        if any(marker in text for marker in ('/work/gt_v','/work/full_data/','/work/dynamic_edges/',
                  '/course-jiaotongshujuwajue/','/wechatt/xwechat_files/')):
            raise PermissionError(f'Portable runtime refuses historical external path: {resolved}')
    sys.addaudithook(portable_guard)
