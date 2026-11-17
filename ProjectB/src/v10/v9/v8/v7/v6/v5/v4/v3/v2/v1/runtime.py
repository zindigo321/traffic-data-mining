"""Local process DLL search configuration; does not modify the Conda environment."""
import os,sys
from pathlib import Path
_dll_handles=[]
if os.name=='nt':
 library=Path(sys.prefix)/'Library/bin'
 if library.exists():
  os.environ['PATH']=str(library)+os.pathsep+os.environ.get('PATH','')
  _dll_handles.append(os.add_dll_directory(str(library)))
 gdal=Path(sys.prefix)/'Library/share/gdal'
 if gdal.exists():os.environ.setdefault('GDAL_DATA',str(gdal))
