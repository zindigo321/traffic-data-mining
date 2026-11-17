import os, shutil
from pathlib import Path
import requests
from v5_io import sha,save,now
root=Path(os.environ["GT5_ROOT"])
base="https://data.cityofnewyork.us/api/views/i4gi-tjb9/files/02907e84-631f-4c04-814e-6f7144cf8dd1"
response=requests.get(base,params={"download":"true","filename":"metadata_trafficspeeds.pdf"},timeout=(8,20))
obj=dict(url=response.url,utc=now(),status=response.status_code,bytes=len(response.content),content_type=response.headers.get("Content-Type"))
if response.content.startswith(b"%PDF"):
    path=root/"evidence/nyc_dictionary.pdf"
    path.write_bytes(response.content)
    obj["sha256"]=sha(path)
obj["pdftotext"]=shutil.which("pdftotext")
save(root/"evidence/dictionary_receipt.json",obj)
print(obj)
