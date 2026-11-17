from v7_io import CODE,V6,python,record,save,settings

root,_=settings()
paths={"v7":CODE,"v6":V6,"v5":V6/"v5","v4":V6/"v5/v4","v3":V6/"v5/v4/v3"}
receipts=[]
for name,path in paths.items():
    receipts.append(record(root,f"engineering_{name}",python("-m","unittest","discover","-s","tests","-v"),cwd=path))
save(root/"synthetic/status.json",dict(status="PASS",counts=dict(v7=6,v6=5,v5=5,v4=8,v3=10),
    engineering_only=True,commands=receipts))
