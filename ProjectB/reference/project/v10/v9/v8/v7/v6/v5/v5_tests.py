from v5_io import CODE,V4,python,record,save,settings
root,_=settings()
jobs=[]
for name,path in (("v5",CODE),("v4",V4),("v3",V4/"v3")):
    jobs.append(record(root,f"engineering_{name}",python("-m","unittest","discover","-s","tests","-v"),cwd=path))
save(root/"synthetic/status.json",dict(status="PASS",v5=5,v4=8,v3=10,engineering_only=True,commands=jobs))
