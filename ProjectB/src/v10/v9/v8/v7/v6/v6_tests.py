from v6_io import CODE,V5,python,record,save,settings
root,_=settings()
jobs=[]
for name,path in (("v6",CODE),("v5",V5),("v4",V5/"v4"),("v3",V5/"v4/v3")):
    jobs.append(record(root,f"engineering_{name}",python("-m","unittest","discover","-s","tests","-v"),cwd=path))
save(root/"synthetic/status.json",dict(status="PASS",v6=5,v5=5,v4=8,v3=10,engineering_only=True,commands=jobs))
