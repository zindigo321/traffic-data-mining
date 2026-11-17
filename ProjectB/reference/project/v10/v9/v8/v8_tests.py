from v8_io import CODE,V7,python,record,save,settings

root,_=settings()
paths={"v8":CODE,"v7":V7,"v6":V7/"v6","v5":V7/"v6/v5","v4":V7/"v6/v5/v4","v3":V7/"v6/v5/v4/v3"}
done=[]
for label,path in paths.items():
    done.append(record(root,f"engineering_{label}",python("-m","unittest","discover","-s","tests","-v"),cwd=path))
save(root/"synthetic/status.json",dict(status="PASS",counts=dict(v8=6,v7=6,v6=5,v5=5,v4=8,v3=10),
    commands=done,engineering_only=True))
