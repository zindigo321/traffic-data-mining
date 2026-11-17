"""Static directed topology and target-centred graph samples, without label features."""
from collections import defaultdict
import json
import numpy as np
import pandas as pd

def osm_graph(path):
    raw=json.loads(path.read_text(encoding='utf-8'))
    nodes={}
    for e in raw['elements']:
        ns=e.get('nodes',[]);geo=e.get('geometry',[])
        if len(ns)<2 or len(geo)<2:continue
        tags=e.get('tags',{});one=tags.get('oneway','').lower()
        if one in ('yes','1','true') or (one=='' and tags.get('junction')=='roundabout'):dirs=[1]
        elif one=='-1':dirs=[-1]
        else:dirs=[1,-1]
        for direction in dirs:
            a,b=(ns[0],ns[-1]) if direction==1 else (ns[-1],ns[0])
            nodes[f"{e['id']}:{direction}"]={'entry':str(a),'exit':str(b),'way':e['id'],'direction':direction}
    outgoing=defaultdict(list)
    for k,v in nodes.items():outgoing[v['entry']].append(k)
    edges=[]
    for k,v in nodes.items():
        for nxt in outgoing[v['exit']]:
            if nodes[nxt]['way']!=v['way']:edges.append((k,nxt)) # prohibit immediate U-turn
    return nodes,edges

def neighborhood(center,edges,hops=2,max_nodes=16):
    neighbors=defaultdict(set)
    for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
    return select_nodes(center,neighbors,hops,max_nodes)

def select_nodes(center,neighbors,hops=2,max_nodes=16):
    if max_nodes<1:raise ValueError('max_nodes must be >= 1')
    found={center};front={center}
    for _ in range(hops):
        nxt=set().union(*(neighbors.get(k,set()) for k in front))-found if front else set()
        # Stable BFS, center never removed; graph is static and independent of all labels.
        take=sorted(nxt)[:max_nodes-len(found)]
        found.update(take);front=set(take)
        if len(found)>=max_nodes:break
    return [center]+sorted(found-{center})

def arrays_from_records(records,max_nodes,feature_count):
    count=len(records)
    out={'x':np.full((count,max_nodes,feature_count),np.nan,np.float32),
         'adj':np.zeros((count,max_nodes,max_nodes),bool),'valid':np.zeros((count,max_nodes),bool),
         'center':np.zeros(count,np.int64),'y':np.zeros(count,np.int64)}
    meta=[]
    for i,r in enumerate(records):
        ids=r['nodes'];n=len(ids)
        if not n or n>max_nodes or r['target'] not in ids:raise ValueError('Invalid graph/target')
        out['x'][i,:n]=r['x'];out['valid'][i,:n]=True
        idx={k:j for j,k in enumerate(ids)}
        for a,b in r['edges']:
            if a in idx and b in idx:out['adj'][i,idx[a],idx[b]]=True
        out['center'][i]=idx[r['target']];out['y'][i]=r['label']
        m={k:v for k,v in r.items() if k not in ('x','edges')};meta.append(m)
    return out,meta

def save_graphs(folder,arrays,meta):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'graphs.npz',**arrays)
    (folder/'samples.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def load_graphs(folder):
    with np.load(folder/'graphs.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    return arrays,json.loads((folder/'samples.json').read_text(encoding='utf-8'))

class TrainPreprocessor:
    """Median/standard deviation fitted on unique training (time,node) observations."""
    def fit(self,x,valid,meta=None):
        if meta is None:a=x[valid]
        else:
            seen=set();rows=[]
            for i,m in enumerate(meta):
                for j,k in enumerate(m['nodes']):
                    key=(m['time'],k)
                    if key not in seen:seen.add(key);rows.append(x[i,j])
            a=np.asarray(rows)
        if not len(a):raise ValueError('No training features')
        self.median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T],np.float32)
        a=np.where(np.isfinite(a),a,self.median)
        self.mean=a.mean(0);self.scale=a.std(0);self.scale=np.where(self.scale>1e-6,self.scale,1.)
        return self
    def transform(self,x,valid):
        missing=~np.isfinite(x)
        z=(np.where(missing,self.median,x)-self.mean)/self.scale
        return np.concatenate((z,missing.astype(np.float32)),axis=-1)*valid[:,:,None]
    def save(self,path):np.savez(path,median=self.median,mean=self.mean,scale=self.scale)
    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path,allow_pickle=False) as z:
            for k in ('median','mean','scale'):setattr(obj,k,z[k])
        return obj

def supervised_indices(y):return np.flatnonzero(np.isin(y,[0,1]))

def check_temporal(meta):
    ranges={s:[] for s in ('train','validation','test')}
    for m in meta:
        t=pd.Timestamp(m['time']);a=pd.Timestamp(m.get('feature_start',m['time']))
        e=pd.Timestamp(m.get('feature_end',m['time']))
        le=pd.Timestamp(m.get('label_end',m['time']))
        if a>e or e>t or le>t:raise ValueError('Future feature or label')
        ranges[m['split']].append((a,t))
    previous=None
    for split in ranges:
        if not ranges[split]:continue
        lo=min(a for a,b in ranges[split]);hi=max(b for a,b in ranges[split])
        if previous is not None and lo<=previous:raise ValueError('Input/label spans cross split boundary')
        previous=hi
    return {s:{'samples':len(v),'input_first':str(min(a for a,b in v)) if v else None,'decision_last':str(max(b for a,b in v)) if v else None} for s,v in ranges.items()}
