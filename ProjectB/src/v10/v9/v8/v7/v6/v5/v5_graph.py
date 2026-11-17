"""Build an explicit directed link graph from a predevelopment geometry snapshot."""
from collections import defaultdict
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from v5_io import save, sha
from v5_download import source_files


DIRECTIONS = re.compile(r"\b(?:NORTHBOUND|SOUTHBOUND|EASTBOUND|WESTBOUND|NB|SB|EB|WB|N|S|E|W)\b")
METERS_PER_DEGREE = 111195.0


def parse_polyline(encoded):
    value=shift=0
    deltas=[]
    for c in encoded:
        digit=ord(c)-63
        if not 0<=digit<=63 or shift>=35:
            raise ValueError("Invalid polyline character")
        value |= (digit & 31) << shift
        if digit < 32:
            deltas.append(-(value>>1)-1 if value&1 else value>>1)
            value=shift=0
        else:
            shift+=5
    if shift or len(deltas)<4 or len(deltas)%2:
        raise ValueError("Incomplete polyline")
    return np.cumsum(np.array(deltas,dtype=float).reshape(-1,2),axis=0)*1e-5


def parse_points(text):
    return np.array([[float(v) for v in token.split(",")] for token in text.split()],dtype=float)


def check_points(points):
    if (points.ndim!=2 or points.shape[0]<2 or points.shape[1]!=2 or not np.isfinite(points).all()
            or not np.all((points[:,0]>40.3)&(points[:,0]<41.1)&(points[:,1]>-74.55)&(points[:,1]<-73.2))):
        raise ValueError("Geometry missing or outside the NYC region")
    # Ignore repeated points; they do not define a tangent.
    return points[np.r_[True,(np.diff(points,axis=0)!=0).any(axis=1)]]


def positions(row):
    for field,fn in (("encoded_poly_line",parse_polyline),("link_points",parse_points)):
        try:
            points=check_points(fn(row[field]))
            if len(points)>1 and not np.allclose(points[0],points[-1],rtol=0,atol=1e-6):
                return points,field
        except (ValueError,TypeError):
            continue
    raise ValueError("No usable ordered geometry")


def vector(a,b):
    delta=np.asarray(b)-np.asarray(a)
    return np.array([delta[0],delta[1]*math.cos(math.radians((a[0]+b[0])/2))])*METERS_PER_DEGREE


def distance(a,b):
    return float(np.linalg.norm(vector(a,b)))


def corridor(name):
    parts=DIRECTIONS.split(name.upper(),maxsplit=1)
    return re.sub(r"[^A-Z0-9]","",parts[0]).strip() if len(parts)>1 else ""


def graph_arrays(paths,table,cfg):
    n=len(table)
    neighbors=defaultdict(set)
    successors=defaultdict(set)
    edges=[]
    cosine=math.cos(math.radians(cfg["turn_angle_max"]))
    for a,src in enumerate(paths):
        u=vector(src[-2],src[-1])
        for b,dst in enumerate(paths):
            if a==b or distance(src[0],dst[-1])<=cfg["endpoint_near_m"]:
                continue
            gap=distance(src[-1],dst[0])
            same=bool(table.iloc[a]["corridor"] and table.iloc[a]["corridor"]==table.iloc[b]["corridor"])
            if gap>cfg["endpoint_near_m"] and not (same and gap<=cfg["endpoint_same_corridor_m"]):
                continue
            v=vector(dst[0],dst[1])
            alignment=float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
            if alignment<=cosine:
                continue
            successors[a].add(b)
            neighbors[a].add(b)
            neighbors[b].add(a)
            edges.append(dict(src=a+1,dst=b+1,gap_m=gap,cosine=alignment,
                              rule="near_endpoint" if gap<=cfg["endpoint_near_m"] else "same_corridor"))
    groups=[[i,*sorted(neighbors[i])[:cfg["max_nodes"]-1]] for i in range(n)]
    width=max(map(len,groups))
    lookup=np.zeros((n,width),dtype=np.int32)
    valid=np.zeros_like(lookup,dtype=bool)
    adj=np.zeros((n,width,width),dtype=bool)
    for i,group in enumerate(groups):
        lookup[i,:len(group)]=group
        valid[i,:len(group)]=True
        index={road:j for j,road in enumerate(group)}
        for road,j in index.items():
            for target in successors[road] & index.keys():
                adj[i,j,index[target]]=True
    return dict(lookup=lookup,valid=valid,adj=adj,road_ids=table.index.to_numpy(dtype=np.int64)),edges


def build(root,cfg):
    [(file,receipt)]=list(source_files(root,"geometry"))
    frame=pd.read_csv(file,dtype=str,keep_default_na=False)
    if len(frame)!=receipt["rows"]:
        raise ValueError("Metadata row count differs")
    cols=["link_id","link_name","borough","encoded_poly_line","link_points"]
    summary=frame.value_counts(cols).rename("count").reset_index()
    winners=summary.sort_values(["link_id","count",*cols[1:]],ascending=[True,False,True,True,True,True]).drop_duplicates("link_id")
    rows, paths, excluded=[],[],[]
    for row in winners.to_dict("records"):
        if not row["link_id"]:
            continue
        try:
            pts, field=positions(row)
        except ValueError:
            excluded.append(row["link_id"])
            continue
        delta=vector(pts[0],pts[-1])
        direction=(("NB" if delta[0]>0 else "SB") if abs(delta[0])>=abs(delta[1]) else ("EB" if delta[1]>0 else "WB"))
        miles=sum(distance(a,b) for a,b in zip(pts[:-1],pts[1:],strict=True))/1609.344
        rows.append(dict(road_code=len(rows)+1,link_id=row["link_id"],link_name=row["link_name"],borough=row["borough"],
                         geometry_field=field,metadata_records=row["count"],direction=direction,length=miles,
                         corridor=corridor(row["link_name"]),entry_lat=pts[0,0],entry_lon=pts[0,1],
                         exit_lat=pts[-1,0],exit_lon=pts[-1,1]))
        paths.append(pts)
    table=pd.DataFrame(rows).set_index("road_code")
    if not 1<len(table)<1000:
        raise ValueError("Graph needs a fixed nonempty roster with codes under 1000")
    graph,edges=graph_arrays(paths,table,cfg)
    if not edges:
        raise ValueError("No validated direction-compatible links")
    prepared=root/"prepared"
    prepared.mkdir(exist_ok=True)
    table.to_csv(root/"evidence/nyc_roads.csv",float_format="%.17g")
    pd.DataFrame(edges).to_csv(root/"evidence/nyc_edges.csv",index=False,float_format="%.17g")
    np.savez_compressed(prepared/"topology.npz",**graph)
    info=dict(road_count=len(table),edges=len(edges),isolated=int((graph["valid"].sum(axis=1)==1).sum()),
              max_subgraph_nodes=int(graph["valid"].sum(axis=1).max()),outgoing_and_incoming_scope="one-hop, fixed ordered endpoints",
              anchor=cfg["geometry_window"],metadata_links=int(winners.shape[0]),excluded_links=excluded,
              metadata_link_conflicts=int((summary.groupby("link_id").size()>1).sum()),
              geometry_methods=table["geometry_field"].value_counts().to_dict(),
              source_sha256=receipt["sha256"],roster_sha256=sha(root/"evidence/nyc_roads.csv"),
              adjacency_sha256=sha(prepared/"topology.npz"),edge_sha256=sha(root/"evidence/nyc_edges.csv"))
    save(root/"evidence/graph.json",info)
    print(info,flush=True)
    return table,graph
