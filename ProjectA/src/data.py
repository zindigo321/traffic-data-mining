"""ZIP inventory and traceable, trajectory-local preprocessing."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer

COLS = ['lat', 'lon', 'reserved', 'alt_ft', 'day', 'date', 'clock']

def read_plt(z, name):
    # CRCRLF occurs in this release; splitlines would create blank lines.
    text = z.read(name).decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '')
    d = pd.read_csv(io.StringIO(text), skiprows=6, names=COLS)
    d['raw_index'] = np.arange(len(d))  # zero-based data row; physical line = +7
    for c in COLS[:5]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['time'] = pd.to_datetime(d.date.astype(str)+' '+d.clock.astype(str),
                             format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)
    return d

def inside(d, cfg):
    w,e,s,n = cfg['bounds']
    return d.lon.between(w,e) & d.lat.between(s,n)

def valid(d):
    return d.lat.between(-90,90) & d.lon.between(-180,180) & d.time.notna()

def inventory(zip_path, out, cfg):
    records, preview = [], []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.plt'))
        labels = [n.split('/')[-2] for n in z.namelist() if n.endswith('/labels.txt')]
        for i,name in enumerate(names):
            d = read_plt(z,name)
            v = valid(d)
            a=d.loc[v]
            dt=a.time.diff().dt.total_seconds()
            records.append(dict(member=name,user_id=name.split('/')[-3],
                trajectory_id=name.split('/')[-3]+'_'+Path(name).stem,points=len(d),
                valid_points=int(v.sum()),region_points=int((v & inside(d,cfg)).sum()),
                invalid=int((~v).sum()),missing_fields=int(d[COLS].isna().any(axis=1).sum()),
                duplicates=int(d.duplicated(COLS).sum()),alt_invalid=int(d.alt_ft.eq(-777).sum()),
                nonpositive_dt=int(dt.le(0).sum()),gap_count=int(dt.gt(cfg['gap_seconds']).sum()),
                time_outside_release=int((v & ~d.time.between(pd.Timestamp(cfg['valid_time_start']),pd.Timestamp(cfg['valid_time_end']),inclusive='left')).sum()),
                median_dt_s=dt[dt.gt(0)].median(),time_start=str(a.time.min()),time_end=str(a.time.max()),
                lon_min=a.lon.min(),lon_max=a.lon.max(),lat_min=a.lat.min(),lat_max=a.lat.max()))
            if len(a):
                preview.append(a.iloc[::max(1,len(a)//20)][['lon','lat']].head(20))
            if i%2000==0: print(f'inventory {i}/{len(names)}',flush=True)
    inv=pd.DataFrame(records)
    inv.to_csv(out/'inventory.csv',index=False)
    pd.concat(preview).to_csv(out/'spatial_preview.csv.gz',index=False)
    summary=dict(users=int(inv.user_id.nunique()),trajectories=len(inv),points=int(inv.points.sum()),
        labels_users=len(labels),label_user_ids=labels,time_start=inv.time_start.min(),time_end=inv.time_end.max(),
        lon_min=float(inv.lon_min.min()),lon_max=float(inv.lon_max.max()),
        lat_min=float(inv.lat_min.min()),lat_max=float(inv.lat_max.max()),
        region_points=int(inv.region_points.sum()),invalid=int(inv.invalid.sum()),
        missing_fields=int(inv.missing_fields.sum()),duplicates=int(inv.duplicates.sum()),
        altitude_invalid=int(inv.alt_invalid.sum()),nonpositive_dt=int(inv.nonpositive_dt.sum()),
        gaps=int(inv.gap_count.sum()),time_outside_release=int(inv.time_outside_release.sum()))
    (out/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return inv

def clean_trajectory(d, cfg):
    """Quality marks on ALL rows. Speed never crosses rejected rows or >300s gaps."""
    d=d.copy()
    d['reason']='kept'
    d.loc[~valid(d),'reason']='invalid_coordinate_or_time'
    period=d.time.between(pd.Timestamp(cfg['valid_time_start']),pd.Timestamp(cfg['valid_time_end']),inclusive='left')
    d.loc[d.reason.eq('kept') & ~period,'reason']='time_outside_release'
    dup=d.duplicated(COLS)
    d.loc[dup & d.reason.eq('kept'),'reason']='exact_duplicate'
    a=d[d.reason.eq('kept')].sort_values(['time','raw_index']).copy()
    conflict=a.time.duplicated(keep=False)
    # Same timestamp with incompatible coordinates has no defensible ordering.
    d.loc[a.index[conflict],'reason']='timestamp_conflict'
    a=a[~conflict].copy()
    dt=a.time.diff().dt.total_seconds()
    lat=np.radians(a.lat.to_numpy()); lon=np.radians(a.lon.to_numpy())
    hav=np.sin(np.diff(lat)/2)**2+np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(np.diff(lon)/2)**2
    dist=np.r_[np.nan,6371008.8*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))]
    contiguous=a.raw_index.diff().eq(1) & dt.gt(0) & dt.le(cfg['gap_seconds'])
    speed=np.where(contiguous,dist/dt,np.nan)
    a['speed_mps']=speed
    jump=pd.Series(speed>cfg['jump_speed_mps'],index=a.index)
    # Mark both endpoints; exclude suspect edges rather than reconnecting them.
    suspect=jump | jump.shift(-1,fill_value=False)
    d.loc[a.index[suspect],'reason']='jump_edge_endpoint'
    a=a[~suspect].copy()
    a.loc[~a.raw_index.diff().eq(1),'speed_mps']=np.nan
    a['alt_m']=a.alt_ft.where(a.alt_ft.ne(-777))*0.3048
    return a,d

def prepare(zip_path,out,cfg,inv):
    eligible=inv.groupby('user_id').region_points.sum()
    users=sorted(eligible[eligible>=cfg['min_region_points']].index)[:cfg['users']]
    selected=[]
    for u in users:
        g=inv[(inv.user_id==u)&(inv.region_points>0)].copy()
        g['hash']=g.member.map(lambda x:hashlib.sha256(f"{cfg['seed']}:{x}".encode()).hexdigest())
        selected.append(g.sort_values('hash').head(cfg['trajectories_per_user']))
    manifest=pd.concat(selected)
    manifest.to_csv(out/'trajectory_manifest.csv',index=False)
    frames,quality,mapping,steps=[],[],[],[]
    trans=Transformer.from_crs('EPSG:4326','EPSG:32650',always_xy=True)
    with zipfile.ZipFile(zip_path) as z:
        for i,row in enumerate(manifest.itertuples()):
            raw=read_plt(z,row.member)
            a,q=clean_trajectory(raw,cfg)
            counts=q.reason.value_counts()
            before=len(raw)
            for reason in ['invalid_coordinate_or_time','time_outside_release','exact_duplicate','timestamp_conflict','jump_edge_endpoint']:
                removed=int(counts.get(reason,0))
                steps.append(dict(trajectory_id=row.trajectory_id,step=reason,before=before,removed=removed,after=before-removed))
                before-=removed
            for t in [q,a]:
                t['user_id']=row.user_id;t['trajectory_id']=row.trajectory_id
            quality.append(q[q.reason.ne('kept')][['user_id','trajectory_id','raw_index','time','lon','lat','reason']])
            b=a[inside(a,cfg)].copy()
            steps.append(dict(trajectory_id=row.trajectory_id,step='outside_region',before=len(a),removed=len(a)-len(b),after=len(b)))
            if b.empty: continue
            b['time_bin']=b.time.dt.floor(f"{cfg['thin_seconds']}s").astype(str)
            reps=b.groupby('time_bin').raw_index.transform('first')
            mp=b[['user_id','trajectory_id','raw_index','time_bin']].copy()
            mp['representative_raw_index']=reps
            mapping.append(mp)
            c=b.drop_duplicates('time_bin').copy()
            steps.append(dict(trajectory_id=row.trajectory_id,step='temporal_thinning',before=len(b),removed=len(b)-len(c),after=len(c)))
            c['x'],c['y']=trans.transform(c.lon.to_numpy(),c.lat.to_numpy())
            c['date_local']=c.time.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
            c['hour_local']=c.time.dt.tz_convert('Asia/Shanghai').dt.hour
            frames.append(c)
            if i%200==0: print(f'preprocess {i}/{len(manifest)}',flush=True)
    if not frames: raise ValueError('No valid points in configured region')
    thin=pd.concat(frames,ignore_index=True)
    sampled=[]
    for u,g in thin.groupby('user_id'):
        sampled.append(g.sample(n=min(len(g),cfg['points_per_user']),random_state=cfg['seed']))
    points=pd.concat(sampled).sort_values(['user_id','trajectory_id','time','raw_index']).reset_index(drop=True)
    points['point_id']=np.arange(len(points))
    points.to_csv(out/'analysis_sample.csv.gz',index=False)
    thin.to_csv(out/'thinned_points.csv.gz',index=False)
    mp=pd.concat(mapping)
    keys=points[['trajectory_id','raw_index','point_id']].rename(columns={'raw_index':'representative_raw_index'})
    mp=mp.merge(keys,how='left',on=['trajectory_id','representative_raw_index'])
    mp.to_csv(out/'raw_to_sample.csv.gz',index=False)
    pd.concat(quality).to_csv(out/'quality_issues.csv.gz',index=False)
    st=pd.DataFrame(steps)
    st.to_csv(out/'cleaning_by_trajectory.csv',index=False)
    st.groupby('step',sort=False)[['before','removed','after']].sum().to_csv(out/'cleaning_summary.csv')
    user=pd.DataFrame({'user_id':users})
    user['thin_points']=user.user_id.map(thin.groupby('user_id').size()).fillna(0).astype(int)
    user['sample_points']=user.user_id.map(points.groupby('user_id').size()).fillna(0).astype(int)
    user.to_csv(out/'user_manifest.csv',index=False)
    return points
