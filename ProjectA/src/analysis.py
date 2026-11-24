import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import DBSCAN,KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import psutil

def score(X,y,limit,seed):
    if len(X)<3 or not 2<=len(np.unique(y))<len(X):
        return np.nan,0,'NA: fewer than two clusters or insufficient points'
    idx=np.random.default_rng(seed).choice(len(X),min(limit,len(X)),replace=False)
    if not 2<=len(np.unique(y[idx]))<len(idx):
        return np.nan,len(idx),'NA: sampled labels do not satisfy silhouette conditions'
    return float(silhouette_score(X[idx],y[idx])),len(idx),'ok'

def clusters(p,label):
    q=p[p[label]>=0]
    if q.empty:
        return pd.DataFrame(columns=[label,'points','users','trajectories','days','lon','lat','width_m','height_m'])
    t=q.groupby(label).agg(points=('point_id','size'),users=('user_id','nunique'),
        trajectories=('trajectory_id','nunique'),days=('date_local','nunique'),
        lon=('lon','mean'),lat=('lat','mean'),x_min=('x','min'),x_max=('x','max'),
        y_min=('y','min'),y_max=('y','max'))
    t['width_m']=t.x_max-t.x_min;t['height_m']=t.y_max-t.y_min
    return t.sort_values('points',ascending=False).reset_index()

def analyze(out,cfg):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                        'axes.unicode_minus':False,'font.size':10})
    figs=out/'figures';figs.mkdir(exist_ok=True)
    p=pd.read_csv(out/'analysis_sample.csv.gz',dtype={'user_id':str},parse_dates=['time'])
    X=p[['x','y']].to_numpy()
    if len(X)<3: raise ValueError('At least 3 analysis points are required')
    seed=cfg['seed']; lim=cfg['silhouette_sample']
    # Pilot measures real resource cost before the formal grid.
    pilot_idx=np.random.default_rng(seed).choice(len(X),min(5000,len(X)),replace=False)
    t=time.perf_counter()
    DBSCAN(eps=max(cfg['eps_m']),min_samples=min(cfg['min_samples']),algorithm='kd_tree',n_jobs=4).fit(X[pilot_idx])
    pilot={'n':len(pilot_idx),'seconds':time.perf_counter()-t,'rss_mb_after':psutil.Process().memory_info().rss/2**20}
    (out/'pilot.json').write_text(json.dumps(pilot,indent=2),encoding='utf-8')
    dbrows=[]; db={}
    for eps in cfg['eps_m']:
        for m in cfg['min_samples']:
            t=time.perf_counter()
            model=DBSCAN(eps=eps,min_samples=m,algorithm='kd_tree',n_jobs=4).fit(X)
            elapsed=time.perf_counter()-t; y=model.labels_; keep=y>=0
            s,ns,status=score(X[keep],y[keep],lim,seed)
            sizes=np.unique(y[keep],return_counts=True)[1]
            dbrows.append(dict(eps_m=eps,min_samples=m,n=len(X),clusters=len(sizes),
                noise_fraction=float((~keep).mean()),coverage=float(keep.mean()),
                largest_cluster_fraction=float(sizes.max()/len(X)) if len(sizes) else 0,
                fit_seconds=elapsed,silhouette=s,silhouette_n=ns,status=status,
                rss_mb_after=psutil.Process().memory_info().rss/2**20))
            db[(eps,m)]=y
            print('DBSCAN',eps,m,dbrows[-1],flush=True)
            if (eps,m)==(cfg['main_eps_m'],cfg['main_min_samples']):
                p['core']=False
                p.loc[model.core_sample_indices_,'core']=True
    dt=pd.DataFrame(dbrows); dt.to_csv(out/'dbscan_experiments.csv',index=False)
    main=(cfg['main_eps_m'],cfg['main_min_samples'])
    if main not in db: raise ValueError('Main DBSCAN settings must belong to the experiment grid')
    p['dbscan']=db[main]; keep=p.dbscan.to_numpy()>=0
    kr=[]; km={}
    for k in cfg['k_values']:
        if k>=len(X): continue
        t=time.perf_counter()
        model=KMeans(n_clusters=k,random_state=seed,n_init=10,algorithm='lloyd').fit(X)
        sec=time.perf_counter()-t
        s,ns,status=score(X,model.labels_,lim,seed)
        kr.append(dict(k=k,n=len(X),inertia=model.inertia_,silhouette=s,
                       silhouette_n=ns,fit_seconds=sec,status=status))
        km[k]=model.labels_
    kt=pd.DataFrame(kr)
    if kt.empty: raise ValueError('No feasible K in the configured range')
    kt.to_csv(out/'kmeans_experiments.csv',index=False)
    # Geometric elbow: largest departure below line joining normalized endpoints.
    ks=kt.k.to_numpy(); inertia=kt.inertia.to_numpy()
    xn=(ks-ks.min())/max(1,ks.max()-ks.min())
    yn=(inertia-inertia.min())/max(1e-12,inertia.max()-inertia.min())
    elbow=1-xn-yn
    chosen=int(ks[np.argmax(elbow)])
    p['kmeans']=km[chosen]
    common=[]
    for name,y in [('DBSCAN',p.dbscan.to_numpy()),('KMeans',p.kmeans.to_numpy())]:
        s,ns,status=score(X[keep],y[keep],lim,seed)
        common.append(dict(model=name,common_n=int(keep.sum()),silhouette=s,silhouette_n=ns,status=status))
    pd.DataFrame(common).to_csv(out/'common_sample_comparison.csv',index=False)
    p.to_csv(out/'point_labels.csv.gz',index=False)
    clusters(p,'dbscan').to_csv(out/'hotspots.csv',index=False)
    clusters(p,'kmeans').to_csv(out/'kmeans_clusters.csv',index=False)
    userrows=[]
    for u,g in p.groupby('user_id'):
        y=DBSCAN(eps=main[0],min_samples=main[1],algorithm='kd_tree').fit_predict(g[['x','y']])
        userrows.append(dict(user_id=u,points=len(g),trajectories=g.trajectory_id.nunique(),
            personal_clusters=len(set(y)-{-1}),personal_noise_fraction=float(np.mean(y<0)),
            group_hotspots=g.loc[g.dbscan>=0,'dbscan'].nunique(),group_noise_fraction=float(g.dbscan.lt(0).mean()),
            dominant_group_hotspot=int(g.loc[g.dbscan>=0,'dbscan'].mode().iloc[0]) if g.dbscan.ge(0).any() else -1))
    users=pd.DataFrame(userrows);users.to_csv(out/'user_comparison.csv',index=False)
    relation=p[p.dbscan>=0].groupby(['user_id','dbscan']).size().reset_index(name='points')
    relation.to_csv(out/'user_hotspot_relation.csv',index=False)
    thin=pd.read_csv(out/'thinned_points.csv.gz',dtype={'user_id':str},parse_dates=['time'])
    available=thin.groupby('trajectory_id').size()
    quality=pd.read_csv(out/'quality_issues.csv.gz')
    qc=quality.groupby('trajectory_id').size()
    traj=p.groupby(['user_id','trajectory_id']).agg(points=('point_id','size'),
        noise_points=('dbscan',lambda v:int(v.lt(0).sum())),time_start=('time','min'),
        time_end=('time','max'),max_valid_speed_mps=('speed_mps','max'),
        lon_min=('lon','min'),lon_max=('lon','max'),lat_min=('lat','min'),lat_max=('lat','max')).reset_index()
    traj['noise_fraction']=traj.noise_points/traj.points
    traj['available_thinned_points']=traj.trajectory_id.map(available)
    traj['sampling_fraction']=traj.points/traj.available_thinned_points
    traj['quality_issue_rows']=traj.trajectory_id.map(qc).fillna(0).astype(int)
    traj['candidate']=(traj.points>=cfg['anomaly_min_points']) & (traj.noise_fraction>=cfg['anomaly_noise_fraction'])
    traj['interpretation']=np.where(traj.candidate,'Low-density travel candidate (unverified)','Rule not triggered')
    traj.to_csv(out/'trajectory_candidates.csv',index=False)
    cases=traj[traj.candidate].sort_values(['noise_fraction','points','trajectory_id'],ascending=[False,False,True]).head(3).copy()
    cases['case_type']='candidate'
    normal=traj[(traj.points>=cfg['anomaly_min_points']) & (traj.noise_fraction<=.1) & traj.quality_issue_rows.eq(0)].sort_values(['points','trajectory_id'],ascending=[False,True]).head(1).copy()
    normal['case_type']='reference'
    cases=pd.concat([cases,normal]);cases.to_csv(out/'case_studies.csv',index=False)
    # Density bias diagnostics: the same spatial grid before/after thinning.
    mp=pd.read_csv(out/'raw_to_sample.csv.gz',usecols=['trajectory_id','representative_raw_index'])
    weights=mp.groupby(['trajectory_id','representative_raw_index']).size().rename('raw_points')
    thin=thin.merge(weights,left_on=['trajectory_id','raw_index'],right_index=True,how='left')
    thin['gx']=np.floor(thin.x/cfg['grid_m']).astype(int);thin['gy']=np.floor(thin.y/cfg['grid_m']).astype(int)
    grid=thin.groupby(['gx','gy']).agg(raw_points=('raw_points','sum'),thinned_points=('raw_index','size'),users=('user_id','nunique')).reset_index()
    grid['raw_share']=grid.raw_points/grid.raw_points.sum()
    grid['thinned_share']=grid.thinned_points/grid.thinned_points.sum()
    grid.to_csv(out/'density_bias_grid.csv',index=False)
    # Fixed DBSCAN footprint, normalized within each UTC+8 time period.
    p['period']=pd.cut(p.hour_local,[-1,5,9,16,23],labels=['Night 00-06','Morning 06-10','Daytime 10-17','Evening 17-24'])
    time_table=p.groupby(['period','dbscan'],observed=True).size().reset_index(name='points')
    time_table['within_period_share']=time_table.points/time_table.groupby('period',observed=True).points.transform('sum')
    time_table.to_csv(out/'time_period_hotspots.csv',index=False)
    # Fixed selected subset for 90% perturbation stability; rerun all labels on subset.
    idx=np.sort(np.random.default_rng(seed).choice(len(X),int(.9*len(X)),replace=False))
    pert=DBSCAN(eps=main[0],min_samples=main[1],algorithm='kd_tree',n_jobs=4).fit_predict(X[idx])
    stab={'subset_fraction':.9,'n':len(idx),'ari_including_noise':adjusted_rand_score(p.dbscan.to_numpy()[idx],pert),
          'noise_fraction':float(np.mean(pert<0)),'clusters':len(set(pert)-{-1})}
    result={'main_eps_m':main[0],'main_min_samples':main[1],'kmeans_k':chosen,
        'points':len(p),'users':p.user_id.nunique(),'trajectories':p.trajectory_id.nunique(),
        'candidate_count':int(traj.candidate.sum()),'stability':stab,
        'grid_bias_total_variation':float(.5*np.abs(grid.raw_share-grid.thinned_share).sum()),
        'time_start':str(p.time.min()),'time_end':str(p.time.max())}
    (out/'analysis_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    def save(name):
        plt.savefig(figs/name,dpi=170,bbox_inches='tight');plt.close()
    extent=(X[:,0].min()/1000,X[:,0].max()/1000,X[:,1].min()/1000,X[:,1].max()/1000)
    def axes(ax,title):
        ax.set(title=title,xlabel='UTM 50N easting (km)',ylabel='UTM 50N northing (km)')
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
        ax.grid(alpha=.15)
    def scatter(ax,frame,label,title):
        y=frame[label].to_numpy();mask=y>=0
        if (~mask).any(): ax.scatter(frame.x[~mask]/1000,frame.y[~mask]/1000,c='#a7acb4',s=2,alpha=.45,label='Noise')
        ax.scatter(frame.x[mask]/1000,frame.y[mask]/1000,c=y[mask]%20,cmap='tab20',vmin=0,vmax=19,s=3,alpha=.65)
        axes(ax,title)
        if (~mask).any(): ax.legend(loc='lower right')
    fig,aa=plt.subplots(1,2,figsize=(13,5))
    preview=pd.read_csv(out/'spatial_preview.csv.gz')
    aa[0].scatter(preview.lon,preview.lat,s=.2,alpha=.3)
    from matplotlib.patches import Rectangle
    w,e,s,n=cfg['bounds'];aa[0].add_patch(Rectangle((w,s),e-w,n-s,fill=False,edgecolor='red'))
    aa[0].set(title='Figure 1a. Full trajectory overview (up to 20 points per file)',xlabel='WGS84 longitude (°)',ylabel='Latitude (°)')
    aa[1].scatter(p.x/1000,p.y/1000,s=1,alpha=.25)
    axes(aa[1],'Figure 1b. Study sample distribution');save('01_region.png')
    st=pd.read_csv(out/'cleaning_summary.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    labels={'invalid_coordinate_or_time':'Invalid coordinates or time','time_outside_release':'Outside collection period',
            'exact_duplicate':'Exact duplicate','timestamp_conflict':'Timestamp conflict','jump_edge_endpoint':'Endpoints of GPS jumps',
            'outside_region':'Outside study area','temporal_thinning':'One-minute thinning'}
    bars=ax.barh(st.step.map(labels),st.removed,color='#35759c')
    ax.set(xscale='symlog',xlabel='Points removed or filtered (symmetric log scale)',title='Figure 2. Cleaning, region filtering, and thinning')
    ax.bar_label(bars,labels=[f'{v:,}' for v in st.removed],padding=5)
    ax.set_xlim(0,max(st.removed)*5)
    save('02_cleaning.png')
    fig,ax=plt.subplots(figsize=(7,6));scatter(ax,p,'dbscan',f'Figure 3. DBSCAN {main[0]} m / {main[1]}');save('03_dbscan.png')
    fig,aa=plt.subplots(1,2,figsize=(11,4))
    for m,g in dt.groupby('min_samples'):
        aa[0].plot(g.eps_m,g.clusters,'o-',label=f'min_samples={m}')
        aa[1].plot(g.eps_m,100*g.noise_fraction,'o-',label=f'min_samples={m}')
    for ax,title in zip(aa,['Clusters','Noise (%)']):
        ax.set(xlabel='eps / m',ylabel=title,title='Figure 4. '+title);ax.legend();ax.grid(alpha=.2)
    save('04_sensitivity.png')
    fig,ax=plt.subplots(figsize=(7,6))
    edgesx=np.arange(np.floor(X[:,0].min()/500)*500,X[:,0].max()+501,500)
    edgesy=np.arange(np.floor(X[:,1].min()/500)*500,X[:,1].max()+501,500)
    h=ax.hist2d(p.x/1000,p.y/1000,bins=[edgesx/1000,edgesy/1000],norm=LogNorm(),cmap='inferno',cmin=1)
    fig.colorbar(h[3],ax=ax,label='Analysis points per 500 m × 500 m cell')
    axes(ax,'Figure 5. Spatial density');save('05_heatmap.png')
    fig,aa=plt.subplots(1,2,figsize=(13,6))
    scatter(aa[0],p,'dbscan',f'Figure 6a. DBSCAN {main}')
    scatter(aa[1],p,'kmeans',f'Figure 6b. K-means K={chosen}');save('06_comparison.png')
    chosen_users=users.sort_values(['points','user_id'],ascending=[False,True]).head(3).user_id.tolist()
    fig,aa=plt.subplots(1,3,figsize=(16,5))
    for ax,u in zip(aa,chosen_users):
        g=p[p.user_id==u].copy();g['personal']=DBSCAN(eps=main[0],min_samples=main[1]).fit_predict(g[['x','y']])
        scatter(ax,g,'personal',f'Figure 7. Hotspots for user {u} (n={len(g)})')
    save('07_users.png')
    fig,aa=plt.subplots(1,max(1,len(cases)),figsize=(5*max(1,len(cases)),5),squeeze=False)
    for ax,row in zip(aa[0],cases.itertuples()):
        g=p[p.trajectory_id==row.trajectory_id]
        full=thin[thin.trajectory_id==row.trajectory_id].sort_values('time').copy()
        # Break lines at time gaps; never draw a link across omitted segments.
        segment=full.time.diff().dt.total_seconds().gt(cfg['gap_seconds']).cumsum()
        for _,seg in full.groupby(segment): ax.plot(seg.x/1000,seg.y/1000,color='#9ba8b6',lw=.6,zorder=0)
        ax.scatter(g.x/1000,g.y/1000,c=np.where(g.dbscan.lt(0),'#e06b4e','#247b9c'),s=10)
        ax.set(title=f'{row.case_type}\n{row.trajectory_id}\nNoise {row.noise_fraction:.1%}, n={row.points}',
               xlabel='Easting (km)',ylabel='Northing (km)');ax.set_aspect('equal');ax.grid(alpha=.2)
    if cases.empty: aa[0,0].text(.1,.5,'No eligible candidates or controls')
    fig.suptitle('Figure 8. Candidates and controls (orange: noise, blue: hotspot, gray: thinned track)',y=1.04)
    save('08_cases.png')
    neighbors=min(cfg['main_min_samples'],len(X))
    neigh=NearestNeighbors(n_neighbors=neighbors,algorithm='kd_tree').fit(X)
    distance=neigh.kneighbors(X)[0][:,-1]
    pd.Series(distance).describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_csv(out/'kdistance_quantiles.csv')
    fig,aa=plt.subplots(1,3,figsize=(15,4))
    aa[0].plot(np.sort(distance));aa[0].set(yscale='log',xlabel='Samples sorted by distance',ylabel=f'Distance to neighbor {neighbors} (m)',title='Figure 9a. k-distance (including self)')
    aa[1].plot(kt.k,kt.inertia,'o-');aa[1].axvline(chosen,color='orange');aa[1].set(xlabel='K',ylabel='SSE / m²',title='Figure 9b. K-means elbow')
    aa[2].plot(kt.k,kt.silhouette,'o-');aa[2].set(xlabel='K',ylabel='Silhouette score',title='Figure 9c. K-means on the fixed sample')
    save('09_diagnostics.png')
    fig,aa=plt.subplots(1,3,figsize=(16,5))
    for ax,key in zip(aa,[(min(cfg['eps_m']),20),main,(500,5)]):
        p['display']=db[key];scatter(ax,p,'display',f'Figure 10. eps={key[0]} m, min={key[1]}')
    save('10_parameter_examples.png')
    top=p[p.dbscan>=0].dbscan.value_counts().head(6).index
    temp=time_table[time_table.dbscan.isin(top)].pivot(index='period',columns='dbscan',values='within_period_share').fillna(0)
    fig,ax=plt.subplots(figsize=(10,4))
    if temp.empty:
        ax.text(.1,.5,'All points are noise. No hotspots to compare.',transform=ax.transAxes)
    else:
        temp.plot.bar(ax=ax)
    ax.set(title='Figure 11. Fixed hotspots by time of day (UTC+8)',ylabel='Share of samples in the time period',xlabel='Time period')
    if not temp.empty: ax.legend(title='Hotspot ID',ncol=3)
    plt.xticks(rotation=0);save('11_time.png')
    print('Analysis and figures complete',flush=True)
