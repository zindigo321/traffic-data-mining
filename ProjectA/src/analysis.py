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
    traj['interpretation']=np.where(traj.candidate,'低密度出行候选，非已证实行为异常','未触发规则')
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
    p['period']=pd.cut(p.hour_local,[-1,5,9,16,23],labels=['夜间00-06','早间06-10','日间10-17','晚间17-24'])
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
        ax.set(title=title,xlabel='UTM 50N 东向 / km',ylabel='UTM 50N 北向 / km')
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
        ax.grid(alpha=.15)
    def scatter(ax,frame,label,title):
        y=frame[label].to_numpy();mask=y>=0
        if (~mask).any(): ax.scatter(frame.x[~mask]/1000,frame.y[~mask]/1000,c='#a7acb4',s=2,alpha=.45,label='噪声')
        ax.scatter(frame.x[mask]/1000,frame.y[mask]/1000,c=y[mask]%20,cmap='tab20',vmin=0,vmax=19,s=3,alpha=.65)
        axes(ax,title)
        if (~mask).any(): ax.legend(loc='lower right')
    fig,aa=plt.subplots(1,2,figsize=(13,5))
    preview=pd.read_csv(out/'spatial_preview.csv.gz')
    aa[0].scatter(preview.lon,preview.lat,s=.2,alpha=.3)
    from matplotlib.patches import Rectangle
    w,e,s,n=cfg['bounds'];aa[0].add_patch(Rectangle((w,s),e-w,n-s,fill=False,edgecolor='red'))
    aa[0].set(title='图1a 全量轨迹空间预览（每文件最多20点）',xlabel='WGS84 经度 / °',ylabel='纬度 / °')
    aa[1].scatter(p.x/1000,p.y/1000,s=1,alpha=.25)
    axes(aa[1],'图1b 研究样本分布');save('01_region.png')
    st=pd.read_csv(out/'cleaning_summary.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    labels={'invalid_coordinate_or_time':'非法坐标或时间','time_outside_release':'超出采集期',
            'exact_duplicate':'完全重复','timestamp_conflict':'时间戳冲突','jump_edge_endpoint':'跳点边端点',
            'outside_region':'研究区域外','temporal_thinning':'分钟抽稀'}
    bars=ax.barh(st.step.map(labels),st.removed,color='#35759c')
    ax.set(xscale='symlog',xlabel='删除或筛除的点数（对称对数刻度）',title='图2 清洗、区域筛选和抽稀（分开记录）')
    ax.bar_label(bars,labels=[f'{v:,}' for v in st.removed],padding=5)
    ax.set_xlim(0,max(st.removed)*5)
    save('02_cleaning.png')
    fig,ax=plt.subplots(figsize=(7,6));scatter(ax,p,'dbscan',f'图3 DBSCAN：{main[0]} m / {main[1]}');save('03_dbscan.png')
    fig,aa=plt.subplots(1,2,figsize=(11,4))
    for m,g in dt.groupby('min_samples'):
        aa[0].plot(g.eps_m,g.clusters,'o-',label=f'min_samples={m}')
        aa[1].plot(g.eps_m,100*g.noise_fraction,'o-',label=f'min_samples={m}')
    for ax,title in zip(aa,['有效簇数','噪声比例 / %']):
        ax.set(xlabel='eps / m',ylabel=title,title='图4 '+title);ax.legend();ax.grid(alpha=.2)
    save('04_sensitivity.png')
    fig,ax=plt.subplots(figsize=(7,6))
    edgesx=np.arange(np.floor(X[:,0].min()/500)*500,X[:,0].max()+501,500)
    edgesy=np.arange(np.floor(X[:,1].min()/500)*500,X[:,1].max()+501,500)
    h=ax.hist2d(p.x/1000,p.y/1000,bins=[edgesx/1000,edgesy/1000],norm=LogNorm(),cmap='inferno',cmin=1)
    fig.colorbar(h[3],ax=ax,label='分析样本点数 / 500 m × 500 m')
    axes(ax,'图5 空间密度热力图');save('05_heatmap.png')
    fig,aa=plt.subplots(1,2,figsize=(13,6))
    scatter(aa[0],p,'dbscan',f'图6a DBSCAN {main}')
    scatter(aa[1],p,'kmeans',f'图6b K-means K={chosen}');save('06_comparison.png')
    chosen_users=users.sort_values(['points','user_id'],ascending=[False,True]).head(3).user_id.tolist()
    fig,aa=plt.subplots(1,3,figsize=(16,5))
    for ax,u in zip(aa,chosen_users):
        g=p[p.user_id==u].copy();g['personal']=DBSCAN(eps=main[0],min_samples=main[1]).fit_predict(g[['x','y']])
        scatter(ax,g,'personal',f'图7 用户{u}个人热点 (n={len(g)})')
    save('07_users.png')
    fig,aa=plt.subplots(1,max(1,len(cases)),figsize=(5*max(1,len(cases)),5),squeeze=False)
    for ax,row in zip(aa[0],cases.itertuples()):
        g=p[p.trajectory_id==row.trajectory_id]
        full=thin[thin.trajectory_id==row.trajectory_id].sort_values('time').copy()
        # Break lines at time gaps; never draw a link across omitted segments.
        segment=full.time.diff().dt.total_seconds().gt(cfg['gap_seconds']).cumsum()
        for _,seg in full.groupby(segment): ax.plot(seg.x/1000,seg.y/1000,color='#9ba8b6',lw=.6,zorder=0)
        ax.scatter(g.x/1000,g.y/1000,c=np.where(g.dbscan.lt(0),'#e06b4e','#247b9c'),s=10)
        ax.set(title=f'{row.case_type}\n{row.trajectory_id}\n噪声{row.noise_fraction:.1%}, n={row.points}',
               xlabel='东向 / km',ylabel='北向 / km');ax.set_aspect('equal');ax.grid(alpha=.2)
    if cases.empty: aa[0,0].text(.1,.5,'没有满足规则的候选或对照')
    fig.suptitle('图8 候选与对照：橙=噪声，蓝=热点；灰线=抽稀轨迹',y=1.04)
    save('08_cases.png')
    neighbors=min(cfg['main_min_samples'],len(X))
    neigh=NearestNeighbors(n_neighbors=neighbors,algorithm='kd_tree').fit(X)
    distance=neigh.kneighbors(X)[0][:,-1]
    pd.Series(distance).describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_csv(out/'kdistance_quantiles.csv')
    fig,aa=plt.subplots(1,3,figsize=(15,4))
    aa[0].plot(np.sort(distance));aa[0].set(yscale='log',xlabel='按距离排序的样本',ylabel=f'第{neighbors}近邻距离 / m',title='图9a k-distance（含自身）')
    aa[1].plot(kt.k,kt.inertia,'o-');aa[1].axvline(chosen,color='orange');aa[1].set(xlabel='K',ylabel='SSE / m²',title='图9b K-means 肘部')
    aa[2].plot(kt.k,kt.silhouette,'o-');aa[2].set(xlabel='K',ylabel='轮廓系数',title='图9c K-means（同一固定样本）')
    save('09_diagnostics.png')
    fig,aa=plt.subplots(1,3,figsize=(16,5))
    for ax,key in zip(aa,[(min(cfg['eps_m']),20),main,(500,5)]):
        p['display']=db[key];scatter(ax,p,'display',f'图10 eps={key[0]} m, min={key[1]}')
    save('10_parameter_examples.png')
    top=p[p.dbscan>=0].dbscan.value_counts().head(6).index
    temp=time_table[time_table.dbscan.isin(top)].pivot(index='period',columns='dbscan',values='within_period_share').fillna(0)
    fig,ax=plt.subplots(figsize=(10,4))
    if temp.empty:
        ax.text(.1,.5,'无有效热点（全噪声），不计算热点时段构成',transform=ax.transAxes)
    else:
        temp.plot.bar(ax=ax)
    ax.set(title='图11 固定热点的时段构成（UTC+8）',ylabel='占该时段全部样本的比例',xlabel='时段')
    if not temp.empty: ax.legend(title='热点编号',ncol=3)
    plt.xticks(rotation=0);save('11_time.png')
    print('Analysis and figures complete',flush=True)
