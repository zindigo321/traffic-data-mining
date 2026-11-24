from pathlib import Path
import os,sys
_dll=None
if os.name=='nt' and (Path(sys.prefix)/'Library/bin').is_dir():
    _dll=os.add_dll_directory(str(Path(sys.prefix)/'Library/bin'))
    os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.font_manager import FontProperties

TASK=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=WORK/'figures';OUT.mkdir(exist_ok=True)
A=Path('G:/course-jiaotongshujuwajue/ProjectA/results')
B=TASK/'outputs/ProjectB_GT_niulaoshi_20261112/ProjectB_GT_niulaoshi/reference'
plt.rcParams.update({'font.family':'Times New Roman','font.size':15,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#111111','text.color':'#111111',
 'axes.edgecolor':'#777777','grid.color':'#DBE1EA','savefig.facecolor':'white'})
BLUE='#1B4F91';RED='#C43C39'

def finish(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.15)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

p=pd.read_csv(A/'point_labels.csv.gz',usecols=['x','y','dbscan'])
z=p.assign(gx=np.floor(p.x/500).astype(int),gy=np.floor(p.y/500).astype(int))
g=z.groupby(['gx','gy']).agg(all_points=('dbscan','size'),clustered_points=('dbscan',lambda v:int((v>=0).sum()))).reset_index()
g.to_csv(OUT/'A_grid500m.csv',index=False)
arr=np.full((g.gy.max()-g.gy.min()+1,g.gx.max()-g.gx.min()+1),np.nan)
arr[g.gy-g.gy.min(),g.gx-g.gx.min()]=np.log10(1+g.clustered_points)
fig,ax=plt.subplots(figsize=(7.2,4.9),layout='constrained')
im=ax.imshow(arr,origin='lower',cmap='Blues',extent=[g.gx.min()*.5,(g.gx.max()+1)*.5,g.gy.min()*.5,(g.gy.max()+1)*.5],aspect='equal')
ax.set(xlabel='UTM Easting (km)',ylabel='UTM Northing (km)')
cb=fig.colorbar(im,ax=ax,shrink=.85);cb.set_label('log10(1 + clustered points / 500 m cell)')
finish(fig,'A_spatial_density')

d=pd.read_csv(A/'dbscan_experiments.csv');d=d[d.min_samples==20]
fig,ax=plt.subplots(figsize=(7.2,4.4),layout='constrained')
ax.plot(d.eps_m,d.noise_fraction*100,'o-',color=BLUE,lw=2)
main=d[d.eps_m==50].iloc[0]
ax.scatter([50],[main.noise_fraction*100],color=RED,s=100,zorder=5)
ax.annotate('50 m / 20',xy=(50,main.noise_fraction*100),xytext=(85,55),color=RED,
 arrowprops={'arrowstyle':'-','color':RED},fontsize=15)
ax.set(xlabel='DBSCAN radius (m)',ylabel='Noise fraction (%)',xscale='log',ylim=(0,70),xticks=d.eps_m)
ax.set_xticklabels([str(v) for v in d.eps_m]);ax.grid(axis='y',alpha=.7)
ax.xaxis.set_minor_locator(NullLocator())
finish(fig,'A_parameter_noise')

k=pd.read_csv(A/'kmeans_experiments.csv')
fig,ax=plt.subplots(figsize=(7.1,4.8),layout='constrained')
ax.plot(k.k,k.inertia/1e12,'o-',color=BLUE,lw=2)
q=k[k.k==5].iloc[0];ax.scatter([5],[q.inertia/1e12],color=RED,s=100,zorder=5)
ax.annotate('K = 5 (elbow rule)',xy=(5,q.inertia/1e12),xytext=(5.4,1.7),color=RED,
 arrowprops={'arrowstyle':'-','color':RED},fontsize=15)
ax.set(xlabel='Number of clusters K',ylabel='Inertia (10$^{12}$ m$^2$)',xticks=k.k)
ax.grid(axis='y',alpha=.7);finish(fig,'A_kmeans_elbow')

sizes=p[p.dbscan>=0].groupby('dbscan').size().sort_values(ascending=False).head(10)
fig,ax=plt.subplots(figsize=(5.1,3.1),layout='constrained')
ax.bar(np.arange(1,11),sizes.values,color=BLUE,width=.65)
ax.set(xlabel='Cluster rank',ylabel='Points',xticks=[1,3,5,7,9]);ax.grid(axis='y',alpha=.5)
finish(fig,'A_top_cluster_sizes')

stats=json.loads((B/'results/summary.json').read_text())['2']['constrained']
rows=[]
for m,v in stats.items():
    rows.append({'model':m,**{key:v[key] for key in ('AP','AP_sd','f1','f1_sd','balanced_accuracy','recall','FP_day','recovery_FP_day')}})
pd.DataFrame(rows).to_csv(OUT/'B_plot_values.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.9),layout='constrained')
for ax,key,label in zip(axes,['AP','f1'],['Average Precision (AP)','Positive-class F1']):
    for i,(pre,color,name) in enumerate([('graph_',BLUE,'Static'),('dynamic_',RED,'Dynamic')]):
        models=[pre+q for q in ['mean','max','attention']]
        x=np.arange(3)+(i-.5)*.32
        means=[stats[m][key] for m in models];sds=[stats[m][key+'_sd'] for m in models]
        ax.bar(x,means,width=.32,yerr=sds,capsize=4,color=color,label=name)
        for xp,value in zip(x,means):ax.text(xp,value+.045,f'{value:.3f}',ha='center',va='bottom',fontsize=13,color=color)
    ax.axhline(.6,color='#333333',lw=1,ls='--')
    ax.set(ylabel=label,xticks=np.arange(3),xticklabels=['Mean','Max','Attention'],ylim=(0,.82))
    ax.grid(axis='y',alpha=.6)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
finish(fig,'B_pooling_AP_F1')

paired=json.loads((B/'results/paired_summary.json').read_text())['2']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
for ax,key,label in zip(axes,['FP_per_day','recovery_FP_per_day'],['Total FP / day','Recovery FP / day']):
    for i,(pre,color,name) in enumerate([('graph_',BLUE,'Static'),('dynamic_',RED,'Dynamic')]):
        values=[paired[pre+q]['matched60'][key] for q in ['mean','max','attention']]
        sds=[paired[pre+q]['matched60'][key+'_sd'] for q in ['mean','max','attention']]
        ax.bar(np.arange(3)+(i-.5)*.32,values,width=.32,yerr=sds,capsize=4,color=color,label=name)
    ax.set(ylabel=label,xticks=np.arange(3),xticklabels=['Mean','Max','Attention'],ylim=(0,20 if key=='FP_per_day' else 10))
    ax.grid(axis='y',alpha=.6)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
finish(fig,'B_matched_recall')
print(json.dumps({'figures':list(p.name for p in OUT.glob('*.png')),'privacy':'500 m aggregate only; no user IDs or individual traces'},ensure_ascii=False))
