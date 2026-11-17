import json
import re
import numpy as np
import pandas as pd
from pyproj import Transformer,Geod

def verify(out,cfg):
    p=pd.read_csv(out/'point_labels.csv.gz',dtype={'user_id':str})
    a=pd.read_csv(out/'analysis_sample.csv.gz',dtype={'user_id':str})
    db=pd.read_csv(out/'dbscan_experiments.csv')
    h=pd.read_csv(out/'hotspots.csv')
    s=json.loads((out/'analysis_summary.json').read_text())
    assert p.point_id.is_unique and p.point_id.tolist()==a.point_id.tolist()
    assert np.array_equal(p[['x','y']].values,a[['x','y']].values)
    assert p.dbscan.ge(-1).all() and p.kmeans.ge(0).all()
    assert h.points.sum()==p.dbscan.ge(0).sum()
    assert len(db)==len(cfg['eps_m'])*len(cfg['min_samples'])
    assert db.n.eq(len(p)).all()
    assert np.allclose(db.coverage+db.noise_fraction,1)
    assert len(p)==s['points'] and p.user_id.nunique()==s['users']
    assert not p.duplicated(['trajectory_id','raw_index']).any()
    assert p.lon.between(cfg['bounds'][0],cfg['bounds'][1]).all()
    assert p.lat.between(cfg['bounds'][2],cfg['bounds'][3]).all()
    t=pd.to_datetime(p.time,utc=True)
    assert t.between(pd.Timestamp(cfg['valid_time_start']),pd.Timestamp(cfg['valid_time_end']),inclusive='left').all()
    assert p.speed_mps.dropna().le(cfg['jump_speed_mps']).all()
    trans=Transformer.from_crs(4326,32650,always_xy=True)
    x,y=trans.transform(p.lon.iloc[:10],p.lat.iloc[:10])
    assert np.allclose(x,p.x.iloc[:10]) and np.allclose(y,p.y.iloc[:10])
    x0,y0=trans.transform(116.3,39.9);x1,y1=trans.transform(116.301,39.9)
    geod=Geod(ellps='WGS84').inv(116.3,39.9,116.301,39.9)[2]
    assert abs(np.hypot(x1-x0,y1-y0)/geod-1)<.001
    text=(out/'实验报告.md').read_text(encoding='utf-8')
    assert f"{len(p):,}" in text and f"{int(p.dbscan.max()+1)} 个热点簇" in text
    figures=re.findall(r'!\[[^]]*\]\(([^)]+)\)',text)
    assert len(figures)>=8
    for name in figures: assert (out/name).is_file() and (out/name).stat().st_size>10000
    for name in ['raw_to_sample.csv.gz','quality_issues.csv.gz','trajectory_candidates.csv',
                 'common_sample_comparison.csv','case_studies.csv','user_comparison.csv']:
        assert (out/name).is_file()
    result={'status':'passed','points':len(p),'dbscan_grid_rows':len(db),
            'report_figures':len(figures),'checks':['sample identity','coordinate order and meter distances',
            'label and coverage accounting','time range','speed bounds','hotspot totals','report figures and key values']}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verification passed',flush=True)
