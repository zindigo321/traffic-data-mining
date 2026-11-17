"""Synthetic fixtures only test code; they never enter the formal experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run  # Sets up Windows native DLL paths before numeric imports.
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from src.data import clean_trajectory,COLS
from src.analysis import score

CFG=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())

def fixture(seconds,lons=None):
    n=len(seconds)
    t=pd.Timestamp('2008-01-01',tz='UTC')+pd.to_timedelta(seconds,unit='s')
    return pd.DataFrame(dict(lat=np.full(n,39.9),lon=lons if lons is not None else np.full(n,116.3),
        reserved=np.zeros(n),alt_ft=np.full(n,-777),day=np.asarray(seconds)/86400+39448,
        date=t.strftime('%Y-%m-%d'),clock=t.strftime('%H:%M:%S'),raw_index=np.arange(n),time=t))

class Checks(unittest.TestCase):
    def test_gap_not_speed(self):
        a,_=clean_trajectory(fixture([0,600],[116.3,117.3]),CFG)
        self.assertTrue(a.speed_mps.isna().all())
    def test_same_place_different_time(self):
        a,_=clean_trajectory(fixture([0,60,120]),CFG)
        self.assertEqual(len(a),3)
        self.assertTrue(a.alt_m.isna().all())
    def test_exact_duplicates(self):
        d=fixture([0,0,60]);a,q=clean_trajectory(d,CFG)
        self.assertEqual(len(a),2)
        self.assertEqual(q.reason.eq('exact_duplicate').sum(),1)
    def test_timestamp_conflict(self):
        a,q=clean_trajectory(fixture([0,0,60],[116.3,116.4,116.3]),CFG)
        self.assertEqual(q.reason.eq('timestamp_conflict').sum(),2)
        self.assertEqual(len(a),1)
    def test_jump_both_endpoints(self):
        a,q=clean_trajectory(fixture([0,1,2],[116.3,117.3,116.3]),CFG)
        self.assertEqual(q.reason.eq('jump_edge_endpoint').sum(),3)
        self.assertTrue(a.empty)
    def test_time_range(self):
        d=fixture([0]);d['time']=pd.Timestamp('2000-01-01',tz='UTC')
        a,q=clean_trajectory(d,CFG)
        self.assertTrue(a.empty)
        self.assertEqual(q.reason.iloc[0],'time_outside_release')
    def test_sort_and_no_reconnect(self):
        a,_=clean_trajectory(fixture([120,0,60]),CFG)
        self.assertTrue(a.time.is_monotonic_increasing)
        self.assertTrue(pd.isna(a.speed_mps.iloc[-1]))
    def test_empty_single_all_noise(self):
        for X,y in [(np.empty((0,2)),np.array([])),(np.zeros((3,2)),np.zeros(3))]:
            self.assertTrue(np.isnan(score(X,y,100,42)[0]))
        X=np.array([[0,0],[1000,0],[2000,0]])
        y=DBSCAN(eps=1,min_samples=2).fit_predict(X)
        self.assertTrue((y==-1).all())
        self.assertTrue(np.isnan(score(X[y>=0],y[y>=0],100,42)[0]))
    def test_bin_independent_of_datetime_storage_unit(self):
        d=fixture([0,59,60,119,120])
        self.assertEqual(d.time.dt.floor('60s').nunique(),3)

if __name__=='__main__': unittest.main()
