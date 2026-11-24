"""English report generated directly from measured result tables."""
import json
from pathlib import Path
import pandas as pd

def table(df):
    d=df.copy()
    for c in d.select_dtypes('number'):
        d[c]=d[c].map(lambda v:'NA' if pd.isna(v) else (f'{v:.4f}' if isinstance(v,float) else str(v)))
    d=d.fillna('NA').astype(str)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in d.values)

def report(out,cfg):
    read=lambda n:pd.read_csv(out/n,dtype={'user_id':str})
    j=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    inv=j('inventory_summary.json');s=j('analysis_summary.json');meta=j('run_metadata.json')
    db=read('dbscan_experiments.csv');km=read('kmeans_experiments.csv');hot=read('hotspots.csv')
    main=db[(db.eps_m==s['main_eps_m'])&(db.min_samples==s['main_min_samples'])].iloc[0]
    users=read('user_comparison.csv');cases=read('case_studies.csv')
    clean=read('cleaning_summary.csv');manifest=read('trajectory_manifest.csv')
    common=read('common_sample_comparison.csv');pilot=j('pilot.json')
    low=db[(db.eps_m==min(cfg['eps_m']))&(db.min_samples==20)].iloc[0]
    high=db[(db.eps_m==500)&(db.min_samples==5)].iloc[0]
    um=read('user_manifest.csv')
    chosen_users=users.sort_values(['points','user_id'],ascending=[False,True]).head(3)
    summary_hot=''
    hotspot_insight='There are too few hotspots to compare user coverage across the top four.'
    if len(hot)>=4:
        hotspot_insight=f"""The two largest hotspots include {int(hot.iloc[0].users)} and {int(hot.iloc[1].users)} users, respectively.
The fourth-largest hotspot, H{int(hot.iloc[3].dbscan)}, includes only {int(hot.iloc[3].users)} users.
High point density and broad user participation measure different properties. Consider user coverage when interpreting hotspots.
Density chaining can create kilometer-scale bounding boxes even with an eps of 50 m. Interpret eps as a neighborhood distance."""
    for row in hot.head(5).itertuples():
        summary_hot+=f'- H{row.dbscan}: center ({row.lon:.5f}, {row.lat:.5f}), {row.points:,} points, {row.users} users, {row.trajectories} trajectories, {row.days} local dates. Projected bounding box {row.width_m/1000:.2f} × {row.height_m/1000:.2f} km.\n'
    case_text=''
    for row in cases.itertuples():
        case_text+=f"""
### {'Candidate' if row.case_type=='candidate' else 'Low-noise control'} {row.trajectory_id}
Analysis period (UTC) {row.time_start} to {row.time_end}. The {row.points} analysis points had {row.noise_fraction:.1%} noise and represented {row.sampling_fraction:.1%} of the thinned points inside the region for this trajectory.
The maximum valid adjacent-edge speed was {row.max_valid_speed_mps:.2f} m/s. NA indicates an unavailable edge. The original trajectory had {row.quality_issue_rows} rows with quality issues.
Criteria: {'at least 20 analysis points and at least 80% noise' if row.case_type=='candidate' else 'at least 20 analysis points, at most 10% noise, and no recorded quality issues'}.
This result measures overlap with dense group activity. Unusual but reasonable trips, uneven coverage, and regional boundary truncation can all produce this pattern.
Investigate GPS drift when the source has quality issues. The gray lines in Figure 8 connect thinned points separated by up to 300 seconds. Driving behavior requires separate ground truth.
"""
    text=f"""# Project A Travel Hotspot Discovery

## 1 Summary

This experiment clusters spatial trajectory points from local Geolife 1.3 data. The full inventory has {inv['users']} users, {inv['trajectories']:,} trajectories, and {inv['points']:,} points. The analysis sample contains {s['points']:,} points from {s['trajectories']:,} trajectories and {s['users']} users.

The main DBSCAN model uses eps={s['main_eps_m']} m and min_samples={s['main_min_samples']} and produces {int(main.clusters)} hotspot clusters. Noise accounts for {main.noise_fraction:.2%} of all sample points, and the largest cluster accounts for {main.largest_cluster_fraction:.2%}. K-means compares K=2–10 on the same sample and the elbow rule selects K={s['kmeans_k']}. The low-density rule flags {s['candidate_count']} candidates without manual anomaly labels. The results describe activity density in this study sample.

## 2 Coursework and Scope

The experiment reviewed physical pages 8–10 of `Lecture1-logistics-DM2026V3-part1.pdf`. Page 8 lists grading. Pages 9–10 require Python and scikit-learn, Geolife loading and cleaning, feature extraction, DBSCAN tuning, comparison with K-means, hotspots across subjects, unusual trajectories, scatterplots, heatmaps, and a report.

Time analysis is an extension. Road-network and traffic-event identification are outside the required scope used here. The experiment selected 30 users, sample limits, a study rectangle, and thresholds. The course uses the term vehicles. Geolife also includes walking and cycling, so this report describes users and travel trajectories. The historical project acceptance crosswalk records the requirement mapping.

## 3 Data and Integrity

The source is Microsoft Geolife. The input file is `{meta.get('input_zip_name','Geolife Trajectories 1.3.zip')}` with SHA-256 `{meta.get('input_sha256','see run_metadata.json')}`. The original ZIP is read-only. It stores trajectories under `Geolife Trajectories 1.3/Data/<user_id>/Trajectory/<timestamp>.plt` and mode labels in user-level `labels.txt` files.

After six header lines, the parser reads latitude, longitude, a reserved field, altitude in feet, days since 1899-12-30, date, and time. Coordinates are WGS84. Altitude -777 indicates an invalid value. Convert feet to meters with 0.3048. Speed comes from adjacent points. The version 1.3 guide uses GMT for timestamps and mode labels. This workflow parses them as UTC and uses Asia/Shanghai (UTC+8) for calendar analysis.

| Inventory field | Observed value |
| --- | --- |
| Users, trajectories, and points | {inv['users']} / {inv['trajectories']:,} / {inv['points']:,} |
| Original time range including anomalies | {inv['time_start']} – {inv['time_end']} |
| Valid longitude and latitude ranges | Longitude {inv['lon_min']:.6f} to {inv['lon_max']:.6f}, latitude {inv['lat_min']:.6f} to {inv['lat_max']:.6f} |
| Invalid coordinate or time | {inv['invalid']:,} |
| Missing fields | {inv['missing_fields']:,} |
| Exact duplicate rows within a file | {inv['duplicates']:,} |
| Invalid altitude (-777) | {inv['altitude_invalid']:,} |
| Nonpositive interval in original order | {inv['nonpositive_dt']:,} |
| Original intervals over 300 seconds | {inv['gaps']:,} |
| Timestamps outside the documented collection period | {inv.get('time_outside_release','see inventory.csv')} |
| Users with mode label files | {inv['labels_users']} / {inv['users']} ({inv['labels_users']/inv['users']:.1%}) |
| Valid points within the study rectangle | {inv['region_points']:,} ({inv['region_points']/inv['points']:.1%} of all points) |

The guide's opening paragraph lists 17,621 trajectories and its version table lists 18,670. The local inventory matches the version table. The guide lists 73 users with labels and the local archive contains 69 label files. The table reports user-level file coverage. It does not estimate point-level labels, and the analysis does not filter by mode or draw motor-vehicle conclusions.

Some original timestamps predate 2007. The documented collection period was used for quality screening, excluding points before 2007-04-01 and at or after 2012-09-01. See `inventory.csv` for file-level intervals, including medians. A mean of file medians does not estimate the pooled point-level median.

## 4 Study Area, Cleaning, and Features

The inventory includes a spatial preview of up to 20 points per trajectory. The study rectangle is contiguous and spans longitude 116.1–116.7° and latitude 39.7–40.2° in Beijing. It contains {inv['region_points']/inv['points']:.1%} of the original points. The rectangle was chosen before clustering. The region is roughly 50 km across, so the projection uses meters in WGS84 / UTM 50N (EPSG:32650). `always_xy=True` fixes input order to longitude and latitude. Both models use x and y only. This projection serves local distance calculations.

Select the first 30 users by ID with at least 1,000 valid points in the region. For each user, order trajectories by `SHA256(seed:ZIP member name)` and retain up to 40 containing regional points. This selects {len(manifest):,} source files. After cleaning, sort each trajectory by time and retain its first point per UTC minute. With seed 20261020, randomly retain at most 2,000 points per user. Thinning leaves {int(um.thin_points.sum()):,} points, and final sampling leaves {s['points']:,}. User order, per-user caps, duration, and devices affect representativeness.

The workflow retains `user_id`, `trajectory_id`, `raw_index`, and `point_id` in the local full-run outputs. The raw data index starts at 0, and physical line number is `raw_index + 7`. `raw_to_sample.csv.gz` maps clean regional points to minute representatives and final point IDs. An empty point ID means no final selection. `quality_issues.csv.gz` records quality exclusions.

### 4.1 Cleaning Rules and Counts

Remove invalid coordinates or timestamps, times outside the collection period, exact duplicates over all seven fields, and conflicting records at the same timestamp. Retain the same location at different times. Record original order issues before sorting the points within each trajectory.

Compute spherical distance and speed only for adjacent original rows with positive intervals of at most 300 seconds. If speed exceeds 100 m/s, flag and exclude both endpoints of the possible GPS jump. Leave the broken edge disconnected. The threshold is a conservative data-quality rule and can exclude real high-speed trips. Compute distances within each trajectory and user. Count quality cleaning separately from region and sample filters.

{table(clean)}

![Study area and sample](figures/01_region.png)
![Cleaning and filtering counts](figures/02_cleaning.png)

### 4.2 Density Bias

Minute thinning reduces contributions from high-frequency devices and repeat points within the minute. Longer stays and repeated trips can still contribute more points. Compare normalized clean regional and minute-thinned counts in the same 500 m grid. The distributions have total variation distance {s['grid_bias_total_variation']:.4f}, with zero denoting identical distributions. See `density_bias_grid.csv`. Each hotspot reports distinct users, trajectories, and local dates. The workflow does not identify stop locations.

## 5 DBSCAN Method, Parameters, and Results

A core point has at least min_samples points within eps, including itself. Density-connected points form clusters. A point adjacent to a core point that has insufficient density itself is a border point. Other points receive noise label -1. The implementation uses Euclidean distances in meters and a kd-tree.

The pilot uses {pilot['n']:,} real points with maximum eps and minimum min_samples. Fit time was {pilot['seconds']:.3f} seconds and post-fit process RSS was {pilot['rss_mb_after']:.1f} MB. The pilot does not measure peak memory. The initial nine cases used 100, 200, and 500 m with 5, 10, and 20 points. At 200 m and 10 points the largest cluster held about 77% of the sample, so 30 m and 50 m were added. The final grid retains all 15 cases. Figure 9a shows the distance to the 20th neighbor including self. See `kdistance_quantiles.csv`.

Silhouette scores use metric distance and a fixed-seed sample of at most 2,000 non-noise points. Scores are NA when there are fewer than two clusters or all sample points have distinct clusters. The largest-cluster fraction uses all points as its denominator. Coverage equals one minus noise share. Fit time excludes silhouette scoring and plotting.

{table(db[['eps_m','min_samples','n','clusters','noise_fraction','coverage','largest_cluster_fraction','fit_seconds','silhouette','silhouette_n']])}

The main 50 m and 20-point combination retains more than half the points while reducing density chaining. It retains more points than 30 m and 20 and gives a smaller largest cluster than eps of 100 m or more. GPS error can still affect a 50 m neighborhood. This setting is an interpretability trade-off.

A 30 m and 20-point setting gives {int(low.clusters)} clusters and {low.noise_fraction:.1%} noise. At 500 m and five points, there are {int(high.clusters)} clusters, {high.noise_fraction:.1%} noise, and a largest-cluster share of {high.largest_cluster_fraction:.1%}. Use Figure 10 to inspect merged chains along routes. The main model's largest cluster accounts for {main.largest_cluster_fraction:.1%} and may still contain chained activity.

A random 90% subsample, clustered again, has ARI {s['stability']['ari_including_noise']:.4f} against the original labels on the same support, including noise. It gives {s['stability']['clusters']} clusters and {s['stability']['noise_fraction']:.1%} noise. This single perturbation is a local stability check. Large clusters and noise influence ARI.

![Main clustering](figures/03_dbscan.png)
![Parameter sensitivity](figures/04_sensitivity.png)
![Strict, intermediate, and loose configurations](figures/10_parameter_examples.png)

Colors separate structures, cycle when there are more than 20 clusters, and are local to each model. Use the point-level output for exact cluster IDs.

## 6 K-means and Common-Support Comparison

K-means alternates assignment and centroid updates to minimize the sum of squared Euclidean distances (SSE). It uses seed 20261020, k-means++ initialization, n_init=10, Lloyd iterations, and K=2–10. The reproducible elbow is the greatest downward deviation from the normalized K/SSE endpoint line, selecting K={s['kmeans_k']}. Silhouettes and spatial plots provide additional diagnostics.

The silhouette is {km.loc[km.k.eq(4),'silhouette'].iloc[0]:.4f} at K=4 and {km.loc[km.k.eq(5),'silhouette'].iloc[0]:.4f} at K=5. Increasing K from 4 to 5 reduces SSE by about {1-km.loc[km.k.eq(5),'inertia'].iloc[0]/km.loc[km.k.eq(4),'inertia'].iloc[0]:.1%}. K=5 is the finer elbow choice, and K=4 remains a coarser alternative.

{table(km)}

![Distance, elbow, and silhouette diagnostics](figures/09_diagnostics.png)
![Clustering on the same sample and coordinate range](figures/06_comparison.png)

DBSCAN does not require a preset K and can represent noise and irregular shapes. It is sensitive to density and parameters. K-means assigns every point to a cluster and favors compact partitions. It requires a specified K and is sensitive to outliers. DBSCAN suits recurring local concentrations, while K-means provides a preselected number of coarser partitions.

The full K-means and non-noise DBSCAN scores use different supports. The following comparison uses the same fixed evaluation points in the main DBSCAN non-noise support and retains each model's original labels.

{table(common)}

The methods have different cluster counts and objectives. These metrics describe internal structure. There are no ground-truth labels for accuracy, recall, or F1. See `hotspots.csv`, `kmeans_clusters.csv`, and the parameter tables for bounding boxes, cluster sizes, and execution cost. Bounding-box dimensions do not measure covered area.

## 7 Hotspots and User Differences

The five largest hotspots by analysis points are summarized below using coordinates and cluster IDs. The report does not infer a place, address, or identity.

{summary_hot}

{table(hot.head(10)[['dbscan','points','users','trajectories','days','lon','lat','width_m','height_m']])}

{hotspot_insight}

![500 m heatmap](figures/05_heatmap.png)

Select three users deterministically by descending point count and ascending user ID. Personal hotspots use the same eps and min_samples in separate fits. `user_hotspot_relation.csv` records each user's points in the fixed group hotspots. Sparse personal samples may yield more noise under the same absolute threshold.

{table(chosen_users)}

The three personal hotspot counts are {', '.join(str(v) for v in chosen_users.personal_clusters)}, and their covered group hotspot counts are {', '.join(str(v) for v in chosen_users.group_hotspots)}. The differences show how other users' contributions to shared areas affect density-based assignments.

![Individual hotspot plots](figures/07_users.png)

## 8 Low-Density Candidates and Controls

The configured rule flags trajectories with at least 20 analysis points and at least 80% DBSCAN noise. It finds {s['candidate_count']} candidates. Show up to three by descending noise fraction, descending point count, and ascending trajectory ID. Per-user sampling creates gaps, so the rule does not infer continuous noise intervals. Each case reports sample coverage and describes the cleaned, sampled portion within the region. Review recorded quality issues separately. Noise and low-noise controls have no manual anomaly or normal-behavior ground truth.

{case_text}

![Candidates and controls](figures/08_cases.png)

## 9 Time-of-Day Extension

Convert to UTC+8 and group into night 00–06, morning 06–10, daytime 10–17, and evening 17–24. Keep spatial hotspot labels fixed. Divide counts in the six largest hotspots by all points in the corresponding period, including noise. See `time_period_hotspots.csv` for full counts. Period lengths, users, and years differ, so the shares do not establish trip purpose, congestion severity, or causality.

![Hotspot shares by time of day](figures/11_time.png)

## 10 Uses, Limitations, and Conclusions

The method supports sample activity summaries, exploratory service-area selection, and trajectory-quality review. The study does not measure service gains or congestion reduction. The sample forms {int(main.clusters)} density-connected areas. Noise ranges from {low.noise_fraction:.1%} under the stricter setting to {high.noise_fraction:.1%} under the looser setting. Hotspot counts and boundaries depend on scale. The user comparison reports personal clusters and group overlap. The candidate rule screens low-density travel.

Limitations include nonrandom users, sampling frequency and duration, per-user caps, regional truncation, a common density threshold across users, positioning quality, missing labels and manual validation, and a single stability perturbation. Treat results as descriptive of the study sample.

## 11 Reproduction, Resources, and Verification

Python version is {meta['python']}. Dependency versions are `{json.dumps(meta['versions'],ensure_ascii=False)}`. The ZIP is {meta.get('input_zip_bytes',0)/2**20:.1f} MiB. See `config.json` for parameters and seeds.

Recorded stage times in seconds are `{json.dumps(meta.get('stage_seconds',{}),ensure_ascii=False)}`. These exclude subsequent reporting and verification. The first full run reads the entire ZIP again. The formal sample spans {s['time_start']} to {s['time_end']} UTC. Fit tables contain post-fit RSS. The original reproduction guidance recommends at least 4 GB available memory and about 1 GB of workspace. More data or larger eps can increase memory use. No map service is required.

See the README for the full-run entry point. Use `python run_original.py --stage verify --data <ZIP> --out <existing-full-run>` to check full-run sample consistency, labels, statistics, files, and report values. `python -m unittest discover -s tests -v` runs synthetic boundary tests. Road network matching, congestion and parking event detection, and motor-vehicle-specific analysis are outside this run. The full analysis writes a Markdown report and static PNGs.

## 12 References

1. Course slides, physical pages 8–10. See the English extraction in the public Evidence directory.
2. Geolife 1.3 ZIP User Guide, Version 1.3 (2012/08/01), sections 2 and 4 and the version table. The [Microsoft overview](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) includes older counts. This report uses the version 1.3 guide and local inventory.
3. [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), consulted 2026-10-20. KMeans showed 1.9.1, matching the local version. Some DBSCAN index entries showed 1.9.0. The local 1.9.1 interface and run were checked separately.
4. Zheng et al. Mining interesting locations and travel sequences from GPS trajectories. WWW 2009: 791–800.
5. Zheng et al. Understanding Mobility Based on GPS Data. UbiComp 2008: 312–321.
6. Zheng, Xie, Ma. GeoLife: A Collaborative Social Networking Service among User, location and trajectory. IEEE Data Engineering Bulletin 33(2), 2010: 32–40.
"""
    (out/'experiment_report.md').write_text(text,encoding='utf-8')
    print('Report generated from measured tables',flush=True)
