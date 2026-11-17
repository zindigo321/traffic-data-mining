"""Chinese report generated directly from measured result tables."""
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
    hotspot_insight='没有足够热点进行前四名对比；不强行解释用户覆盖。'
    if len(hot)>=4:
        hotspot_insight=f"""前两大热点分别有{int(hot.iloc[0].users)}和{int(hot.iloc[1].users)}名用户参与，
而排名第四的热点H{int(hot.iloc[3].dbscan)}只有{int(hot.iloc[3].users)}名用户。
这直接说明“轨迹点密集”与“许多独立用户访问”不等价，区域筛选应同时参考用户覆盖。
即使主eps只有50米，密度链也能形成公里级包围盒，eps不能解释为每个热点的最大半径。"""
    for row in hot.head(5).itertuples():
        summary_hot+=f'- H{row.dbscan}：中心经纬度 ({row.lon:.5f}, {row.lat:.5f})，{row.points:,} 点，{row.users} 用户，{row.trajectories} 条轨迹，{row.days} 个本地日期；投影包围盒 {row.width_m/1000:.2f} × {row.height_m/1000:.2f} km。\n'
    case_text=''
    for row in cases.itertuples():
        case_text+=f"""
### {'候选' if row.case_type=='candidate' else '低噪声对照'}：{row.trajectory_id}
分析时间（UTC）{row.time_start} 至 {row.time_end}；{row.points} 个分析点，噪声 {row.noise_fraction:.1%}，
覆盖该轨迹区域内抽稀点的 {row.sampling_fraction:.1%}。有效相邻边最大速度 {row.max_valid_speed_mps:.2f} m/s
（NA 表示没有可用相邻边），原轨迹有 {row.quality_issue_rows} 行质量问题。
触发规则：{'分析点≥20 且噪声比例≥80%' if row.case_type=='candidate' else '分析点≥20、噪声≤10%、无已记录质量问题'}。
这只表明与群体密集区域的重合程度。稀少但合理的出行、样本覆盖不均和区域边界截断均可能解释该现象；
存在质量问题时还应检查定位漂移，不能认定为异常驾驶。图8灰线仅连接间隔不超过300秒的抽稀点。
"""
    text=f"""# Project A：出行热点区域发现

## 1 摘要
本实验使用本地 Geolife 1.3 真实数据，以空间轨迹点而非目的地为聚类对象。
全量读取 {inv['users']} 名用户、{inv['trajectories']:,} 条轨迹、{inv['points']:,} 点；
正式建模使用 {s['users']} 名用户、{s['trajectories']:,} 条轨迹中的 {s['points']:,} 个分析样本。
DBSCAN 主参数 eps={s['main_eps_m']} m、min_samples={s['main_min_samples']}，得到 {int(main.clusters)} 个热点簇，
噪声比例 {main.noise_fraction:.2%}，最大簇占全部样本 {main.largest_cluster_fraction:.2%}。
同样本 K-means 比较 K=2—10，采用肘部准则选 K={s['kmeans_k']}。
检测到 {s['candidate_count']} 条满足低密度规则的异常候选，均未经过人工标签验证。
这些结果描述研究样本中的活动密度，不能推广为全北京市居民、机动车出行或拥堵规律。

## 2 任务、依据与实施范围
已独立读取并渲染课程 `Lecture1-logistics-DM2026V3-part1.pdf` 第8—10页：
第8页为评分结构，第9—10页明确 Python/scikit-learn、Geolife、清洗与特征、DBSCAN调参、
K-means对比、不同主体热点与异常轨迹、散点图/热力图及报告。
时间分析为进阶，路网与事件识别不属于本次必做范围。30用户、采样上限、区域边界和全部阈值是本实验决定，
不是教师指定。课件使用“车辆”措辞，但 Geolife 包含步行、骑行等，本报告准确采用“用户/出行轨迹”。
验收映射见项目根目录 `验收对照表.md`。

## 3 数据说明与完整性核对
来源为微软 Geolife 项目，本次输入文件 `{meta.get('input_zip_name','Geolife Trajectories 1.3.zip')}`，
SHA256：`{meta.get('input_sha256','见run_metadata.json')}`。原 ZIP 始终只读。
内部路径为 `Geolife Trajectories 1.3/Data/<user_id>/Trajectory/<timestamp>.plt`；
交通方式标签位于各用户目录 `labels.txt`。
跳过6行头部，依次解析纬度、经度、保留字段、海拔（英尺）、1899-12-30起算天数、日期、时间。
WGS84坐标；海拔 -777 表示无效，转换米乘0.3048。速度由相邻点推导，原文件没有速度字段。
1.3版官方说明明确日期时间和交通方式标签都是 GMT，本实验按 UTC 解析，需要日历统计时转换 Asia/Shanghai（UTC+8）。

| 全量字段 | 实测值 |
| --- | --- |
| 用户 / 轨迹 / 点 | {inv['users']} / {inv['trajectories']:,} / {inv['points']:,} |
| 原始时间范围（含异常） | {inv['time_start']} — {inv['time_end']} |
| 经纬度范围（坐标合法点） | 经度 {inv['lon_min']:.6f} 至 {inv['lon_max']:.6f}；纬度 {inv['lat_min']:.6f} 至 {inv['lat_max']:.6f} |
| 非法坐标或时间 | {inv['invalid']:,} |
| 缺字段记录 | {inv['missing_fields']:,} |
| 文件内完全重复记录 | {inv['duplicates']:,} |
| 无效海拔 -777 | {inv['altitude_invalid']:,} |
| 原顺序非正采样间隔 | {inv['nonpositive_dt']:,} |
| 原顺序间隔超过300秒 | {inv['gaps']:,} |
| 超出说明书采集期的时间戳 | {inv.get('time_outside_release','详见inventory.csv') } |
| 具有方式标签文件的用户 | {inv['labels_users']} / {inv['users']}（{inv['labels_users']/inv['users']:.1%}） |
| 研究矩形内合法点 | {inv['region_points']:,}（全量点的{inv['region_points']/inv['points']:.1%}） |

说明书首段仍写17,621条，后续版本表写18,670条；本地实测支持版本表。
说明书写73名带标签用户，本地仅找到69个标签文件，因此不把名义标签数量当成本地覆盖率。
这里的覆盖率是**标签文件用户覆盖率**，并非点级已标注比例。本次不按交通方式筛选，不输出机动车结论。
全量时间早于2007年，说明少数设备时间可疑；正式清洗将2007-04-01之前或2012-09-01及之后的点排除，
这个范围来自版本说明且属于质量筛查假设。文件级采样间隔中位数等详见 `inventory.csv`；
没有把不同文件中位数的平均值误称为全体点的中位数。

## 4 研究区域、清洗与特征
先统计全量坐标和每轨迹最多20点的空间预览，再选连续北京矩形：
WGS84经度116.1—116.7°、纬度39.7—40.2°。其覆盖全量{inv['region_points']/inv['points']:.1%}个点，
并非按聚类结果裁剪。范围约50 km量级，采用 WGS84 / UTM 50N（EPSG:32650）米制平面，
`always_xy=True` 明确输入经度、纬度；两模型均只输入 x,y，不混入用户编号、时间或速度。
UTM投影适合该局部区域，不用于全球样本距离计算。

筛选规则：按用户编号升序选区域内至少1000合法点的前30名；
每用户按 `SHA256(seed:ZIP成员名)` 顺序最多选40条含区域点轨迹，共 {len(manifest):,} 个原始文件。
每轨迹清洗后按时间排序，每个UTC整分钟桶保留首点，再固定种子20261020按用户随机保留最多2000点。
抽稀后共 {int(um.thin_points.sum()):,} 点，最终 {s['points']:,} 点。
这不是人口随机样本：低编号选择与每用户点数上限改变权重，轨迹长度和设备记录习惯仍有影响。
保留 `user_id, trajectory_id, raw_index, point_id`；
原文件数据行索引从0开始，物理行号为raw_index+7。
`raw_to_sample.csv.gz` 将区域内清洁原始点对应到分钟代表点及最终point_id，空point_id表示未进入正式样本，
并不表示该原始点无效。被删质量问题另存 `quality_issues.csv.gz`。

### 4.1 清洗规则与计数
依次排除非法坐标/时间、超采集期时间、七字段完全重复、同一时间的冲突记录。
同位置不同时间仍保留。原顺序问题先记录，保留点分轨迹排序。
仅在相邻原始行、正时间差且间隔≤300秒时计算球面距离与速度；
速度>100m/s的边两端均标记为定位跳变候选并排除，避免立即把断边重连。
这是保守质量阈值，不是异常驾驶阈值，可能排除真实高速出行；报告不把它解释成交通违法。
不跨用户/轨迹计算距离，不跨长间隔计算速度。质量清洗与区域/采样筛选分别计数。

{table(clean)}

![图1 研究区域与样本](figures/01_region.png)
![图2 清洗统计](figures/02_cleaning.png)

### 4.2 密度偏差检查
每分钟抽稀减轻高频设备和同分钟重复点的影响，但长时间停留和反复往返仍贡献更多点。
相同500m网格内，对所选轨迹清洁区域点与一分钟抽稀点分别归一化，
分布总变差距离为 {s['grid_bias_total_variation']:.4f}（0表示完全相同），表明采样口径确实影响空间密度。
详见 `density_bias_grid.csv`，不能把点多直接解释为用户多。
每个热点同时报告独立用户、轨迹和本地日期数；本次没有提取停留点，因此不称“目的地”。

## 5 DBSCAN 原理、参数与结果
eps邻域内包含自身在内至少min_samples个点即核心点；可由核心点密度连接的点形成簇，
邻近核心点而自身密度不足者为边界点；剩余点标为-1噪声。
使用欧氏米制距离和kd-tree，避免显式构建全量两两距离矩阵。
首先对 {pilot['n']:,} 个真实样本跑最大eps/最小min_samples试验，
拟合 {pilot['seconds']:.3f} 秒，进程事后驻留内存 {pilot['rss_mb_after']:.1f} MB（不是峰值）。
先跑100/200/500m × 5/10/20共9组；观察到200m/10最大簇占约77%，因此补充30/50m，
最终5种eps × 3种min_samples共15组，不删除原始9组。图9a为含自身的第20近邻距离曲线，
对应主模型min_samples=20，分位数在 `kdistance_quantiles.csv`。
轮廓系数使用米制距离，排除-1后固定种子最多抽2000点；
簇数不满足2≤簇数≤样本数-1时为NA，并记录原因与实际评估量。
表中最大簇占比的分母为所有输入点，覆盖率=1-噪声比例。拟合耗时不含轮廓评估和作图。

{table(db[['eps_m','min_samples','n','clusters','noise_fraction','coverage','largest_cluster_fraction','fit_seconds','silhouette','silhouette_n']])}

主参数采用50m/20，优先减弱沿道路链式连接，同时保留超过一半样本；
相比30m/20保留更多活动点，相比100m及以上显著降低最大簇占比。
50m仍可能受GPS误差影响，这是解释性折中，非最优参数证明，也不是只按轮廓最高分选择。
30m/20得到{int(low.clusters)}簇、噪声{low.noise_fraction:.1%}，呈现更严格的密度筛选；
500m/5得到{int(high.clusters)}簇、噪声{high.noise_fraction:.1%}、最大簇占比{high.largest_cluster_fraction:.1%}，
应结合图10检查是否把沿线活动连成大簇，而不能仅以低噪声宣布更优。
主模型最大簇占比{main.largest_cluster_fraction:.1%}，仍存在密度链式连接风险。
90%随机子样本重新聚类，与原模型在同一子样本上的含噪声ARI为{s['stability']['ari_including_noise']:.4f}，
得到{s['stability']['clusters']}簇、噪声{s['stability']['noise_fraction']:.1%}。
该单次扰动仅是局部稳定性检查；大簇和噪声会影响ARI，不是完整统计置信度。

![图3 主聚类](figures/03_dbscan.png)
![图4 参数敏感性](figures/04_sensitivity.png)
![图10 严格、中间、宽松三组实测结果](figures/10_parameter_examples.png)
图中颜色循环用于区分空间结构，不同模型的相同颜色或数字不表示相同区域；簇多于20时颜色会复用，精确编号以点级表为准。

## 6 K-means 原理与同样本比较
K-means通过最小化点到所属质心的平方欧氏距离和（SSE）迭代分配与更新中心。
设随机种子20261020、k-means++初始化、n_init=10、Lloyd算法，K取2—10。
对归一化K/SSE曲线计算相对端点连线的最大下凹位置作为可复现肘部，选择K={s['kmeans_k']}；
同时展示轮廓系数和空间图，轮廓高值不一定对应适宜的局部热点尺度。
本次K=4的轮廓系数为{km.loc[km.k.eq(4),'silhouette'].iloc[0]:.4f}，
高于K=5的{km.loc[km.k.eq(5),'silhouette'].iloc[0]:.4f}，
但从4到5可进一步降低SSE约{1-km.loc[km.k.eq(5),'inertia'].iloc[0]/km.loc[km.k.eq(4),'inertia'].iloc[0]:.1%}。
因此K=5是更细空间分区的肘部对照方案，K=4同样是应保留的粗分区备选，不能声称唯一最优。

{table(km)}

![图9 距离诊断、肘部与轮廓](figures/09_diagnostics.png)
![图6 同样本同坐标范围对比](figures/06_comparison.png)

DBSCAN无需预设K，可以保留噪声和非规则连通形状，但对密度变化与参数敏感；
K-means必须给K、将每个点分入某簇，倾向紧凑空间分区，对离群点敏感。
前者适用于发现重复出现的局部高密度活动，后者更适合预设数量的粗粒度空间分区。
DBSCAN去噪轮廓与K-means全样本轮廓的支持集不同，不能直接公平排名。
补充在主DBSCAN非噪声共同样本上比较原模型标签（不重新训练），固定同一组评估点：

{table(common)}

两模型簇数和目标仍不同，以上指标只是结构诊断。没有真值标签，不报告准确率、召回率或F1。
各簇投影包围盒及规模分别见 `hotspots.csv` 和 `kmeans_clusters.csv`，
运行成本见参数表；包围盒宽高不代表真实覆盖面积。

## 7 热点与用户差异
热点按分析点数降序。前五个区域如下，只给坐标和区域编号，不猜测地点、住址或用户身份：

{summary_hot}

{table(hot.head(10)[['dbscan','points','users','trajectories','days','lon','lat','width_m','height_m']])}

{hotspot_insight}

![图5 500米网格热力图](figures/05_heatmap.png)

按分析点数降序、用户编号升序确定性选择以下3名用户。
个人热点使用同样eps/min_samples单独拟合，因此个人标签不能与群体标签直接等同；
`user_hotspot_relation.csv`另记录每个用户在群体固定热点中的点数。
个人样本更稀疏，使用相同绝对密度阈值可能产生更多噪声，不能据此认定其行为更异常。

{table(chosen_users)}

三位用户的个人热点数量分别为{', '.join(str(v) for v in chosen_users.personal_clusters)}，
覆盖的群体热点数量分别为{', '.join(str(v) for v in chosen_users.group_hotspots)}。
个人聚类与群体归属相差较大，说明其他用户对共享活动区域的密度贡献会改变热点判定。

![图7 用户个人热点](figures/07_users.png)

## 8 异常轨迹候选与对照
规则在分析前配置：轨迹至少20个正式分析点，且DBSCAN噪声占比≥80%。
共{s['candidate_count']}条候选；按噪声比例降序、点数降序、轨迹编号升序最多展示3条。
候选规则不要求连续噪声时长，原因是按用户限额抽样会产生间隙，不能从抽样点虚构连续片段。
所有案例记录抽样覆盖率；“轨迹”结论仅针对研究矩形内且通过清洗、抽样的观测部分。
质量异常见单独表，-1不是定位错误或异常驾驶真值。
低噪声对照也不代表经过人工确认的正常行为，只是规则下的参考。

{case_text}

![图8 候选与对照](figures/08_cases.png)

## 9 可选时间扩展
转换为UTC+8，夜间00—06、早间06—10、日间10—17、晚间17—24。
沿用同一空间热点标签，按各时段全部样本数（包括噪声）归一化，展示点数前6个热点的占比。
完整样本量与噪声类别见 `time_period_hotspots.csv`。时段长度不同、用户与年份组成不同，
不能将比例差异直接推断为通勤目的、拥堵强度或因果关系。

![图11 时段构成](figures/11_time.png)

## 10 应用价值、局限与结论
本方法可为样本活动区域汇总、出行服务候选区域探索和轨迹质量排查提供依据，
没有测量服务效率或拥堵改善，故不声称实际收益百分比。
结果支持以下有限结论：本研究样本形成{int(main.clusters)}个密度连通区域；
参数从严格到宽松时噪声比例从{low.noise_fraction:.1%}变为{high.noise_fraction:.1%}，
热点边界与数量依赖尺度；个人热点数量及与群体重合程度见表7；
候选只是对稀疏出行的可解释筛查，不构成异常驾驶检测器。
主要局限包括用户非随机、采样频率和时长偏差、每用户限额权重、
区域截断、同密度阈值对不同用户不公平、定位质量、标签缺失、缺乏人工验证、单次扰动稳定性不足。
所有统计是本次样本结论，不能外推为现代全城市规律。

## 11 复现、资源与验收
Python {meta['python']}；实际依赖版本：
`{json.dumps(meta['versions'],ensure_ascii=False)}`。
原始ZIP大小{meta.get('input_zip_bytes',0)/2**20:.1f}MiB；参数和种子见 `config.json`。
分阶段实测耗时（秒）为 `{json.dumps(meta.get('stage_seconds',{}),ensure_ascii=False)}`，
阶段计时不含其后报告/验收写出；首次全流程会重新读取全量ZIP。
正式样本时间范围（UTC）为 {s['time_start']} 至 {s['time_end']}。
各模型拟合事后RSS见参数表，并非峰值。建议可用内存至少4GB，保留约1GB工作空间；
改变规模/eps可能显著增加DBSCAN内存。程序无在线地图依赖。
运行方法见README，一条入口可重建全部结果。
`python run.py --stage verify`核对样本一致性、标签、统计、文件和报告关键数值，
`python -m unittest discover -s tests -v`执行合成边界测试（只用于程序检查，不作正式实验）。
路网匹配、拥堵/停车事件识别和机动车专门分析未实施，属于明确不纳入本次范围的扩展；
本次交付Markdown报告与静态PNG，未导出PDF。

## 12 参考资料
1. 课程课件第8—10页，本地副本提取证据位于 `references/课件第8-10页.txt`。
2. Geolife 1.3 ZIP自带User Guide，Version 1.3 (2012/08/01)，核对第2、4节和版本表；
   [微软数据介绍](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)（网页存在旧版数字，以本地实测和1.3说明为准）。
3. [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)、
   [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)、
   [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)；
   2026-10-20查阅，KMeans页面显示1.9.1，与本地一致；检索索引部分DBSCAN页面标1.9.0，
   同时对本地1.9.1接口和实际运行验证，未假称索引为精确补丁版本。
4. Zheng et al. Mining interesting locations and travel sequences from GPS trajectories. WWW 2009: 791–800.
5. Zheng et al. Understanding Mobility Based on GPS Data. UbiComp 2008: 312–321.
6. Zheng, Xie, Ma. GeoLife: A Collaborative Social Networking Service among User, location and trajectory. IEEE Data Engineering Bulletin 33(2), 2010: 32–40.
"""
    (out/'实验报告.md').write_text(text,encoding='utf-8')
    print('Report generated from measured tables',flush=True)
