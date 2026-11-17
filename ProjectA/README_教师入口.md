# Project A 教师提交入口

项目：Geolife出行热点发现。代码与正式聚类结果来自现有Project A；本次副本按隐私原则移除逐点/逐人精确轨迹，仅提交聚合结果与安全图表。

推荐使用现有niulaoshi环境，不安装或修改环境。在此目录：

```bat
conda activate niulaoshi
python run.py
python run.py tests
```

默认离线校验不依赖原数据ZIP。它复算聚合总量、覆盖/噪声、KMeans肘部、共同支持、密度偏差与时间占比，检查图和隐私字段，不将这些检查称为全量重训。

需要从原始数据完整重跑时：

```bat
python run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\ProjectA_runs\new_run"
```

必须选择不存在的新目录，保留当前结果。源数据需由复核人自行合法获取，原始ZIP保持只读。本包不分发个人精确轨迹或原始压缩包；完整重跑会在本地生成用于复核的点级结果，不应直接另行公开。`run_original.py`保留原流程，默认历史路径仅为原代码记录；推荐只经本包 `run.py full` 显式传入数据。

阅读顺序：results/实验报告.md、analysis_summary.json、dbscan_experiments.csv、common_sample_comparison.csv、evidence/。points=53,786，DBSCAN=174簇，KMeans=5簇，候选115；ARI与轮廓系数不是分类准确率。

清单说明：evidence/original_checksums.sha256仅保存原清单，原清单含缺失引用并与当前多项文件不符。新的整体提交清单由总期末包生成，依据本次实际文件；来源哈希和脱敏变换见 evidence/source_provenance.json。
