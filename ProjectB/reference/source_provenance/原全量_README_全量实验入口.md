# Project B 全量实验入口

根据2026-11-12最新授权，放宽实验准入并实际使用全部可获得的纽约项目数据。原范围为121条有向路段、双高峰07–09与17–19，数据自2026-05-11至11-11。54份分片共1019891条记录，包含新增15353条。无需等待完整7/28日、40%覆盖或最低事件数才进入实验。

全量时间网格537240个窗口，标签已知299988个，unknown237252个。unknown保留出分和覆盖。全量监督配方每轮遍历相应训练段的全部已知样本，负例不封顶。三种池化各3种子，包含中心、组内排序、步数匹配抽样对照及确定性基线。 [审计报告](审计报告.md)

| 导航 | 文件 |
|---|---|
| 登记与定义 | [计划](plan.md)、[方法与数据合同](方法与数据合同.md) |
| 数据与时间 | [快照和逐文件哈希](evidence/source_snapshot.json)、[全量分段](evidence/cohort_final.json)、[原始漏斗](evidence/prepared.json) |
| 完成状态 | [实验状态](evidence/experiment_status.json)、[选型](results/development_selection.json)、[阈值冻结](evidence/model_threshold_freeze.json) |
| 逐样本和模型 | results/models、results/decisions |
| 指标 | [逐种子工作点](results/operating_points.csv)、[分组](results/groups.csv)、[等召回](results/matched_recall.csv)、[每日误报](results/daily_errors.csv) |
| 审计与资源 | [测试报告](测试报告.md)、[核验](evidence/verification_replay.json)、[资源](results/training_resources.csv)、[变更](reports/变更说明.md) |
| 完整性 | [manifest.csv](manifest.csv)、[SHA256SUMS](SHA256SUMS) |

本轮使用先前已公开日期及11月9–11日记录，结果属于全量时间回溯实验。模型评估采用按日期先后划分的训练、校准、测试。新的未见独立日期应另立协议。

使用已验证的只读CPU环境 D:/conda_envs/niulaoshi/python.exe。依赖版本固定于project/requirements.txt，实际环境在logs/environment.json。所有命令输出放在包外，原数据和已有审计包只读使用。

```powershell
$fpy = 'D:/conda_envs/niulaoshi/python.exe'
$fbundle = (Get-Location).Path
$fwork = 'G:/gpt-file/Codex/work/ProjectB_全量复算'

# 包内逐样本分数和阈值复算，无需下载大数据
& $fpy -B -X utf8 ./project/full_pack.py verify-dir --output $fbundle
& $fpy -B -X utf8 ./project/run_full.py --phase verify --artifacts $fbundle --output "$fwork/offline"
& $fpy -B -X utf8 ./project/run_full.py --phase tests --output "$fwork/tests"

# 使用快照中已经哈希验证的外部原始分片，一键重建、训练、校准、评价
& $fpy -B -X utf8 ./project/run_full.py --phase all --output "$fwork/rerun"

# 分阶段运行
& $fpy -B -X utf8 ./project/run_full.py --phase initialize --output "$fwork/staged"
foreach ($fphase in 'check','fetch','prepare','tests','development','select','final','freeze','evaluate','analysis','verify','report') {
    & $fpy -B -X utf8 ./project/run_full.py --phase $fphase --output "$fwork/staged"
    if ($LASTEXITCODE -ne 0) { throw "Full experiment phase $fphase failed" }
}
```

默认fetch附接协议目录中已固定的source_snapshot，并逐文件核验。外部原始分片路径在快照中，可通过--source-snapshot指定已映射的快照路径。没有快照时才访问官方接口取得当前续期，新的快照应另建运行目录。大体积原始CSV和完整特征数组不随包复制，缺文件时记录具体缺口。

阶段prepare重建两份development/final特征数组。已经重建时可重放实际模型。

```powershell
& $fpy -B -X utf8 ./project/run_full.py --phase verify --artifacts $fbundle --replay-root "$fwork/rerun" --output "$fwork/replay"
```

本地实际进行了源码、测试、数据重建及模型重放，干净目录的实际检查结果在压缩包旁解压复核收据中。个别未做的重复训练按收据如实记录。每模型训练ID、每轮唯一访问数、更新步数、权重和预处理器均保留，便于验证“全量”。
