# Project B 动态边权实验交付

真实 NYC 全量时间回测已完成：18 次新动态训练，18 个已核验静态模型配对复用。主模型 dynamic_mean；AP=0.6664 ± 0.0243，正类 F1=0.6501 ± 0.0195，达到回测数值目标。总FP/日=14.600，恢复FP/日=7.504。误报预算在回测是否成立见审计报告，数值达标不等于预算达标。

原测试已公开，这是回测消融；新独立盲测尚未完成。Geolife 与 Chicago 原口径继续 blocked。本轮沿用已有纽约快照，没有修改标签、剔除困难测试窗口或降低准确率来凑数。

| 配方 | AP（均值±样本SD） | 正类 F1（均值±样本SD） | 平衡准确率 | 召回 | 总FP/日 | 恢复FP/日 |
| --- | --- | --- | --- | --- | --- | --- |
| graph_mean | 0.6553 ± 0.0348 | 0.6455 ± 0.0114 | 0.7892 | 0.5870 | 14.756 | 7.578 |
| graph_max | 0.6591 ± 0.0262 | 0.6551 ± 0.0158 | 0.7955 | 0.5994 | 14.652 | 7.630 |
| graph_attention | 0.6638 ± 0.0332 | 0.6388 ± 0.0196 | 0.7832 | 0.5747 | 14.289 | 7.459 |
| dynamic_mean | 0.6664 ± 0.0243 | 0.6501 ± 0.0195 | 0.7919 | 0.5922 | 14.600 | 7.504 |
| dynamic_max | 0.6591 ± 0.0144 | 0.6440 ± 0.0053 | 0.7833 | 0.5743 | 13.341 | 6.852 |
| dynamic_attention | 0.6687 ± 0.0134 | 0.6541 ± 0.0079 | 0.7941 | 0.5965 | 14.489 | 7.348 |


阅读顺序：审计报告.md → 方法与数据合同.md → plan.md → results/operating_points.csv、matched_recall.csv、paired_date_blocks.csv → evidence/dynamic_threshold_freeze.json → logs/ → 测试报告.md。

代码：project/dynamic_model.py（有界边权），dynamic_train.py（全量训练），dynamic_score.py（开发选择/校准冻结/回测），dynamic_verify.py（独立复算与安全重放），run_dynamic.py（一键入口）。配置 project/config.json；原输入预处理模块保留在 project/full_prepare.py 及继承代码内。

```powershell
$python = 'D:/conda_envs/niulaoshi/python.exe'
# 在解压审计包根目录执行：仅依赖包内预测，独立重算指标与阈值
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_verify_new --no-replay
# 加上外部已核验 prepared 数组，抽样重放全部 36 个模型
& $python -B -X utf8 project/run_dynamic.py --phase verify --artifacts (Get-Location).Path --output G:/gpt-file/Codex/work/ProjectB_dynamic_replay_new
# 新目录完整重跑 18 次动态训练；静态 18 个模型仍从原审计包核验后复用
& $python -B -X utf8 project/run_dynamic.py --phase all --output G:/gpt-file/Codex/work/ProjectB_dynamic_refit_new
```
输出目录必须选择尚未使用的名称。分阶段命令将 `--phase all` 替换为 check、tests、development、select、final、freeze、evaluate、analysis、verify、report；初始化资料通过完整入口复制。不能用 check 覆盖既有注册收据。


离线指标复算只需包内预测及原 Python 依赖；重放还需 evidence/prepared.json 中两份只读数组；完整新训练还需原全量审计包及其实际原始快照路径。原始公开大表、私人 Geolife 轨迹和旧巨大压缩包未复制。先核验文件哈希后才能重放。依赖实际版本见 logs/environment_dynamic.json 和 project/requirements_actual.txt。

清单 manifest.csv 枚举除自身和 SHA256SUMS 外所有文件；SHA256SUMS 包含 manifest，不包含自身。解压复核收据及 ZIP 整体哈希放在交付目录外层，避免自指循环。
