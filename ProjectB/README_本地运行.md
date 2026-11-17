# Project B Graph Transformer — niulaoshi 完整运行包

这是包含真实公开数据、完整 Python 源码、预处理器、模型权重和参考结果的本地运行包。模型为有向路段 Graph Transformer，含 Mean、Max、Attention 三池化和动态边权。默认 CPU。原有 Conda 环境不变，包内程序不安装依赖、不下载数据。包可解压到任意可写路径，包含空格的路径也支持。

## 最短运行方法

用 Anaconda Prompt 进入解压后的 `ProjectB_GT_niulaoshi` 文件夹：

```bat
conda activate niulaoshi
python run.py check
python run.py
```

`python run.py` 默认独立复算参考结果，成功会显示 `Exit: 0`。每次输出新建在 `runs/时间戳_命令/`，日志和退出码保存在其中。也可以运行 `02_verify.cmd`，它会优先使用已激活环境，否则调用 `conda run -n niulaoshi`；本机解释器已核验为 `D:\conda_envs\niulaoshi\python.exe`。显式运行示例：

```bat
D:\conda_envs\niulaoshi\python.exe -B -X utf8 run.py replay
```

在 PyCharm / VS Code 中选择 niulaoshi 的解释器，运行根目录 `run.py`；参数留空是指标复算，参数填 `train` 是完整重训。代码根据自身位置找数据，不依赖 IDE 的工作目录。

## 八种运行命令

| 命令 | 实际用途 |
| --- | --- |
| `python run.py check` | 校验环境、包内清单、54 个原始分片、道路几何和两份数据数组 |
| `python run.py verify` | 从逐样本分数独立重算 AP、F1、混淆、分组、等召回点和阈值 |
| `python run.py replay` | 使用包内数据、预处理器和 weights_only 权重抽样重放全部 36 个模型 |
| `python run.py tests` | 运行原有及动态边权 75 项测试，再运行 6 项路径/环境/完整性测试，共 81 项 |
| `python run.py smoke` | 在 32 条真实训练样本上完成三池化的反向、优化、保存和加载；仅工程联调 |
| `python run.py prepare` | 从包内 54 个原始 CSV 和道路几何重建图、标签与特征，并校验数组字节哈希 |
| `python run.py predict --pool mean --seed 11` | 导出冻结模型的 130,680 个历史测试窗预测，含未知标签窗口 |
| `python run.py train` | 按原协议完整重训三池化×三种子×开发/最终，共 18 次动态训练；复用已核验的 18 个静态对照 |

完整训练与 smoke 不同；smoke 不产生正式实验成绩。完整训练仍保持 12 轮、batch 128、2 线程/进程、最多 4 个任务、种子 11/22/33。训练耗时取决于 CPU 和系统负载，请以实际日志为准；建议可用内存至少 8 GB，另留 2 GB 磁盘给新结果。

指定独立输出目录：

```bat
python run.py train --output "D:\ProjectB_runs\dynamic_new"
```

分阶段运行（输出目录必须保持相同；不要对已完成目录再次 initialize）：

```bat
python run.py train --phase initialize --output "D:\ProjectB_runs\stage_run"
python run.py train --phase tests --output "D:\ProjectB_runs\stage_run"
python run.py train --phase development --output "D:\ProjectB_runs\stage_run"
python run.py train --phase select --output "D:\ProjectB_runs\stage_run"
python run.py train --phase final --output "D:\ProjectB_runs\stage_run"
python run.py train --phase freeze --output "D:\ProjectB_runs\stage_run"
python run.py train --phase evaluate --output "D:\ProjectB_runs\stage_run"
python run.py train --phase analysis --output "D:\ProjectB_runs\stage_run"
python run.py train --phase verify --output "D:\ProjectB_runs\stage_run"
python run.py train --phase report --output "D:\ProjectB_runs\stage_run"
```

所有新结果保存在输出目录；原包中的 reference、data、src 保持不变。若某阶段失败，先查看 logs 中完整异常和退出码；不会覆盖部分训练结果。模型/阈值冻结后禁止继续训练。本包入口会先核验清单，直接编辑源码或配置会导致完整性检查失败。若改变模型、标签或超参，应另建项目、登记协议、更新交付清单并重新训练相同配置的静态对照；本运行包一键复现沿用已冻结配置。

## 文件导航

- `run.py`：统一入口；`run_niulaoshi.cmd`：Conda 启动器。
- `src/config.json`：原实验的全部数据/图/标签/训练超参；`src/dynamic_model.py`：动态边权；`src/dynamic_train.py`、`dynamic_score.py`：训练和评估；`portable_tasks.py`：包内路径与运行适配。
- `data/raw/`：54 个官方 NYC 聚合速度分片；`data/geometry/roads.csv`：道路几何；原查询、版本/时间和哈希在 data 下的 JSON。没有私人 Geolife 轨迹。
- `data/prepared/development.npz`、`final.npz`：两份完整真实数据；不能用 pickle 加载。
- `reference/`：原动态边权审计包内容，冻结收据、逐样本预测、36 个新旧模型、日志、结果图与报告均保留。此处的历史绝对路径只作证据，运行时由包内绑定替代。
- `requirements-lock.txt`：实际已验证版本；`requirements.txt`：同一依赖清单。现有 niulaoshi 已具备依赖，无需先执行 pip install。
- `本地运行验证.json`：本包实际验收范围；`运行适配说明.md`：相对旧包的改动；`manifest.csv`、`SHA256SUMS`：包内完整性。

`reference/` 中关于“当时未做第二次完整训练”的文字属于原审计历史记录。本运行包已另外在含空格的新目录完整重训 18 次，具体执行、哈希和结果见根目录 `本地运行验证.json` 与 `validation_logs/`，历史收据保持原样。

为兼容 Windows 路径长度，validation_logs 中的日志使用短编号；各子目录 `_name_map.json` 记录原始文件名及对应关系，JSON 内原命令和日志名称均保留。

## 数据和结果口径

NYC i4gi-tjb9 固定 121 条有向路段，07–09/17–19，十分钟格；最新动态输入来自 t−2，目标 t−1/t 两格速度只作真值。历史均速基准和双格 40% 代理规则、known/unknown、时间划分全部保持原协议。全部已知窗参加评价，unknown 继续出分但不作负类。数据快照最新源时间为 2026-11-11T17:12:06；不自动抓取当前数据。

参考 test 共 80,536 已知窗、2,866 正例；动态 Mean 三种子 AP/F1 均值为 0.6664/0.6501，Max 为 0.6591/0.6440，Attention 为 0.6687/0.6541。主模型仍是开发期选中的 Mean，误报预算尚未满足。它们是已公开日期上的回测，不能说成新的独立盲测。Chicago 与 Geolife 原任务保持原有 blocked 状态，不能用 NYC 成绩替代。

只需要在 Conda 环境运行，无需原 G: 项目目录、旧任务压缩包、API 密钥或网络。依赖环境二进制本身不在 ZIP 内。
