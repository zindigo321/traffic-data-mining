# 交通数据挖掘期末作业提交说明

提交日期：2026年11月17日

本包包含Project A和重跑Project B的源码、配置、实测结果、图表、实验总报告与10页汇报。PPT采用白底、黑字、蓝红强调，中文宋体、英文Times New Roman；PDF便于直接预览，逐页讲稿可配合答辩。

## 1. 文件导航

```text
FinalCourse_ProjectAB/
  README_提交说明.md
  01_汇报/             10页PPTX、PDF、逐页讲稿
  ProjectA/            安全聚合提交版、源码、配置、结果与证据
  ProjectB/            niulaoshi完整运行代码、公开数据、权重、结果与日志
  Figures/             本次重绘PNG、SVG及作图数值CSV
  Report/实验总报告.md
  Evidence/            当前复核、课程要求、来源与验收范围
  manifest.csv
  SHA256SUMS
```

建议先看`01_汇报/`，再读`Report/实验总报告.md`；方法细节、数值和运行证据可逐项对应到两个项目目录。

也可在已配置Conda的终端双击`一键复核.cmd`，它使用`niulaoshi`先核对整个提交包哈希，再运行两个默认复核入口。等效命令为`python verify_submission.py --run`；仅核对哈希用`python verify_submission.py`。输出写入全新`runs/`目录。

## 2. 最短复核命令

使用Anaconda Prompt，在解压后的总目录执行：

```bat
conda activate niulaoshi
python ProjectA/run.py
python ProjectB/run.py
```

这两个默认入口分别验证A的聚合结果一致性和B的逐样本指标，不会把已有结果覆盖成新训练结果。新输出在各自runs的新目录，实际命令、用时和退出码保留。无需改装已有Conda环境。

进一步验证：

```bat
python ProjectA/run.py tests
python ProjectB/run.py tests
python ProjectB/run.py replay
python ProjectB/run.py predict
```

Project B提供完整公开数据、prepared数组和权重，replay会实际加载模型并抽样重放；predict实际前向并导出固定历史测试窗的预测。

## 3. 从数据完整重跑

A因保护个体精确轨迹，原始ZIP及点级文件未随包分发。请自行合法取得Geolife Trajectories 1.3，并选择尚不存在的新输出目录：

```bat
python ProjectA/run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\CourseRuns\A_new"
```

用于原样复核的原ZIP SHA-256为：

`16c2caed998e80828acd5fe7b15404593d1752f2a1524505633c23fdfac8a066`

B可完全使用包内固定公开快照：

```bat
python ProjectB/run.py prepare
python ProjectB/run.py train --output "D:\CourseRuns\B_new"
```

B完整训练实际为三池化×三种子×开发/最终共18次新动态训练，静态对照另计18次核验复用；训练配置和评价口径不因提交封装改变。程序不自动从网络抓取新日期，也不安装依赖。完整矩阵所需时间取决于CPU负载，实际资源见项目日志。

## 4. 本次主要结果与正确表述

- A：30用户、1,102轨迹、53,786分析点；DBSCAN主参数50米/20点，174簇，覆盖55.85%、噪声44.15%；KMeans按几何肘部选K=5。同DBSCAN非噪声30,040点支持上的轮廓约0.109和0.428，属于内部评价。90%单次扰动ARI约0.906，低密度候选115个，均不能称为分类准确率或已确认异常驾驶。
- B：原Geolife工程578窗全部unknown、正式训练blocked，99.165%是近邻匹配率。真实重跑采用纽约公开路段估速，严格提前十分钟、有向Graph Transformer和动态边权。开发预选Mean在80,536个known测试窗上AP=0.6664±0.0243、正类F1=0.6501±0.0195，均为三种子均值±样本SD。
- B主模型总误报14.600/日、恢复误报7.504/日，仍高于12/6预算。当前日期结果已经公开，属于历史回测，不是新的独立盲测。纽约任务不等价满足原课件100用户、2010—2012、北京路网和至少5分钟持续拥堵证据。

数据截止、协议差异、每种池化的退化结果和统计局限都在总报告和项目内报告保留。

## 5. 隐私、完整性与验收边界

提交A的聚合统计和500米图，排除个人精确点、轨迹路径、点级标签、时间/身份联结与pickle缓存；A完整点级分析需在复核人本地授权数据上运行。提交B的是官方公开路段聚合估速，不含私人Geolife轨迹。包内没有凭据或Conda环境二进制。

A原历史清单存在3项缺失和14项当前SHA不符，已记录并按当前点表重拟合复核。根目录manifest/SHA256SUMS是本次实际提交文件新建的总清单，旧清单只作历史证据。SHA256SUMS不自我包含，ZIP整体哈希和新目录解压验收收据放外侧或随交付说明给出，最终以实际验收记录为准。

已知验证范围：A原9项工程测试通过，DBSCAN/KMeans在当前点表独立重拟合标签一致；A教师入口为不含原始数据的聚合一致性检查，不会在包内重新计算点级轮廓或ARI。B既有完整运行验收为81项测试、36模型重放概率差0、原始CSV重建数组字节一致、18次动态完整重训权重/指标重现。新总包还应对最终ZIP解压后按根目录清单、默认入口复核，具体完成情况见实际日志与验收收据。


## 本总包实际验收（2026年11月17日）

已将候选ZIP解压至全新目录，在现有niulaoshi环境实际执行：根目录哈希核验、两个项目默认复核入口、A的9项测试、B的81项测试与36模型重放，均通过。原命令、退出码、用时、环境及控制台输出见`Evidence/本次新目录验收/`。收录日志后，仅新增证据与更新说明，代码、数据、模型、图表及PPT保持字节一致；最终ZIP另作全清单和哈希核验，整体SHA保存在交付ZIP旁的`交付文件_SHA256.txt`。
