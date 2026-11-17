# 交通数据挖掘课程作业与项目记录

本仓库整理每周作业、Project A、重跑Project B及期末汇报。资料整理于2026-11-17；项目数据截止和实验日期以各自证据记录为准。

| 目录 | 内容 |
| --- | --- |
| [assignments](assignments/) | 第一至第三周作业PDF，第三周另有可编辑DOCX |
| [ProjectA](ProjectA/) | Geolife轨迹清洗、DBSCAN/KMeans热点与低密度候选分析 |
| [ProjectB](ProjectB/) | 严格提前10分钟、动态边权Graph Transformer、三池化三种子 |
| [01_汇报](01_汇报/) | 新版10页PPTX、PDF和逐页讲稿 |
| [Report](Report/实验总报告.md) | 实验全流程、结果及课程要求差异 |
| [Figures](Figures/) | 实测结果图与作图数值 |
| [Evidence](Evidence/) | 来源、数据合同、模型/指标复核与实际运行日志 |
| [archive](archive/legacy-presentation/) | 原答辩版本，仅作历史记录 |

## 主要结果

- Project A：30用户、53,786分析点；DBSCAN 50米/20点形成174簇，覆盖55.85%，噪声44.15%；KMeans按肘部规则选K=5。聚类内部指标及ARI不能当作分类准确率，115个低密度候选没有异常驾驶真值。
- Project B原Geolife目录：578窗全部unknown，正式监督训练blocked。重跑是纽约公开路段估速的独立补充代理任务，不能替代原课件100用户、2010—2012年与五分钟连续拥堵证据。
- 纽约动态Mean（开发预选）在80,536个known历史测试窗上，三个种子AP=0.6664±0.0243、正类F1=0.6501±0.0195。总误报14.600/日、恢复误报7.504/日仍超12/6预算；当前日期为已公开历史回测，并非新的独立盲测。

## 运行

在已有`niulaoshi`环境中，从仓库根目录执行：

```bat
conda activate niulaoshi
python verify_submission.py --run
python ProjectA/run.py tests
python ProjectB/run.py tests
python ProjectB/run.py replay
```

默认仅新建runs目录，不覆盖参考实验。A默认复核聚合结果；完整重跑需使用者自行合法取得Geolife Trajectories 1.3 ZIP：

```bat
python ProjectA/run.py full --data "D:\data\Geolife Trajectories 1.3.zip" --out "D:\CourseRuns\A_new"
python ProjectB/run.py prepare
python ProjectB/run.py train --output "D:\CourseRuns\B_new"
```

B使用包内固定NYC公开快照、prepared数组和实际权重；train执行18次动态训练，静态18次对照为核验复用。依赖见各项目requirements文件；不需要上传或复制Conda环境。

## 公开范围和完整性

未复制私人Geolife原始轨迹、个人精确点表、pickle、教师课件或原始课程压缩包。B公开聚合路段数据的来源、快照、修改与使用条件见[数据说明](ProjectB/数据来源与使用说明.md)；原始与第三方内容的权利不因本仓库公开而转移。本次未替第三方材料指定开源许可。公开副本已移除文档署名、学号、个人作者属性、个人账号和本机用户目录标识；审查范围见PrivacyReview.json。

根目录manifest.csv和SHA256SUMS已为匿名公开副本重建，覆盖本次全部公开文件；ProjectB内部清单也已为匿名副本更新。历史实验记录的个人路径采用匿名映射，历史原始字节哈希与当前公开副本哈希分开理解。`.gitattributes`禁止换行转换，以保持原清单字节哈希有效。请勿在仓库中原位改写参考结果；新实验输出写入runs。

GitHub首次提交和后续更新步骤见[提交指南](docs/GITHUB_UPLOAD.md)。
