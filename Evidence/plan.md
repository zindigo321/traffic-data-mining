# Project A + 重跑 Project B 期末提交计划（2026-11-17）

原课程目录只读。A以当前CSV和独立重拟合结果为准，旧checksums有3项缺失、14项不符，重新建立交付清单。B原Geolife目录仍blocked，重跑结果采用已核验的纽约动态边权完整运行包。实际结果数据截至2026-11-11，报告日期2026-11-17。

10页：1封面；2 A原始数据/清洗/采样；3 A DBSCAN与KMeans方法及K选择；4 A热点规模与500m聚合图；5 A参数敏感性/共同支持对照/稳定性与候选；6 B数据/原课件差异/标签/时间划分；7 B严格提前10分钟、Graph Transformer与动态边权/三池化；8 B同支持静动态AP/F1三种子图；9 B等召回误报与预算/区间局限；10 总结、局限与复现交付。

设计：16:9、白底黑字、蓝红强调。中文宋体SimSun，英文Times New Roman。图按实测CSV重绘，采用500m空间聚合，不展示个人精确轨迹。正文与流程图可编辑，科研误差条图保留PNG/SVG及原数值。

在线Artifact Session两次host request failed，使用技能允许的本地Artifact Tool导出PPTX，并实际渲染全部10页及检查字体/版面。

期末包：PPTX、PDF预览、逐页讲稿、完整总报告、ProjectA安全聚合提交副本（重跑需外部授权Geolife ZIP）、ProjectB自包含源码/公开数据/权重、图与数值、证据/日志、manifest和SHA256SUMS。排除个人轨迹明文与pickle、凭据及Conda二进制。新目录解压后核验哈希并运行A离线入口及B复算入口。
