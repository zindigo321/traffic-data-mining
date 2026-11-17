import fs from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const BUILD=path.dirname(fileURLToPath(import.meta.url));
const WORK=path.dirname(BUILD), FIG=path.join(WORK,'figures');
const BLACK='#111111', BLUE='#1B4F91', RED='#C43C39', GRAY='#616974', PALE='#EFF4FA';
const p=Presentation.create({slideSize:{width:1280,height:720}});
const sourceA='G:/course-jiaotongshujuwajue/ProjectA/results/';
const sourceB='outputs/ProjectB_GT_niulaoshi_20261112/ProjectB_GT_niulaoshi/reference/';
let notes=[];
function runs(text,size,color=BLACK,bold=false){return (text.match(/[\x00-\x7f]+|[^\x00-\x7f]+/g)||[]).map(run=>({run,style:{typeface:/[^\x00-\x7f]/.test(run)?'SimSun':'Times New Roman',fontSize:`${size}pt`,color,bold}}));}
function tx(s,text,x,y,w,h,size=21,color=BLACK,bold=false,align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text.style={typeface:'SimSun',fontSize:`${size}pt`,color,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none'};
 q.text=text.split('\n').map(line=>({runs:runs(line,size,color,bold),spaceAfter:350}));return q;
}
function rect(s,x,y,w,h,fill=PALE,stroke='none',sw=0,geometry='rect'){return s.shapes.add({geometry,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw}});}
function title(text,section){const s=p.slides.add();s.background.fill='#FFFFFF';rect(s,58,39,5,55,BLUE);tx(s,text,80,42,1140,66,29,BLACK,true);tx(s,section,60,112,1160,28,13,BLUE);return s;}
function footer(s,num,source){rect(s,60,665,1160,1,'#CCD5E0');tx(s,source,60,678,1080,26,11,GRAY);tx(s,`${num}/10`,1150,678,70,26,11,GRAY,false,'right');}
function note(s,text,sources){s.speakerNotes.text=text+'\n\n证据：\n'+sources.join('\n');notes.push({slide:notes.length+1,text,sources});}
function img(s,file,x,y,w,h){s.images.add({blob:new Uint8Array(readFileSync(path.join(FIG,file))),contentType:'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt:file});}
function block(s,text,x,y,w,h,color=BLUE,size=19){rect(s,x,y,w,h,'#FFFFFF',color,1.5);tx(s,text,x+14,y+14,w-28,h-20,size,color,true,'center');}
function arr(s,x,y,w=32,color=GRAY){rect(s,x,y,w,13,color,'none',0,'rightArrow');}
function table(s,values,x,y,w,h,widths,size=18){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:values.map(row=>row.map(v=>[{runs:runs(String(v),size)}]))});
 t.borders.assign({fill:'#C9D3DF',width:.7,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
   let cell=t.getCell(r,c);cell.fill=r===0?BLUE:(r%2===0?'#F4F7FB':'#FFFFFF');
   cell.text.style={typeface:'SimSun',fontSize:`${size}pt`,color:r===0?'#FFFFFF':BLACK,insets:8,verticalAlignment:'middle'};
   cell.value=[{runs:runs(String(values[r][c]),size,r===0?'#FFFFFF':BLACK,r===0)}];
 }
 return t;
}

// 1 — cover
{
 const s=p.slides.add();s.background.fill='#FFFFFF';
 rect(s,65,91,84,6,BLUE);rect(s,153,91,30,6,RED);
 tx(s,'交通轨迹热点识别\n与道路拥堵预警',66,142,850,170,39,BLACK,true);
 tx(s,'交通数据挖掘课程期末报告',70,340,820,48,24,BLUE);
 tx(s,'Project A  DBSCAN / KMeans\nProject B  Dynamic-edge Graph Transformer',70,419,900,96,23,BLACK);
 tx(s,'汇报人：课程作者    学号：ANON_STUDENT\n2026年11月17日',70,576,1020,70,18,GRAY);
 const ns=[[1030,225],[1150,290],[1015,365],[1150,450]];
 for(const [i,[x,y]]of ns.entries()){rect(s,x,y,48,48,i===2?RED:BLUE,'none',0,'ellipse');}
 arr(s,1083,248,58,BLUE);arr(s,1075,384,65,RED);
 tx(s,'空间分布\n有向关联',984,514,230,84,20,BLUE,false,'center');
 footer(s,1,'实验数据冻结于2026-11-12；纽约快照最新源时间为2026-11-11 17:12:06');
 note(s,'本报告汇总两个独立任务。Project A研究Geolife空间密度与热点，Project B采用有向Graph Transformer进行严格提前十分钟预警。B展示的是纽约公开数据补充回测，原Geolife标签不足的状态仍保留。封面姓名依据原答辩，学号依据作业文件名。',[sourceA+'analysis_summary.json',sourceB+'results/summary.json','G:/course-jiaotongshujuwajue/答辩/交通数据挖掘课程答辩.pptx']);
}
// 2 — A data
{
 const s=title('从原始轨迹到53,786个空间分析点','PROJECT A  数据与清洗');
 const stages=[['2,029,261','选择文件中的原始点'],['1,691,926','区域与质量清洗后'],['101,019','60秒抽稀后'],['53,786','固定限额分析样本']];
 stages.forEach(([n,l],i)=>{let x=60+i*294;tx(s,n,x,183,270,68,33,i===3?RED:BLUE,true);tx(s,l,x,255,268,56,18);if(i<3)arr(s,x+258,215,27);});
 rect(s,60,338,1160,1,'#CCD5E0');
 tx(s,'研究样本',60,371,500,40,23,BLUE,true);
 tx(s,'30名用户，1,102条最终分析轨迹\n北京区域，实际观测覆盖2007—2011年\n固定随机种子与每用户点数上限',60,428,550,163,21);
 tx(s,'质量与尺度',690,371,520,40,23,BLUE,true);
 tx(s,'处理重复、时间冲突与跳变端点\n60秒抽稀，300秒断点分段\n经纬度投影到UTM 50N，以米计算距离',690,428,530,163,21);
 footer(s,2,'统计单位逐级变化；30用户抽样不代表全部Geolife用户');
 note(s,'选取1108份文件中共有2029261点，清洗后区域内1691926点，分钟抽稀后101019点，按配置形成53786点分析样本，最终覆盖1102轨迹。清洗剔除非法点、重复、同刻冲突及跳变端点，距离采用EPSG:32650投影坐标。A的主聚类仅用二维空间x/y，时间用于清洗及后续分时统计。',[sourceA+'cleaning_summary.csv',sourceA+'analysis_summary.json','G:/course-jiaotongshujuwajue/ProjectA/config.json']);
}
// 3 — A algorithms
{
 const s=title('密度热点与空间分区采用两种聚类模型','PROJECT A  方法与参数选择');
 tx(s,'DBSCAN',60,165,500,48,27,BLUE,true);
 tx(s,'主参数 ε = 50 m，MinPts = 20\n在米制投影坐标中寻找密度连通簇\n低密度点保留为噪声标签 −1',60,225,540,150,21);
 tx(s,'参数网格',60,421,510,42,23,BLUE,true);
 tx(s,'ε：30、50、100、200、500 m\nMinPts：5、10、20，共15组\n同时检查覆盖、噪声率与簇规模',60,476,540,150,21);
 tx(s,'KMeans',677,165,500,48,27,RED,true);
 img(s,'A_kmeans_elbow.png',655,221,565,355);
 tx(s,'K = 5按肘部规则选择；不是轮廓系数最优值',660,596,558,48,17,RED);
 footer(s,3,'两种方法使用同一53,786点样本，另在共同非噪声支持上比较');
 note(s,'DBSCAN网格共15组。KMeans考察K为2至10，按惯性肘部规则取5，不能说K=5取得最高轮廓系数。两者模型偏好不同，聚类覆盖范围不同，比较时另外使用DBSCAN非噪声共同支持30040点。',[sourceA+'dbscan_experiments.csv',sourceA+'kmeans_experiments.csv',sourceA+'common_sample_comparison.csv']);
}
// 4 — A measured spatial outcome
{
 const s=title('174个热点簇覆盖55.85%的分析点','PROJECT A  空间聚类结果');
 img(s,'A_spatial_density.png',42,153,755,459);
 tx(s,'174',845,169,350,65,37,BLUE,true);
 tx(s,'个密度热点簇',845,236,350,44,22);
 tx(s,'聚类点 30,040\n噪声点 23,746（44.15%）',845,297,360,84,21);
 img(s,'A_top_cluster_sizes.png',807,403,405,218);
 footer(s,4,'空间图为500m网格聚合的聚类点计数，不展示逐人轨迹或精确点位');
 note(s,'主参数DBSCAN得到174簇，30040点属于簇，23746点属于噪声。图中的500米网格只聚合主分析样本中的非噪声点，颜色为log10(1+点数)，不包含个人标识或逐点路径。噪声是算法定义的低密度状态，不是已确认的异常驾驶。',[sourceA+'point_labels.csv.gz','figures/A_grid500m.csv',sourceA+'dbscan_experiments.csv']);
}
// 5 — A robustness and comparison
{
 const s=title('参数敏感性与重采样检查支持结果复核','PROJECT A  对照、稳健性与候选');
 img(s,'A_parameter_noise.png',49,169,620,353);
 tx(s,'ARI = 0.906',739,173,468,67,31,BLUE,true);
 tx(s,'90%子样本，48,407点\n包含噪声标签的一次一致性检查\n不作为分类准确率或显著性结论',739,252,475,130,20);
 tx(s,'115条低密度候选',739,425,470,42,24,RED,true);
 tx(s,'需结合场景复核，尚无异常驾驶真值',739,480,470,60,19);
 table(s,[['共同30,040点支持','DBSCAN','KMeans (K=5)'],['轮廓系数','0.1086','0.4277']],60,550,1160,90,[580,250,330],17);
 footer(s,5,'固定MinPts=20；ε从30增至500m时噪声率由60.80%降至4.52%');
 note(s,'独立重拟合得到完全相同标签，ARI重算一致。这里的ARI来自一次固定90%子样本，未重复做多种子置信区间。共同支持轮廓系数DBSCAN为0.108585、KMeans为0.427705，不能直接用单个轮廓指标替代密度热点的空间意义。115条轨迹满足低密度候选规则，缺少真实异常标签。',[sourceA+'analysis_summary.json',sourceA+'common_sample_comparison.csv',sourceA+'trajectory_candidates.csv','work/final_course_20261117/A_evidence.json']);
}
// 6 — B protocol
{
 const s=title('纽约公开数据用于拥堵预警补充回测','PROJECT B  原任务状态与数据协议');
 tx(s,'原Geolife目录：578窗全部unknown，正式训练仍blocked',60,166,1160,45,23,RED,true);
 tx(s,'补充数据：121条有向路段，10分钟格，07—09与17—19高峰',60,228,1160,45,22,BLUE);
 table(s,[['部分（右端不含）','已知窗','正例窗'],['训练 05-11 至 08-31','171,672','4,649'],['校准 08-31 至 09-28','47,780','1,476'],['回测 09-28 至 11-12','80,536','2,866']],60,303,1160,223,[670,260,230],20);
 tx(s,'标签：相邻两格估速分别低于各自过去同期均速40%\n历史至少3个合格过去日；校准/测试历史冻结；unknown不作负类',60,553,1160,90,20);
 footer(s,6,'年份为2026年；课件的100用户、2010—2012与持续≥5分钟要求未由本补充实验满足');
 note(s,'当前课程ProjectB原目录training_status仍blocked，578个窗口都是insufficient_history。NYC代理任务独立报告，不能把近邻道路匹配率99.165%当分类准确率。全快照537240个固定窗，known299988、unknown237252；回测known80536、阳性率3.56%、覆盖61.63%。原始数据为官方聚合估速，双十分钟代理不证明每一分钟连续低速。',[
 'G:/course-jiaotongshujuwajue/ProjectB/results/training_status.json','G:/course-jiaotongshujuwajue/ProjectB/results/pipeline_summary.json',sourceB+'evidence/cohort_final.json','课程PDF物理页11—15']);
}
// 7 — B actual architecture and causality
{
 const s=title('有向图掩码与动态边权约束两层注意力','PROJECT B  Graph Transformer 与三种池化');
 block(s,'路段48',72,179,164,76,BLUE,23);arr(s,252,209,90,BLUE);block(s,'目标路段49',360,179,215,76,RED,22);
 tx(s,'道路是节点，方向相容的连接是边\n子图最多5节点，54条路段为孤立节点',60,280,555,91,19);
 tx(s,'边倍率0.5—2，缺失退回1\nscore = QKᵀ / √8 + log(w) + mask',60,376,555,76,18,BLUE);
 tx(s,'Mean：有效节点均值\nMax：有效节点逐维最大\nAttention：学习汇总权重',690,176,530,139,22);
 tx(s,'子图汇总32维 + 中心节点32维\n分类器输出该目标路段的一个分数',690,345,530,86,21,BLUE);
 const xs=[60,358,656,954];const texts=['输入104维\n52值 + 52缺失位','图注意力32维\n2层 × 4头','目标与子图表示\n32 + 32 = 64维','路段级输出\n冻结阈值判定'];
 texts.forEach((v,i)=>{block(s,v,xs[i],467,266,103,i===3?RED:BLUE,18);if(i<3)arr(s,xs[i]+272,510,20);});
 tx(s,'严格提前10分钟：仅用t−2及更早格，t−1和t两格速度只在真值侧',60,596,1160,43,20,RED,true);
 footer(s,7,'注意力仅允许物理前驱与自身；时间关系由滞后特征和过去历史统计表达');
 note(s,'输入[B,N,104]映射到32维，四头每头8维，两层注意力块。score=QKᵀ/sqrt(8)+log(w)，非物理边mask为负无穷；源到目标方向转置为query目标/key前驱。动态边MLP48→16→4，每层852参数，两层1704参数，使用合法过去速度比、趋势、支持与缺失位。自环及关键速度缺失时倍率1。三池化独立训练。模拟07:30决策预测07:40—07:50，只能使用07:20—07:30及更早观测。',['ProjectB/src/dynamic_model.py','ProjectB/src/v10/v9/v8/v7/v7_prepare.py','ProjectB/src/dynamic_train.py',sourceB+'evidence/graph.json']);
}
// 8 — B primary result
{
 const s=title('三种池化的AP与F1均值均超过0.60','PROJECT B  同支持静态与动态边权对照');
 img(s,'B_pooling_AP_F1.png',45,157,1180,425);
 tx(s,'开发预选主模型：动态Mean   AP 0.6664 ± 0.0243   F1 0.6501 ± 0.0195',60,588,1160,49,20,BLUE,true);
 footer(s,8,'均值±样本标准差；种子11/22/33；12轮预算一致；静态对照为已核验同协议模型');
 note(s,'全部三池化每个动态模型三种子。静态AP/F1：Mean .65526/.64552，Max .65914/.65514，Attention .66382/.63880。动态分别为Mean .66644/.65005、Max .65913/.64398、Attention .66866/.65408。AP是average_precision_score。动态Max的F1下降如实保留。开发按平均AP预选动态Mean，不能因为测试Attention较高就事后改主模型。所有测试支持相同，unknown排除主评价但不筛除困难known样本。',[sourceB+'results/summary.json',sourceB+'results/development_metrics.csv',sourceB+'results/dynamic_selection.json','figures/B_plot_values.csv']);
}
// 9 — B matched FP and limits
{
 const s=title('Mean等召回误报下降，稳定性仍待验证','PROJECT B  恢复误报与阈值预算');
 img(s,'B_matched_recall.png',45,158,1180,392);
 tx(s,'60%召回下，Mean恢复误报 7.815 降至 7.570 / 日',60,556,1160,40,21,BLUE,true);
 tx(s,'冻结阈值回测：总误报14.600 / 日、恢复7.504 / 日，仍超12 / 6预算',60,606,1160,42,19,RED,true);
 footer(s,9,'Mean恢复误报差的日期块95%区间[−0.585, 0.089]跨0；此处是历史回测而非新盲测');
 note(s,'图为全量known支持上相同TP的60%召回排序诊断，不是用测试集拟合部署阈值。Mean总误报15.3778→14.7481/day，恢复7.8148→7.5704/day；三个种子方向一致，但日期块1000次bootstrap恢复区间跨零。冻结阈值按独立校准段受约束最大F1选择，回测平均误报仍超预算。纽约测试日期已经公开，不能描述为新独立留出。',[sourceB+'results/matched_recall.csv',sourceB+'results/paired_date_blocks.csv',sourceB+'results/operating_points.csv']);
}
// 10 — final package and conclusion
{
 const s=title('两项项目形成可复核的课程实验闭环','结论与期末作业交付');
 table(s,[['项目','已完成','主要边界'],['Project A','清洗、投影、聚类对照\n参数与稳定性、热点和候选','30用户限额样本\n无异常驾驶真值'],['Project B','严格预警、三池化三种子\n动态边权与同支持误报比较','纽约代理历史回测\n误报预算与原任务仍有限制']],60,170,1160,236,[200,530,430],19);
 tx(s,'复现入口',60,443,1100,38,23,BLUE,true);
 tx(s,'Project A：离线复核聚合结果；完整重跑需授权Geolife ZIP\nProject B：niulaoshi环境，python run.py复算，train完整重训',60,498,1160,91,21);
 tx(s,'作业包包含源码、配置、汇总结果、真实图表、讲稿与哈希清单',60,608,1160,39,20,BLACK,true);
 footer(s,10,'A独立重拟合与9项测试通过；B 81项测试、18次动态重训及36模型重放已核验');
 note(s,'Project A主要贡献是对热点、噪声和候选的可解释空间分析，不能报告监督分类准确率。Project B实现真实图结构约束的Transformer和动态边权，在独立校准后冻结阈值，三池化对照和误报边界完整报告。交付排除个人精确轨迹明文与pickle，A依赖外部授权原始ZIP才能从头复跑，B含公开快照、数组和权重。课程原有Geolife用户/年份/持续五分钟要求与补充纽约实验的差异保留。',['work/final_course_20261117/A_evidence.json','ProjectA/evidence/','ProjectB/本地运行验证.json','README_提交说明.md']);
}

if(p.slides.items.length!==10)throw new Error('Expected 10 total slides');
await fs.writeFile(path.join(WORK,'speaker_notes.json'),JSON.stringify(notes,null,2));
await fs.writeFile(path.join(WORK,'逐页讲稿.md'),notes.map(n=>`## 第${n.slide}页\n\n${n.text}\n\n来源：\n${n.sources.map(x=>'- '+x).join('\n')}`).join('\n\n'));
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
await fs.mkdir(path.join(BUILD,'render'),{recursive:true});
for(let i=0;i<10;i++){
 try{const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(BUILD,'render',`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));console.log(`rendered ${i+1}`);}catch(e){console.error(`render ${i+1}: ${e}`);}
}
await fs.writeFile(path.join(BUILD,'deck_proto.json'),JSON.stringify(p.toProto()));
console.log('10 slides authored');
