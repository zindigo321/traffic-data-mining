import fs from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const root = path.resolve(process.cwd());
const output = path.resolve(process.argv[2] || 'runs/presentation');
const figures = path.join(root, 'Figures');
const black = '#111111';
const blue = '#1B4F91';
const red = '#C43C39';
const gray = '#616974';
const pale = '#EFF4FA';
const font = 'Times New Roman';
const deck = Presentation.create({slideSize: {width: 1280, height: 720}});
const sourceA = 'ProjectA/results/';
const sourceB = 'ProjectB/reference/';
const notes = [];

function runs(text, size, color = black, bold = false) {
  return [{run: text, style: {typeface: font, fontSize: size + 'pt', color, bold}}];
}
function text(slide, value, x, y, width, height, size = 21, color = black, bold = false, alignment = 'left') {
  const shape = slide.shapes.add({
    geometry: 'textbox',
    position: {left: x, top: y, width, height},
    fill: 'none',
    line: {fill: 'none', width: 0},
  });
  shape.text.style = {
    typeface: font, fontSize: size + 'pt', color, alignment, verticalAlignment: 'top', insets: 0, autoFit: 'none',
  };
  shape.text = value.split('\n').map(line => ({runs: runs(line, size, color, bold), spaceAfter: 350}));
}
function rect(slide, x, y, width, height, fill = pale, stroke = 'none', lineWidth = 0, geometry = 'rect') {
  slide.shapes.add({
    geometry,
    position: {left: x, top: y, width, height},
    fill,
    line: {fill: stroke, width: lineWidth},
  });
}
function title(value, section) {
  const slide = deck.slides.add();
  slide.background.fill = '#FFFFFF';
  rect(slide, 58, 39, 5, 55, blue);
  text(slide, value, 80, 42, 1140, 66, 29, black, true);
  text(slide, section, 60, 112, 1160, 28, 13, blue);
  return slide;
}
function footer(slide, number, source) {
  rect(slide, 60, 665, 1160, 1, '#CCD5E0');
  text(slide, source, 60, 678, 1080, 26, 11, gray);
  text(slide, number + '/10', 1150, 678, 70, 26, 11, gray, false, 'right');
}
function note(slide, value, sources) {
  slide.speakerNotes.text = value + '\n\nEvidence\n' + sources.join('\n');
  notes.push({slide: notes.length + 1, text: value, sources});
}
function image(slide, file, x, y, width, height) {
  slide.images.add({
    blob: new Uint8Array(readFileSync(path.join(figures, file))),
    contentType: 'image/png',
    position: {left: x, top: y, width, height},
    fit: 'contain',
    alt: file,
  });
}
function block(slide, value, x, y, width, height, color = blue, size = 19) {
  rect(slide, x, y, width, height, '#FFFFFF', color, 1.5);
  text(slide, value, x + 10, y + 13, width - 20, height - 20, size, color, true, 'center');
}
function arrow(slide, x, y, width = 32, color = gray) {
  rect(slide, x, y, width, 13, color, 'none', 0, 'rightArrow');
}
function table(slide, values, x, y, width, height, widths, size = 18) {
  const view = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: x,
    top: y,
    width,
    height,
    columnWidths: widths,
    values: values.map(row => row.map(value => [{runs: runs(String(value), size)}])),
  });
  view.borders.assign({fill: '#C9D3DF', width: .7, style: 'solid'});
  for (let row = 0; row < values.length; row++) {
    for (let col = 0; col < values[row].length; col++) {
      const cell = view.getCell(row, col);
      const color = row === 0 ? '#FFFFFF' : black;
      cell.fill = row === 0 ? blue : row % 2 === 0 ? '#F4F7FB' : '#FFFFFF';
      cell.text.style = {typeface: font, fontSize: size + 'pt', color, insets: 8, verticalAlignment: 'middle'};
      cell.value = values[row][col].split('\n').map(value => ({runs: runs(value, size, color, row === 0)}));
    }
  }
}

{
  const slide = deck.slides.add();
  slide.background.fill = '#FFFFFF';
  rect(slide, 65, 91, 84, 6, blue);
  rect(slide, 153, 91, 30, 6, red);
  text(slide, 'Trajectory Hotspots and\nRoad Congestion Warnings', 66, 142, 850, 170, 39, black, true);
  text(slide, 'Traffic Data Mining Course Final Report', 70, 340, 820, 48, 24, blue);
  text(slide, 'Project A  DBSCAN / KMeans\nProject B  Dynamic-edge Graph Transformer', 70, 419, 900, 96, 23);
  text(slide, 'November 17, 2026', 70, 576, 1020, 70, 18, gray);
  const nodes = [[1030, 225], [1150, 290], [1015, 365], [1150, 450]];
  nodes.forEach(([x, y], index) => rect(slide, x, y, 48, 48, index === 2 ? red : blue, 'none', 0, 'ellipse'));
  arrow(slide, 1083, 248, 58, blue);
  arrow(slide, 1075, 384, 65, red);
  text(slide, 'Spatial patterns\nDirected links', 984, 514, 230, 84, 20, blue, false, 'center');
  footer(slide, 1, 'Data freeze: 2026-11-12. Latest NYC snapshot source timestamp: 2026-11-11 17:12:06.');
  note(slide, 'This report covers two independent projects. Project A studies spatial density and hotspots in Geolife trajectories. Project B uses a directed Graph Transformer to issue warnings with a strict 10-minute lead. The NYC public-data backtest is a supplemental task. The original Geolife task still lacks sufficient labels.', [
    sourceA + 'analysis_summary.json',
    sourceB + 'results/summary.json',
    'archive/legacy-presentation/traffic_data_mining_course_defense.pptx',
  ]);
}

{
  const slide = title('From Raw Trajectories to 53,786 Analysis Points', 'PROJECT A  DATA AND CLEANING');
  const stages = [['2,029,261', 'Raw points in selected files'], ['1,691,926', 'After region and quality filters'], ['101,019', 'After 60-second thinning'], ['53,786', 'Analysis sample with fixed caps']];
  stages.forEach(([value, label], index) => {
    const x = 60 + index * 294;
    text(slide, value, x, 183, 270, 68, 33, index === 3 ? red : blue, true);
    text(slide, label, x, 255, 268, 62, 17);
    if (index < 3) arrow(slide, x + 258, 215, 27);
  });
  rect(slide, 60, 338, 1160, 1, '#CCD5E0');
  text(slide, 'Study Sample', 60, 371, 500, 40, 23, blue, true);
  text(slide, '30 users and 1,102 analysis trajectories\nBeijing observations from 2007 to 2011\nFixed seed and per-user point caps', 60, 428, 550, 163, 20);
  text(slide, 'Quality and Scale', 690, 371, 520, 40, 23, blue, true);
  text(slide, 'Duplicates, time conflicts, and jump endpoints removed\n60-second thinning and 300-second segmentation gaps\nDistances in meters using the UTM 50N projection', 690, 428, 530, 180, 19);
  footer(slide, 2, 'Stages report different units. The 30-user sample does not represent the full Geolife population.');
  note(slide, 'The selected 1,108 files contained 2,029,261 points. Region and quality filters retained 1,691,926, minute thinning retained 101,019, and the configured sample contained 53,786 points across 1,102 trajectories. Cleaning removed invalid points, duplicates, same-time conflicts, and jump endpoints. Distances use EPSG:32650 projected coordinates. The main clustering uses only spatial x/y coordinates. Time is used for cleaning and later time-of-day summaries.', [
    sourceA + 'cleaning_summary.csv', sourceA + 'analysis_summary.json', 'ProjectA/config.json',
  ]);
}

{
  const slide = title('Two Models for Hotspots and Spatial Partitioning', 'PROJECT A  METHODS AND PARAMETER SELECTION');
  text(slide, 'DBSCAN', 60, 165, 500, 48, 27, blue, true);
  text(slide, 'Main setting: ε = 50 m, MinPts = 20\nFinds density-connected clusters in meters\nLabels low-density points as noise (−1)', 60, 225, 540, 150, 20);
  text(slide, 'Parameter Grid', 60, 421, 510, 42, 23, blue, true);
  text(slide, 'ε: 30, 50, 100, 200, 500 m\nMinPts: 5, 10, 20, for 15 combinations\nChecks coverage, noise share, and cluster size', 60, 476, 540, 150, 20);
  text(slide, 'KMeans', 677, 165, 500, 48, 27, red, true);
  image(slide, 'A_kmeans_elbow.png', 655, 221, 565, 355);
  text(slide, 'The elbow rule selects K = 5. Silhouette peaks elsewhere.', 660, 596, 558, 48, 17, red);
  footer(slide, 3, 'Both methods use the same 53,786 points. An additional comparison uses shared non-noise support.');
  note(slide, 'The DBSCAN grid contains 15 settings. KMeans tests K from 2 through 10. The inertia elbow rule selected K = 5 independently of the silhouette score. The methods favor different cluster shapes and cover different sets of points, so the comparison also uses the same 30,040-point DBSCAN non-noise support.', [
    sourceA + 'dbscan_experiments.csv', sourceA + 'kmeans_experiments.csv', sourceA + 'common_sample_comparison.csv',
  ]);
}

{
  const slide = title('174 Hotspot Clusters Cover 55.85% of Points', 'PROJECT A  SPATIAL CLUSTERING RESULTS');
  image(slide, 'A_spatial_density.png', 42, 153, 755, 459);
  text(slide, '174', 845, 169, 350, 65, 37, blue, true);
  text(slide, 'Density hotspots', 845, 236, 350, 44, 22);
  text(slide, '30,040 clustered points\n23,746 noise points (44.15%)', 845, 297, 360, 84, 21);
  image(slide, 'A_top_cluster_sizes.png', 807, 403, 405, 218);
  footer(slide, 4, 'The spatial plot counts clustered points in 500 m cells and omits individual paths and exact locations.');
  note(slide, 'The main DBSCAN setting produced 174 clusters, with 30,040 points in clusters and 23,746 labeled noise. The 500 m grid aggregates only non-noise points from the main analysis sample. Color represents log10(1 + point count). The figure omits personal identifiers and point-level tracks. Noise denotes low density under the model, with no verified abnormal-driving label.', [
    sourceA + 'point_labels.csv.gz (restricted original)', 'Figures/A_grid500m.csv', sourceA + 'dbscan_experiments.csv',
  ]);
}

{
  const slide = title('Parameter Sensitivity and Resampling Checks', 'PROJECT A  COMPARISON, STABILITY, AND CANDIDATES');
  image(slide, 'A_parameter_noise.png', 49, 169, 620, 353);
  text(slide, 'ARI = 0.906', 739, 173, 468, 67, 31, blue, true);
  text(slide, '90% subsample, 48,407 points\nOne agreement check including noise\nNo classification or significance claim', 739, 252, 475, 130, 19);
  text(slide, '115 Low-Density Candidates', 739, 425, 470, 42, 23, red, true);
  text(slide, 'Review in context. Ground truth for abnormal driving is unavailable.', 739, 480, 470, 60, 18);
  table(slide, [['Shared Support: 30,040 Points', 'DBSCAN', 'KMeans (K=5)'], ['Silhouette score', '0.1086', '0.4277']], 60, 550, 1160, 90, [580, 250, 330], 17);
  footer(slide, 5, 'At MinPts = 20, increasing ε from 30 to 500 m lowers the noise share from 60.80% to 4.52%.');
  note(slide, 'Independent refitting reproduced every label and the ARI calculation. The ARI comes from one fixed 90% subsample. No multi-seed confidence interval was estimated. On shared support, silhouette scores were 0.108585 for DBSCAN and 0.427705 for KMeans. Silhouette measures compactness and does not establish the spatial relevance of density hotspots. The low-density rule flagged 115 trajectories, with no true abnormal-driving labels available.', [
    sourceA + 'analysis_summary.json', sourceA + 'common_sample_comparison.csv',
    sourceA + 'trajectory_candidates.csv (restricted original)', 'Evidence/A_evidence.json',
  ]);
}

{
  const slide = title('NYC Public-Data Backtest for Congestion Warnings', 'PROJECT B  ORIGINAL TASK STATUS AND NYC DATA PROTOCOL');
  text(slide, 'Geolife: all 578 windows have unknown labels. Supervised training is blocked.', 60, 166, 1160, 45, 22, red, true);
  text(slide, 'NYC supplement: 121 directed links, 10-minute bins, peak hours 07:00–09:00 and 17:00–19:00', 60, 228, 1160, 45, 20, blue);
  table(slide, [['Split (End Exclusive)', 'Known Windows', 'Positive Windows'], ['Train 05-11 to 08-31', '171,672', '4,649'], ['Calibrate 08-31 to 09-28', '47,780', '1,476'], ['Backtest 09-28 to 11-12', '80,536', '2,866']], 60, 303, 1160, 223, [670, 260, 230], 20);
  text(slide, 'Label: both adjacent estimates fall below 40% of their historical slot means.\nAt least three eligible past dates. Calibration and test history is frozen. Unknown labels stay separate.', 60, 553, 1160, 90, 19);
  footer(slide, 6, 'NYC dates are in 2026. The original 100-user, 2010–2012, and continuous five-minute requirements remain unmet.');
  note(slide, 'The original course Project B training status remains blocked. All 578 windows have insufficient history. The NYC proxy is a separate task. The 99.165% nearest-road matching rate measures matching coverage, while classification accuracy requires labels. The full snapshot has 537,240 fixed windows, with 299,988 known and 237,252 unknown. The backtest has 80,536 known windows, a positive rate of 3.56%, and coverage of 61.63%. The source supplies official aggregate speed estimates. Two 10-minute low-speed estimates do not establish continuous minute-level low speed.', [
    'Evidence/Geolife_original_B/training_status.json', 'Evidence/Geolife_original_B/pipeline_summary.json',
    sourceB + 'evidence/cohort_final.json', 'Course PDF, physical pages 11–15',
  ]);
}

{
  const slide = title('Directed Masks and Dynamic Edge Weights', 'PROJECT B  GRAPH TRANSFORMER AND THREE POOLING METHODS');
  block(slide, 'Link 48', 72, 179, 164, 76, blue, 23);
  arrow(slide, 252, 209, 90, blue);
  block(slide, 'Target Link 49', 360, 179, 215, 76, red, 21);
  text(slide, 'Road links are nodes, with edges between compatible directions.\nSubgraphs have up to 5 nodes. 54 links are isolated.', 60, 280, 555, 96, 18);
  text(slide, 'Edge weights range from 0.5 to 2. Missing inputs use 1.\nscore = QKᵀ / √8 + log(w) + mask', 60, 376, 555, 82, 17, blue);
  text(slide, 'Mean: mean of valid nodes\nMax: per-feature maximum over valid nodes\nAttention: learned pooling weights', 690, 176, 530, 139, 20);
  text(slide, '32 pooled features + 32 target-node features\nOne classifier score per target road link', 690, 345, 530, 86, 19, blue);
  const steps = ['104 Inputs\n52 values + 52 masks', '32 Features\n2 layers × 4 heads', 'Target and Subgraph\n32 + 32 = 64 features', 'Road-Link Score\nFrozen threshold'];
  [60, 358, 656, 954].forEach((x, index) => {
    block(slide, steps[index], x, 467, 266, 103, index === 3 ? red : blue, 18);
    if (index < 3) arrow(slide, x + 272, 510, 20);
  });
  text(slide, 'Strict 10-minute lead: input bins end at t−2. Speeds in t−1 and t define the label.', 60, 596, 1160, 43, 20, red, true);
  footer(slide, 7, 'Attention sees physical predecessors and self. Lagged features and historical statistics encode time.');
  note(slide, 'The [B, N, 104] input projects to 32 features. Two attention blocks use four heads with eight features per head. The score is QKᵀ/sqrt(8) + log(w). The mask sets nonphysical edges to negative infinity and transposes the source–target orientation so a target query sees predecessor keys. The dynamic edge MLP has dimensions 48 → 16 → 4, with 852 parameters per layer and 1,704 total. Its inputs include past speed ratios, trends, support, and missingness flags. Self-loops and missing key speeds use a multiplier of 1. Each pooling variant trains independently. A simulated decision at 07:30 predicts 07:40–07:50 using observations from 07:20–07:30 and earlier.', [
    'ProjectB/src/dynamic_model.py', 'ProjectB/src/v10/v9/v8/v7/v7_prepare.py',
    'ProjectB/src/dynamic_train.py', sourceB + 'evidence/graph.json',
  ]);
}

{
  const slide = title('AP and F1 Average Above 0.60 for All Pooling Methods', 'PROJECT B  STATIC AND DYNAMIC EDGES ON MATCHED SUPPORT');
  image(slide, 'B_pooling_AP_F1.png', 45, 157, 1180, 425);
  text(slide, 'Primary model selected on development: Dynamic Mean   AP 0.6664 ± 0.0243   F1 0.6501 ± 0.0195', 60, 588, 1160, 49, 19, blue, true);
  footer(slide, 8, 'Mean ± sample SD. Seeds 11, 22, 33. Equal 12-epoch budgets. Verified static controls use the same protocol.');
  note(slide, 'Three seeds were used for each dynamic pooling model. Static AP/F1 was 0.65526/0.64552 for Mean, 0.65914/0.65514 for Max, and 0.66382/0.63880 for Attention. Dynamic AP/F1 was 0.66644/0.65005, 0.65913/0.64398, and 0.66866/0.65408, respectively. AP uses average_precision_score. Dynamic Max lost F1. Development mean AP selected Dynamic Mean before the test. The higher test Attention result does not change that selection. All models use the same test support. Unknown labels are excluded from primary evaluation, with all difficult known windows retained.', [
    sourceB + 'results/summary.json', sourceB + 'results/development_metrics.csv',
    sourceB + 'results/dynamic_selection.json', 'Figures/B_plot_values.csv',
  ]);
}

{
  const slide = title('Mean False Alarms Fall at Matched Recall', 'PROJECT B  RECOVERY FALSE ALARMS AND THRESHOLD BUDGETS');
  image(slide, 'B_matched_recall.png', 45, 158, 1180, 392);
  text(slide, 'At 60% recall, Mean recovery false positives decline from 7.815 to 7.570 per day.', 60, 556, 1160, 40, 20, blue, true);
  text(slide, 'Frozen threshold: 14.600 total and 7.504 recovery false positives per day exceed budgets of 12 and 6.', 60, 606, 1160, 42, 18, red, true);
  footer(slide, 9, 'The Mean recovery difference has a date-block 95% interval [−0.585, 0.089], crossing zero. Historical backtest.');
  note(slide, 'This ranking comparison uses the same true-positive count at 60% recall on the full known support. Threshold selection uses a separate calibration split. Mean total false positives declined from 15.3778 to 14.7481 per day and recovery false positives declined from 7.8148 to 7.5704 per day. All three seeds moved in the same direction, though the recovery interval from 1,000 date-block bootstrap resamples crosses zero. The frozen operating threshold maximized F1 subject to the calibration constraints. Average backtest false positives still exceed both budgets. NYC test dates were already public, so the backtest lacks a new independent holdout.', [
    sourceB + 'results/matched_recall.csv', sourceB + 'results/paired_date_blocks.csv', sourceB + 'results/operating_points.csv',
  ]);
}

{
  const slide = title('Course Results and Reproduction', 'CONCLUSIONS AND FINAL DELIVERABLES');
  table(slide, [
    ['Project', 'Completed Work', 'Main Limits'],
    ['Project A', 'Cleaning, projection, clustering comparison\nParameter checks, stability, hotspots, and candidates', 'Capped 30-user sample\nNo ground truth for abnormal driving'],
    ['Project B', 'Strict lead, 3 pooling methods, 3 seeds\nDynamic edges and matched-support alerts', 'Historical NYC proxy backtest\nFalse alarm limits and original task gaps'],
  ], 60, 170, 1160, 236, [200, 530, 430], 18);
  text(slide, 'Reproduction', 60, 443, 1100, 38, 23, blue, true);
  text(slide, 'Project A: verify aggregates offline. A full rerun requires an authorized Geolife ZIP.\nProject B: use niulaoshi and ProjectB/run.py. Replay verifies outputs. Train reruns training.', 60, 498, 1160, 91, 20);
  text(slide, 'Source, configs, aggregate results, measured figures, speaker notes, and hash manifests are included.', 60, 608, 1160, 39, 19, black, true);
  footer(slide, 10, 'A: independent refit and 9 tests passed. B: 81 tests, 18 dynamic retrains, and 36 model replays verified.');
  note(slide, 'Project A contributes interpretable spatial analysis of hotspots, noise, and candidates. Supervised classification accuracy requires labels that were unavailable. Project B implements a Transformer constrained by the actual directed graph with dynamic edges. Thresholds were frozen after independent calibration. Results cover three pooling comparisons and false alarm limits. The release excludes plaintext individual trajectories and pickle files. A full Project A rerun requires an authorized original archive. Project B ships the public snapshot, arrays, and weights. The original Geolife requirements for user count, years, and five-minute persistence remain distinct from the supplemental NYC backtest.', [
    'Evidence/A_evidence.json', 'ProjectA/evidence/', 'ProjectB/local_run_verification.json', 'README_submission.md',
  ]);
}

if (deck.slides.items.length !== 10) throw new Error('Expected 10 slides');
await fs.mkdir(output, {recursive: true});
await fs.writeFile(path.join(output, 'speaker_notes.json'), JSON.stringify(notes, null, 2) + '\n');
const notesMarkdown = '# Speaker Notes\n\n' + notes.map(({slide, text, sources}) =>
  '## Slide ' + slide + '\n\n' + text + '\n\nEvidence\n\n' +
  sources.map(source => '- ' + source).join('\n'),
).join('\n\n') + '\n';
await fs.writeFile(path.join(output, 'speaker_notes.md'), notesMarkdown);
await (await PresentationFile.exportPptx(deck)).save(path.join(output, 'final_presentation_project_a_project_b.pptx'));
console.log('Exported the 10-slide presentation and speaker notes to ' + output);
