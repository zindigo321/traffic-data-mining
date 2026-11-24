# Original Coursework Requirements and Project Coverage

Source: `G:/course-jiaotongshujuwajue/ppt/Lecture1-logistics-DM2026V3-part1.pdf`, 46 physical pages. Page numbers count from the start of the PDF and match the footers. The original material was read without modification.

## Assessment

Page 8 allocates 30% to assignments, 60% to the group project, and 10% to quiz and participation. Project A, Project B, and the final presentation each account for 20% of the overall grade.

## Project A, Pages 9–10

The objective is to set up Python and scikit-learn, process Geolife trajectories, and identify travel patterns by clustering. The workflow covers loading, cleaning, feature extraction, modeling, and visualization through scatterplots and heatmaps. Tasks include tuning DBSCAN `eps` and `min_samples`, comparing K-means, studying hotspots and unusual trajectories across subjects, and reporting preprocessing, method choice, results, and use cases. Extensions cover time, road networks, and traffic events.

Project A used a controlled sample of 30 users, 53,786 points, and 1,102 trajectories. It ran 15 DBSCAN configurations, K-means for K = 2–10, a common-support comparison, a 500 m heatmap, time-of-day and low-density candidate analyses, and a stability check. Geolife covers several travel modes. The 115 candidates lack verified anomaly labels, and ARI and silhouette scores are internal clustering measures. Road-network integration and real traffic event validation remain outside the completed Project A work.

## Project B, Pages 11–15

Page 11 covers Geolife preprocessing, rush-hour features, and neural models such as CNN-LSTM and MLP. Page 12 calls for 100 users in 2010–2012, morning and evening peaks at 07:00–09:00 and 17:00–19:00, and timestamp, location, speed, and elevation. It calls for drift and duplicate removal, network matching, speed-change, volume, and time features, model training, and visualization.

Page 13 defines congestion as segment mean speed below 40% of the historical mean for the same period for at least five continuous minutes. Candidate features include road class, lanes, speed standard deviation, acceleration, and calendar information. The sample MLP has two or three hidden layers, learning rate 0.001 or 0.0001, and batch size 32 or 64. Page 14 calls for comparison with traditional methods such as random forests, analysis of factors, and a report. Page 15 proposes CNN-LSTM, user types, and simulated feature interventions as extensions.

The presentation reports the requested Graph Transformer method. The original Geolife project lacks trainable labels. The supplementary public NYC proxy uses 121 fixed directed segments, two consecutive 10-minute low-speed estimates relative to past baselines, and a strict 10-minute prediction lead time. It differs from the original conditions for 100 users, 2010–2012, the Beijing road network, and five minutes of continuous congestion. Causal interpretation and intervention simulation remain outside the evidence.

## Original Defense Identity Review

The public copy omits personal bylines from the original defense and assignments.

The historical review extracted text from the source PDF using the bundled Python and pypdf and inspected the original PPTX through zipfile and XML. It opened and executed no pickle files.
