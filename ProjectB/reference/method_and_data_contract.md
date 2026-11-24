# Dynamic Edge Weight Method and Data Timing Contract

Nodes are the 121 original directed road segments. The original source-to-target directions, 56 candidate edges, 54 isolated nodes, and one-hop local graphs are used. Graphs in a batch are independent. The target maps to local node 0 and attends only to physical predecessors and itself. Pooling covers the target's local subgraph. The input tensor `[B,N,104]` contains 52 train-fitted imputed and standardized values plus their 52 missingness indicators, with `N ≤ 5`. See `evidence/cohort_final.json` for the 52 fields. The experiment uses public segment-level data.

The descriptors use `lag20_relative_speed`, `lag30_relative_speed`, `lag40_relative_speed`, `lag20_minus_lag30_ratio`, `legal_usable_count`, `lag20_message_count`, `lag20_history_days`, and `lag20_half_last_ratio`. Each endpoint contributes eight values and eight original missingness indicators. Values are clipped to standardized ±5 for edge descriptors only. Concatenating source, target, and their elementwise absolute difference yields 48 dimensions. The latest dynamic information is bin t−2. The t−1 and t speeds used in the label and future observations are excluded. Here t is the target window start, scoring occurs at t−10 minutes, and t−2 is [t−20,t−10).

For layer l, head h, and physical edge j to i:

`b[l,h,i,j] = log(2) * tanh(MLP_l([u_j,u_i,abs(u_j-u_i)]))[h]`

`w[l,h,i,j] = exp(b[l,h,i,j]) ∈ [0.5,2]`

`attention = softmax_j(Q_i K_j / sqrt(8) + b[l,h,i,j] + directed_mask)`

The MLP has layers 48→16→4 with GELU. It adds 852 parameters per layer and 1,704 across both layers. Zero initialization of the final linear layer gives all edges initial weight 1 and preserves the backbone and classifier initialization. Self-edge weights stay at 1. A missing endpoint lag20 relative speed also yields weight 1. The mask excludes nonphysical edges. The original QK attention is already input-dependent. Dynamic edge weights add an explicit, auditable multiplicative prior based on legal past traffic attributes. Dynamic parameters learn only from training labels.

Each of the two layers has a `[B,4,N,N]` prior tensor. Physical source-to-target edges map to source keys and target queries. Local attention uses dense computation. Per-layer backbone cost is `O(BN²D + BND²)`, with edge MLP cost `O(BN²(48×16+16×4))` and additional space `O(BHN²)`. Edge weights create no additional roads or cross-batch connections. An isolated graph has only the unit self edge, so toggling dynamic weights produces exactly identical outputs for the same frozen backbone.

Mean pooling averages valid node vectors elementwise. Max pooling takes an elementwise maximum over valid nodes. Attention pooling uses `a_i=softmax(vᵀh_i+c)` and `sum(a_i h_i)`. Each costs `O(BND)`. Concatenate the target node and pooled vectors into `[B,64]` and output one binary logit per target road segment. Reject empty graphs and invalid centers. Padding is excluded from pooling and valid edges.

Use New York local time, 07:00–09:00 and 17:00–19:00, and the original two-bin low-speed label, historical support, and unknown definitions. Past observations in training form the historical baseline and are frozen in subsequent periods. Fit standardization on training inputs only. Edge weights do not depend on current ground truth, risk, later labels, evaluation classes, or sampling. Risk is a t−1 low-speed indicator used for independent calibration of false-positive limits and retrospective recovery diagnostics. Training, routing, and edge weight computations exclude it.

The original data sources, download times, queries, and hashes were inherited and verified against `source_snapshot.json`. This run neither fetched new dates nor rechecked the source terms. See `source_provenance/` for the official source and usage records. This historical audit directory refers to the read-only speed shards and prepared arrays in `evidence/path_mapping.json` and `evidence/prepared.json`. The full input-to-metric pipeline is in `project/full_*.py`. Offline `data_as_of` provides the original assumed cutoff, with no separate observation-arrival timestamp.
