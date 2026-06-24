<milestone number="1" title="Engineering, Mathematical & Data Foundations for ML Systems">
<domain_scope>
By the end of M1 the learner can write correct, performant, well-tested Python systems code; reason about data structures, complexity, and numerical stability; manipulate data at scale with modern DataFrame/SQL tooling; and stand up reproducible compute environments. This is the prerequisite layer for every later milestone: each subsequent milestone assumes the engineer can ship clean code, query data correctly without leakage, and reason mathematically about what models actually compute. Concretely, the learner can take a messy real-world dataset, build a reproducible, version-controlled, containerized pipeline that loads larger-than-memory data, and reason about the linear-algebra and probability primitives that underlie embeddings, optimization, and attention. Without this layer, every model the engineer later builds will be fragile, irreproducible, or quietly wrong.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to write idiomatic, fully type-annotated Python (validated with mypy/ruff), structure it into packages, and achieve meaningful test coverage with pytest including property-based tests for numerical code.
- By the end of this milestone, the engineer should be able to derive and implement gradient descent, the chain rule, and backpropagation from scratch in pure NumPy without relying on autograd.
- By the end of this milestone, the engineer should be able to compute and apply SVD/eigendecomposition and explain low-rank approximation for embedding compression and for understanding LoRA-style adapters later.
- By the end of this milestone, the engineer should be able to write point-in-time-correct ("as-of") joins and explain precisely how temporal leakage inflates offline metrics relative to online performance.
- By the end of this milestone, the engineer should be able to process larger-than-memory datasets with Polars/DuckDB streaming and reason about columnar Arrow/Parquet layouts and zero-copy interchange.
- By the end of this milestone, the engineer should be able to reason about floating-point precision (fp32/bf16/fp8), implement a numerically stable softmax/log-sum-exp, and predict where naive implementations overflow.
- By the end of this milestone, the engineer should be able to design a relational + object-storage schema, version data and pipelines with DVC/lakeFS, and reproduce any artifact from a commit hash.
</competency_targets>
<exhaustive_topics>
- **SVD and low-rank approximation for embeddings and compression.** Why it matters: underpins dimensionality reduction, PCA, embedding compression, and conceptually LoRA. Foundations: orthogonal decomposition, singular values as energy, Eckart–Young optimality. Production realities: truncated/randomized SVD on sparse matrices; numerical rank vs mathematical rank. Pitfalls: treating tiny singular values as signal; centering before PCA. Senior signal: can explain when a rank-k approximation preserves task-relevant variance and when it destroys it.
- **Eigendecomposition and positive semi-definite matrices.** Why: covariance, kernels, optimization curvature. Foundations: spectral theorem, PSD-ness, condition number. Pitfalls: assuming symmetry where none exists; ill-conditioning. Senior signal: links condition number to optimization difficulty.
- **Matrix calculus, Jacobians, and the chain rule.** Why: backprop is mechanical chain rule over tensors. Foundations: vector-Jacobian products, reverse-mode AD. Pitfalls: shape mismatches, transposition errors. Senior signal: hand-derives gradients of a 2-layer MLP and matches autograd.
- **Probability and estimation (MLE, Bayes, conjugacy).** Why: loss functions are negative log-likelihoods. Foundations: likelihood, priors, MAP. Pitfalls: confusing likelihood with probability; improper priors. Senior signal: derives cross-entropy as MLE of a categorical.
- **Numerical precision and stable softmax/log-sum-exp.** Why: fp16/bf16/fp8 training overflows naive code. Foundations: max-subtraction trick, Kahan summation. Pitfalls: exp() overflow, catastrophic cancellation. Senior signal: predicts where fp16 NaNs appear before running code.
- **Gradient descent variants derived by hand (SGD, momentum, Adam).** Why: every training loop depends on them. Foundations: bias-corrected moments, decoupled weight decay. Pitfalls: coupling LR and weight decay; wrong epsilon placement. Senior signal: explains why AdamW differs from Adam+L2.
- **Python performance: vectorization, GIL, multiprocessing, async.** Why: data pipelines and serving are throughput-bound. Foundations: CPython memory model, NumPy broadcasting. Pitfalls: hidden Python loops, GIL contention. Senior signal: profiles and removes the dominant bottleneck.
- **Type systems and testing discipline.** Why: ML code rots silently. Foundations: static typing, fixtures, property tests. Pitfalls: untested numerical edge cases. Senior signal: writes tests that catch shape/dtype regressions.
- **Columnar formats: Parquet and Arrow zero-copy.** Why: the substrate of modern data tooling. Foundations: column chunks, encodings, predicate pushdown. Pitfalls: row-oriented thinking; small-file problems. Senior signal: reasons about scan cost from layout.
- **Polars lazy/streaming engine and DuckDB.** Why: larger-than-memory processing without Spark. Foundations: lazy query plans, out-of-core execution, Arrow interchange. Pitfalls: accidental materialization, eager collect(). Senior signal: builds a streaming pipeline that never loads the full dataset.
- **SQL window functions and point-in-time joins.** Why: feature computation correctness. Foundations: partition/order semantics, as-of joins. Pitfalls: temporal leakage, timezone bugs. Senior signal: validates with anti-joins that no feature timestamp exceeds the label timestamp.
- **Data and pipeline versioning (DVC, lakeFS).** Why: reproducibility. Foundations: content-addressable storage, branching. Pitfalls: versioning code but not data. Senior signal: reproduces a result from a commit + data version.
- **Git, CI, and Docker for reproducible environments.** Why: parity between dev/train/serve. Foundations: layer caching, pinned deps. Pitfalls: latest tags, unpinned CUDA. Senior signal: produces bit-reproducible container builds.
- **Complexity and data structures for ML.** Why: hashing/trees/heaps recur in retrieval and feature stores. Foundations: amortized analysis. Pitfalls: O(n²) joins hidden in pandas. Senior signal: chooses the right structure for the access pattern.
</exhaustive_topics>
<production_tech_stack>
Core languages and frameworks: Python 3.12+, NumPy, type hints, pytest, ruff, mypy — correctness and performance floor for everything later. Data manipulation/storage/query: Polars (lazy/streaming), DuckDB, pandas, Parquet/Arrow, PostgreSQL/SQL — larger-than-memory processing and leakage-free joins. Experimentation/tracking/reproducibility: Git, DVC or lakeFS, Docker — reproducible artifacts from commit hashes. Training/compute/acceleration: local CPU + entry-level GPU (the math, not scale, is the focus here). Serving/deployment/orchestration: none yet — introduced in M4/M5. Monitoring/observability/quality: basic data-quality assertions. Cloud/infra primitives: an object store (S3/GCS) and containers. Each tool earns its place because M1 is about correctness, reproducibility, and scale fundamentals before any model is trained.
</production_tech_stack>
<strategic_capstone_project deploy="false"></strategic_capstone_project>
</milestone>

<milestone number="2" title="Classical ML, Experimentation Discipline & Evaluation Rigor">
<domain_scope>
By the end of M2 the learner can frame a supervised problem, build leakage-free pipelines, train and tune gradient-boosted trees and linear/logistic models, and — most critically — design evaluations that predict production performance rather than overfit a leaderboard. Gradient-boosted trees remain the dominant approach for tabular problems in 2026, so this milestone is not legacy material: it is where the engineer internalizes the discipline (cross-validation design, metric selection, leakage detection, experiment tracking) that separates engineers who ship reliable models from those who ship impressive-looking ones that collapse online. This discipline transfers directly to evaluating deep-learning and LLM systems later.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to build scikit-learn Pipelines that prevent train/test leakage by fitting all transforms (scaling, encoding, imputation) only on training folds.
- By the end of this milestone, the engineer should be able to train, regularize, and tune XGBoost/LightGBM and articulate the bias-variance tradeoff in terms of tree depth, learning rate, and number of estimators.
- By the end of this milestone, the engineer should be able to design cross-validation schemes appropriate to the data — grouped, time-series, stratified, nested — and actively detect leakage.
- By the end of this milestone, the engineer should be able to select metrics matched to business cost (PR-AUC vs ROC-AUC under imbalance, calibration, cost-sensitive thresholds) and justify the choice.
- By the end of this milestone, the engineer should be able to track experiments reproducibly (MLflow / Weights & Biases) capturing seeds, data versions, and code commits.
- By the end of this milestone, the engineer should be able to explain model behavior with SHAP, run slice-based error analysis, and identify failing subpopulations.
</competency_targets>
<exhaustive_topics>
- **Gradient-boosted trees (XGBoost, LightGBM) internals.** Why: the 2026 default for tabular data. Foundations: additive boosting, gradient/hessian, leaf-wise vs level-wise growth, regularization (lambda, gamma, min_child_weight). Pitfalls: overfitting via depth, target leakage through engineered features. Senior signal: tunes for generalization, not training loss.
- **Linear/logistic regression and regularization paths.** Why: interpretable baselines. Foundations: convex optimization, L1/L2, coefficient paths. Pitfalls: unscaled features, multicollinearity. Senior signal: uses a strong linear baseline before reaching for complexity.
- **Feature engineering and encoding pitfalls.** Why: most tabular gains live here. Foundations: target/ordinal/hashing encodings, OOV handling. Pitfalls: target leakage via mean-encoding without fold isolation. Senior signal: encodes inside CV folds.
- **Cross-validation design.** Why: the single biggest source of optimistic bias. Foundations: time-series splits, grouped CV, nested CV for tuning. Pitfalls: random CV on temporal/grouped data. Senior signal: matches the split to the deployment regime.
- **Class imbalance and resampling.** Why: real problems are skewed. Foundations: SMOTE, class weights, threshold moving. Pitfalls: resampling before CV split. Senior signal: prefers threshold tuning + proper metrics over naive oversampling.
- **Probability calibration.** Why: downstream decisions need true probabilities. Foundations: Platt scaling, isotonic regression, reliability diagrams. Pitfalls: calibrating on training data. Senior signal: reports calibration alongside discrimination.
- **Metric selection and threshold optimization.** Why: a model "improves" only if the right metric moves. Foundations: PR vs ROC, F-beta, expected cost. Pitfalls: accuracy on imbalanced data, mismatched offline/online metrics. Senior signal: ties the metric to a business cost function.
- **Data leakage taxonomy.** Why: the canonical reason offline beats online. Foundations: target, temporal, group, preprocessing leakage. Pitfalls: fitting scalers on full data. Senior signal: can name the leakage class from a symptom (offline AUC 0.95 → online 0.78).
- **Experiment tracking and reproducibility.** Why: science requires it. Foundations: run metadata, seeds, lineage. Pitfalls: untracked hyperparameters. Senior signal: any run is reproducible end to end.
- **SHAP and feature-importance shift.** Why: explanation + drift signal. Foundations: Shapley values, additive attributions. Pitfalls: interpreting SHAP causally. Senior signal: uses SHAP to debug, not to decorate.
- **Hyperparameter search.** Why: efficiency. Foundations: Bayesian optimization, Hyperband/ASHA. Pitfalls: tuning on the test set. Senior signal: budgets search and uses nested CV.
- **Error analysis and slice-based evaluation.** Why: aggregate metrics hide failures. Foundations: cohort analysis, confusion slicing. Pitfalls: shipping on a single number. Senior signal: reports per-slice performance.
- **Baseline discipline.** Why: complexity must justify itself. Foundations: majority/heuristic/linear baselines. Senior signal: never deploys a complex model that fails to beat a simple one.
</exhaustive_topics>
<production_tech_stack>
Core: scikit-learn, XGBoost, LightGBM, Python. Data: Polars/pandas, DuckDB, Parquet. Experimentation/tracking: MLflow and/or Weights & Biases, basic model registry. Training/compute: CPU/single GPU. Serving/orchestration: not yet. Monitoring/quality: SHAP, Evidently for offline data-quality and distribution checks. Cloud/infra: object store + tracking server. Each belongs because M2 is about disciplined modeling and evaluation rigor before deep learning.
</production_tech_stack>
<strategic_capstone_project deploy="false"></strategic_capstone_project>
</milestone>

<milestone number="3" title="Deep Learning Foundations, Optimization & Training Mechanics">
<domain_scope>
By the end of M3 the learner can implement, train, and debug neural networks — including transformers — in PyTorch, understanding autograd, optimization dynamics, mixed precision, memory, and the failure modes of unstable training. PyTorch is the dominant framework in 2026; JAX appears where functional/TPU workloads demand it. This is the transition into model-centric systems work: the engineer moves from "fit a model" to "train an architecture I understand at the kernel and memory level."
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to implement a transformer (self-attention, multi-head, MLP, residual connections, normalization, positional encoding) from scratch in PyTorch and match a reference.
- By the end of this milestone, the engineer should be able to diagnose and fix unstable training: exploding/vanishing gradients, loss spikes, dead ReLUs, NaNs.
- By the end of this milestone, the engineer should be able to use mixed precision (bf16, fp8) and gradient checkpointing correctly and explain their memory/throughput/accuracy tradeoffs.
- By the end of this milestone, the engineer should be able to implement learning-rate warmup, schedules, and gradient clipping and explain why each stabilizes training.
- By the end of this milestone, the engineer should be able to profile GPU utilization and distinguish memory-bound from compute-bound bottlenecks.
- By the end of this milestone, the engineer should be able to reason about FlashAttention and why IO-aware kernels matter for both speed and memory.
</competency_targets>
<exhaustive_topics>
- **Autograd and reverse-mode differentiation.** Why: the engine of training. Foundations: computational graphs, VJPs, retain_graph semantics. Pitfalls: in-place ops breaking the graph, detached tensors. Senior signal: debugs a broken gradient by reading the graph.
- **Transformer architecture, fully decomposed.** Why: the substrate of modern AI. Foundations: scaled dot-product attention, multi-head projection, RoPE positional encoding, GQA/MQA/MLA for KV reduction. Pitfalls: wrong masking, positional encoding bugs. Senior signal: implements and explains each subcomponent's role.
- **Normalization (LayerNorm, RMSNorm).** Why: stability. Foundations: normalization statistics, pre- vs post-norm. Pitfalls: norm placement causing divergence. Senior signal: chooses placement for deep stacks.
- **Optimizers (Adam/AdamW, weight-decay decoupling).** Why: convergence. Foundations: moment estimates, decoupled decay. Pitfalls: coupling LR and decay. Senior signal: explains AdamW's advantage.
- **LR schedules and warmup.** Why: early-training stability. Foundations: linear warmup, cosine decay. Pitfalls: no warmup → early divergence. Senior signal: ties schedule to batch size and model scale.
- **Mixed precision (bf16, fp8) and loss scaling.** Why: throughput and memory. Foundations: dynamic range vs precision, master weights. Pitfalls: fp16 overflow without scaling. Senior signal: picks bf16 vs fp8 by hardware and stability.
- **Gradient checkpointing / activation recomputation.** Why: fit bigger models. Foundations: recompute-vs-store tradeoff. Pitfalls: checkpointing the wrong modules. Senior signal: trades compute for memory deliberately.
- **FlashAttention-1/2/3 and IO-awareness.** Why: attention is the bottleneck. Foundations: tiling, streaming softmax, Hopper asynchrony (warp specialization, TMA, WGMMA), FP8 attention. Pitfalls: assuming all attention kernels are equal. Senior signal: explains why minimizing HBM traffic, not FLOPs, is the win.
- **Initialization and residual scaling.** Why: deep nets diverge otherwise. Foundations: Xavier/Kaiming, residual rescaling. Senior signal: links init to gradient flow.
- **Regularization (dropout, label smoothing, weight decay).** Senior signal: applies the right tool for the failure observed.
- **Training instability diagnosis.** Why: real training breaks. Foundations: gradient norms, activation statistics. Pitfalls: silent NaNs from bad data. Senior signal: instruments before it breaks.
- **Data loading bottlenecks.** Why: GPUs starve. Foundations: prefetch, num_workers, pinned memory. Senior signal: keeps the GPU fed.
- **GPU memory model.** Why: OOMs dominate early. Foundations: activation vs parameter vs optimizer-state memory. Senior signal: budgets memory before launching.
- **JAX where relevant.** Why: functional transforms, TPUs. Senior signal: knows when JAX's vmap/pmap beats PyTorch.
</exhaustive_topics>
<production_tech_stack>
Core: PyTorch (dominant), JAX (where relevant), CUDA basics. Data: Parquet/Arrow datasets, efficient DataLoaders. Experimentation/tracking: Weights & Biases, TensorBoard. Training/compute/acceleration: single and multi-GPU A100/H100, FlashAttention, torch.compile, AMP/bf16, Nsight/torch profiler. Serving/orchestration: not yet. Monitoring/quality: gradient/activation logging. Cloud/infra: GPU instances + object store. Each belongs because M3 is about training mechanics and the kernels and memory beneath them.
</production_tech_stack>
<strategic_capstone_project deploy="true">
**Project Title:** Train a small transformer end-to-end with full training observability.
**Why this belongs at this transition:** This is the first capstone — the move into deep-learning systems work. The engineer must prove they can train a real architecture, not just call `.fit()`.
**Problem Statement:** Pretrain or fine-tune a ~100M-parameter transformer on a real corpus with reproducible, fully instrumented training.
**Users/stakeholders:** ML engineers and the team that will reuse the training harness.
**Data sources and contracts:** A tokenized corpus with a documented schema, token-count statistics, and a versioned snapshot (DVC/lakeFS).
**End-to-end architecture:** data ingestion → tokenization → instrumented training loop with checkpointing and resume → held-out eval harness → Weights & Biases logging of loss, gradient norms, throughput, and GPU utilization.
**Hard engineering constraints:** must fit in available GPU memory using bf16 + gradient checkpointing; must produce a stable, monotonically improving loss curve; must support deterministic resume-from-checkpoint.
**Failure modes to expect:** loss spikes, silent NaNs, data-loader starvation, non-reproducible runs, fp16 overflow.
**Exact implementation stack:** PyTorch, FlashAttention, torch.compile, W&B, Parquet/Arrow data, Docker. Note that FlashAttention-2 (Dao, ICLR 2024) reaches up to ~225 TFLOPs/s and ~72% FLOPs utilization on A100, which contextualizes the capstone's ">50% GPU utilization" target as a deliberately conservative, achievable bar rather than a stretch goal.
**Evidence of a strong engineering standard:** reproducible loss curves; a written log of at least one diagnosed-and-fixed instability; profiler output showing >50% GPU utilization; a demonstrated clean resume-from-checkpoint that continues the loss curve without discontinuity.
</strategic_capstone_project>
</milestone>

<milestone number="4" title="Data Pipelines, Scalable Training & Distributed Systems Infrastructure">
<domain_scope>
By the end of M4 the learner can build production data pipelines (batch + streaming) on a lakehouse, orchestrate workflows reproducibly, operate a feature store, and train models across multiple GPUs and nodes using modern parallelism. This is the systems-centric backbone that makes large-scale ML possible and the layer where most "the model was fine but the system failed" problems originate: train/serve skew, data freshness failures, GPU communication bottlenecks, and irreproducible pipelines.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to design a lakehouse on open table formats (Iceberg/Delta) with ACID transactions, schema evolution, time travel, and a REST catalog.
- By the end of this milestone, the engineer should be able to build batch and streaming pipelines (Spark, Flink/Kafka) with explicit data contracts and quality gates.
- By the end of this milestone, the engineer should be able to choose and configure FSDP2, DeepSpeed ZeRO, or Megatron-Core parallelism for a given model size and cluster topology.
- By the end of this milestone, the engineer should be able to tune NCCL/communication, interpret MFU, and diagnose stragglers and comms bottlenecks.
- By the end of this milestone, the engineer should be able to operate a feature store with point-in-time correctness to eliminate train/serve skew.
- By the end of this milestone, the engineer should be able to orchestrate reproducible, observable pipelines with Dagster/Flyte/Prefect/Airflow and reason about which fits the team.
</competency_targets>
<exhaustive_topics>
- **Open table formats (Iceberg, Delta Lake, Hudi) and REST catalogs.** Why: the lakehouse foundation. Foundations: manifest/snapshot metadata, ACID via optimistic concurrency, partition/schema evolution, time travel; catalogs (Polaris, Unity, Nessie). Pitfalls: small-file/compaction debt, positional-delete accumulation. Senior signal: chooses a format by engine ecosystem and mutation pattern.
- **Streaming (Kafka, Flink, RisingWave).** Why: real-time features. Foundations: watermarks, exactly-once, windowing. Pitfalls: late-event handling differing from batch. Senior signal: reasons about end-to-end exactly-once.
- **Batch (Spark) and dbt transformations.** Why: large-scale ETL + declarative SQL models. Pitfalls: skew, shuffle explosions. Senior signal: optimizes partitioning and join strategy.
- **Data contracts and quality (Great Expectations, Soda).** Why: upstream breakage silently corrupts models. Foundations: schema/constraint validation at hand-offs. Senior signal: gates pipelines on contracts.
- **DataFrame engines at scale (Polars streaming, Daft, DuckDB).** Why: not everything needs Spark. Senior signal: picks single-node out-of-core when it suffices.
- **Workflow orchestration (Dagster, Flyte, Metaflow, Prefect, Airflow).** Why: the "when and in what order." Foundations: DAGs, assets, retries, backfills. Pitfalls: orchestration sprawl, non-idempotent tasks. Senior signal: matches tool to team (Metaflow for Python-first ML, Flyte for K8s-native typed pipelines, Prefect/Dagster for modern DX, Airflow for legacy).
- **Parallelism taxonomy: data vs tensor vs pipeline vs sequence/context.** Why: no single strategy scales everything. Foundations: communication/computation tradeoffs of each. Senior signal: composes 3D parallelism for the model+cluster.
- **FSDP2 sharding and prefetch.** Why: PyTorch-native default in 2026. Foundations: per-parameter sharding, overlapped all-gather. Senior signal: knows when FSDP2 is the path of least friction vs Megatron.
- **DeepSpeed ZeRO stages and offload.** Why: memory-bound and MoE training. Foundations: ZeRO-1/2/3, CPU/NVMe offload. Senior signal: picks the stage by memory pressure.
- **Megatron-Core 3D parallelism and Transformer Engine.** Why: frontier-scale. Foundations: tensor+pipeline+data, FP8 via TE, non-uniform sharding. Senior signal: reserves it for very-large-scale runs where MFU matters.
- **NCCL/InfiniBand/NVLink/SHARP communication.** Why: comms is the scaling bottleneck. Pitfalls: BF16 ring-reduce numerical instability at large DP sizes (e.g., DP ≥ 128). Senior signal: tunes the fabric and validates reduction precision.
- **fp8 distributed training.** Why: throughput and memory at scale. Pitfalls: stability vs inference quantization. Senior signal: applies selective precision.
- **Distributed checkpointing and resharding.** Why: failure recovery and parallelism-agnostic restarts. Senior signal: restarts a run under a different parallelism layout.
- **Feature stores (Feast, Tecton) and point-in-time correctness.** Why: the structural fix for train/serve skew. Foundations: offline/online stores, as-of joins, shared transformation code. Pitfalls: offline AUC inflated 5–20% by temporal leakage when as-of joins are wrong. Senior signal: enforces point-in-time joins by construction.
- **Train/serve skew sources.** Why: the classic production failure. Foundations: batch/stream dual-write divergence, categorical vocab mismatch, embedding-service versioning as a feature contract. Senior signal: unifies code paths and version-pins encoders.
</exhaustive_topics>
<production_tech_stack>
Core: PyTorch, FSDP2, DeepSpeed, Megatron-Core, TorchTitan, Transformer Engine. Data manipulation/storage/query: Spark, Flink, Kafka/RisingWave, Iceberg/Delta + Polaris/Unity catalog, Polars, DuckDB, dbt. Experimentation/tracking: W&B/MLflow, distributed checkpoint stores. Training/compute/acceleration: multi-GPU/multi-node H100/H200, NCCL, InfiniBand/NVLink. Serving/orchestration: Dagster/Flyte/Prefect/Airflow for pipeline orchestration. Monitoring/quality: Great Expectations/Soda data-quality, MFU/throughput telemetry. Cloud/infra: Kubernetes, SLURM, S3/object store, Terraform. Each belongs because M4 is about scaling data and training as distributed systems.
</production_tech_stack>
<strategic_capstone_project deploy="false"></strategic_capstone_project>
</milestone>

<milestone number="5" title="Production Deployment, Serving, Observability & Lifecycle Management">
<domain_scope>
By the end of M5 the learner can take a trained model, deploy it behind a reliable serving layer, monitor it for drift and silent degradation, and own its full lifecycle including progressive delivery, rollback, and retraining. This is the transition into production ML systems ownership: the engineer becomes responsible not for a model artifact but for a living service with SLOs, on-call, and the failure modes — train/serve skew in production, drift, label delay, alert fatigue, and stateful-rollback complexity — that only reveal themselves after launch.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to package and serve models (BentoML/KServe/Ray Serve/Seldon/Triton) with autoscaling, health checks, and a model registry source of truth.
- By the end of this milestone, the engineer should be able to implement progressive delivery (canary, shadow/mirror, blue-green) and execute a fast, safe rollback.
- By the end of this milestone, the engineer should be able to detect data, concept, and prediction drift (Evidently, NannyML) and estimate performance without labels.
- By the end of this milestone, the engineer should be able to build CI/CD for ML with registration-time evaluation gates that block regressions.
- By the end of this milestone, the engineer should be able to instrument latency/throughput/cost, define SLOs/SLIs, and design alerting that avoids fatigue.
- By the end of this milestone, the engineer should be able to design retraining triggers and write incident-response runbooks and postmortems.
</competency_targets>
<exhaustive_topics>
- **Model packaging and registries (MLflow, BentoML).** Why: a versioned, deployable source of truth. Foundations: artifact + metadata + lineage + deployment status. Senior signal: every deployed model traces to a registry entry.
- **Serving frameworks (KServe, Ray Serve, Seldon, Triton, BentoML).** Why: reliable inference at scale. Foundations: containerized endpoints, adaptive batching, traffic splitting. Pitfalls: cold starts, GPU packing. Senior signal: picks BentoML for simple REST, KServe/Seldon for K8s traffic management, Triton/Ray for high-throughput.
- **Batch vs online vs streaming serving.** Why: latency/cost shape architecture. Senior signal: matches serving mode to the SLA.
- **Autoscaling (HPA on request rate/queue depth).** Why: cost vs latency. Pitfalls: scaling on CPU for GPU workloads. Senior signal: scales on the true bottleneck signal.
- **Progressive delivery (canary, shadow, blue-green, Argo Rollouts).** Why: safe releases. Foundations: traffic shaping, mirrored traffic, success metrics. Senior signal: defines model-specific rollout success criteria.
- **Rollback complexity for stateful systems.** Why: rolling back a stateful pipeline is not just redeploying. Foundations: in-flight requests, feature/version compatibility. Senior signal: designs rollback before launch.
- **Drift detection (PSI, KS, Chi-square, Jensen-Shannon/Wasserstein, MMD, classifier-based, embedding cosine drift).** Why: models fail silently. Foundations: distributional distance and two-sample tests. Pitfalls: alerting on benign feature drift. Senior signal: links data drift to expected model impact.
- **Performance estimation without labels (NannyML CBPE/DLE).** Why: labels arrive late or never. Foundations: confidence-based and direct-loss estimation. Senior signal: estimates online accuracy under covariate shift.
- **ML observability stack (Evidently, Arize, WhyLabs, Fiddler).** Why: the modern monitoring layer. Foundations: profiles, dashboards, embedding monitors. Senior signal: integrates monitoring into prediction pipelines, not as an afterthought.
- **OpenTelemetry GenAI conventions.** Why: standardized telemetry. Senior signal: instruments against the spec.
- **CI/CD for ML and evaluation gates.** Why: prevent regressions. Foundations: registration-time eval, fairness/accuracy thresholds. Senior signal: blocks merges on regression.
- **Cost monitoring and GPU utilization.** Why: cost explosions are a top production failure. Senior signal: tracks cost-per-prediction and idle GPU.
- **SLOs/SLIs and alerting taxonomy.** Why: reliability is measurable. Foundations: error budgets, p99 latency. Pitfalls: alert fatigue. Senior signal: alerts only on actionable, user-impacting signals.
- **Incident response and postmortems.** Why: systems fail; learning compounds. Senior signal: runs blameless postmortems with concrete action items.
- **Train/serve skew detection in production.** Why: the recurring silent killer. Senior signal: continuously compares training and serving feature distributions.
</exhaustive_topics>
<production_tech_stack>
Core: Python, FastAPI, Docker, Kubernetes. Data: feature store (Feast/Tecton), Parquet. Experimentation/tracking: MLflow registry, W&B. Training/compute: retraining pipelines on prior M4 infra. Serving/deployment/orchestration: BentoML, KServe, Ray Serve, Seldon, Triton; Argo Rollouts for progressive delivery. Monitoring/observability/quality: Evidently, NannyML, Arize/WhyLabs/Fiddler, Prometheus/Grafana, OpenTelemetry. Cloud/infra: SageMaker/Vertex/Azure ML primitives, Terraform, HPA. Each belongs because M5 is about reliable production ownership and lifecycle management.
</production_tech_stack>
<strategic_capstone_project deploy="true">
**Project Title:** End-to-end production ML service with drift monitoring and rollback.
**Why this belongs at this transition:** This is the second capstone — the move into production ML systems ownership. The engineer must prove they can own a living service, not hand off a model.
**Problem Statement:** Serve a tabular ranking/classification model behind an API with the full lifecycle: features, training, eval gate, canary serving, monitoring, and retraining.
**Users/stakeholders:** A product surface and the on-call engineers who inherit the service.
**Data sources and contracts:** A feature store with point-in-time correctness; documented feature schema, freshness SLA, and ownership.
**End-to-end architecture:** ingestion → feature pipeline (point-in-time joins) → training pipeline → offline evaluation gate → online serving with canary rollout → drift and performance monitoring → automated retraining trigger and feedback loop.
**Hard engineering constraints:** explicit p99 latency budget; a cost-per-prediction ceiling; drift alerting within a defined window; enforced train/serve feature parity.
**Failure modes to expect:** train/serve skew, data/concept drift, delayed labels, rollback of in-flight requests, alert fatigue, stale features.
**Exact implementation stack:** Feast, BentoML or KServe, Evidently/NannyML, MLflow registry, Argo Rollouts, Kubernetes, Prometheus/Grafana.
**Evidence of a strong engineering standard:** a demonstrated canary deployment followed by a clean rollback; a drift alert that fires on a deliberately injected distribution shift; a passing train/serve parity test; an SLO dashboard showing latency, cost, and quality; a written runbook for at least one incident class.
</strategic_capstone_project>
</milestone>

<milestone number="6" title="LLM Systems: Retrieval, Fine-Tuning, Inference Optimization & Agentic Architectures">
<domain_scope>
By the end of M6 the learner can build, optimize, and rigorously evaluate production LLM systems: self-hosted serving stacks, RAG pipelines, fine-tuned adapters, and tool-using agents. This milestone reflects the fastest-moving layer of 2026 practice and is explicitly evaluated against current standards — vLLM and SGLang as the dominant open serving engines (HuggingFace TGI entered maintenance mode in December 2025 with HuggingFace itself recommending vLLM or SGLang), prefill/decode disaggregation as standard at scale, MCP as the de facto tool-calling protocol, and eval-driven development as non-negotiable. This is the transition into LLM/agentic systems ownership.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to deploy and tune an inference engine (vLLM/SGLang/TensorRT-LLM) with continuous batching, paged KV cache, FP8, and prefix caching, and benchmark TTFT/ITL/throughput.
- By the end of this milestone, the engineer should be able to reason about prefill/decode disaggregation, speculative decoding, and KV-cache quantization (FP8/INT4) tradeoffs and when each is justified.
- By the end of this milestone, the engineer should be able to build a production RAG pipeline with hybrid search, cross-encoder reranking, deliberate chunking, and retrieval evaluation.
- By the end of this milestone, the engineer should be able to fine-tune with LoRA/QLoRA and apply preference optimization (DPO/ORPO/KTO/GRPO), then evaluate against task metrics and capability regression.
- By the end of this milestone, the engineer should be able to build agents with tool-calling/MCP and evaluate trajectories, tool-call accuracy, and task-completion rate.
- By the end of this milestone, the engineer should be able to design LLM evaluation combining offline golden sets, online tracing, and LLM-as-judge, and detect prompt-eval mismatch.
- By the end of this milestone, the engineer should be able to apply context engineering (compaction, structured memory, sub-agent isolation) to long-horizon agents.
</competency_targets>
<exhaustive_topics>
- **Inference engines (vLLM, SGLang, TensorRT-LLM).** Why: serving choice drives latency and cost. Foundations: vLLM's flexibility/dynamic workloads, SGLang's RadixAttention prefix caching, TensorRT-LLM's compiled peak throughput on NVIDIA. Pitfalls: TensorRT-LLM's build step and vendor lock-in; TGI now maintenance-only. Senior signal: matches engine to workload shape, not benchmark rank.
- **Continuous/in-flight batching.** Why: the single biggest throughput unlock. Foundations: iteration-level scheduling. Senior signal: explains why it beats static batching.
- **PagedAttention vs RadixAttention prefix caching.** Why: KV memory efficiency and shared-prefix reuse. Foundations: virtual-memory paging vs radix-tree prefix reuse. Senior signal: exploits shared system prompts in multi-turn/agentic workloads.
- **Prefill/decode disaggregation.** Why: prefill is compute-bound, decode is memory-bandwidth-bound; co-location causes interference. Foundations: DistServe goodput, NIXL KV transfer, NVIDIA Dynamo/llm-d orchestration, Mooncake. Pitfalls: KV-transfer bandwidth cost; not worth it for short prompts/small models. Senior signal: sizes network for KV transfer and knows when disaggregation hurts.
- **Speculative decoding.** Why: faster generation. Foundations: draft-then-verify, acceptance rate. Senior signal: weighs draft-model overhead vs speedup.
- **KV-cache management and quantization (FP8, INT4/INT8).** Why: KV cache dominates memory at long context. Foundations: per-token KV size, key/value asymmetry, residual windows. Pitfalls: quality loss at INT4 for long context. Senior signal: validates quality on the actual workload before quantizing.
- **Quantization formats (GPTQ, AWQ, FP8, NF4).** Why: fit large models on constrained VRAM. Foundations: activation-aware (AWQ), data-type optimality (NF4 for QLoRA). Senior signal: chooses FP8 on Hopper/Blackwell, AWQ-INT4 for tight VRAM, avoids GGUF for high-throughput serving.
- **Vector databases (pgvector, Qdrant, Weaviate, Pinecone, Milvus) and selection.** Why: retrieval backbone. Foundations: HNSW, filter pushdown, multi-tenancy, hybrid support. Senior signal: defaults to pgvector under ~10M vectors on existing Postgres, Qdrant for self-hosted speed/filtering, Weaviate for native hybrid, Milvus/Vespa at billion-scale.
- **Hybrid search (BM25 + dense, RRF).** Why: pure dense misses exact matches (SKUs, statute numbers). Foundations: sparse+dense fusion. Senior signal: treats hybrid as non-optional for production RAG.
- **Reranking (cross-encoders; Cohere, Voyage, BGE, Jina).** Why: bi-encoder embeddings are lossy. Foundations: two-stage retrieve-then-rerank, joint query-document scoring. Senior signal: passes top-50–100 candidates to a reranker, keeps top 3–10.
- **Chunking strategies and failure modes.** Why: most RAG failures originate in ingestion/chunking. Foundations: recursive/semantic/agentic splitting, overlap. Senior signal: tunes chunking against a labeled eval set, changing one variable at a time.
- **RAG evaluation (RAGAS faithfulness, context precision/recall; recall@k; answer grounding).** Why: RAG without metrics is guesswork. Senior signal: builds a 50–200 query golden set and tracks retrieval and grounding separately from generation.
- **Agentic RAG and query routing.** Why: complex/multi-hop queries break single-pass RAG. Foundations: iterative retrieval, query decomposition, complexity-based routing. Senior signal: routes simple queries to fast RAG and reserves agentic loops for hard ones, accepting latency/token cost.
- **Fine-tuning (LoRA, QLoRA, DoRA, full FT).** Why: shape behavior, not inject facts. Foundations: low-rank adapters, 4-bit base + adapter. Senior signal: follows Prompt → RAG → Fine-tune → Distill and uses fine-tuning for form, retrieval for facts.
- **Preference optimization (DPO, ORPO, KTO, GRPO) and libraries (TRL, Axolotl, Unsloth, torchtune).** Why: align tone/refusal/preference. Foundations: preference loss vs SFT, GRPO's critic-free simplicity. Senior signal: picks the right objective and the right library (Unsloth single-GPU speed, Axolotl config-driven multi-GPU, TRL reference RLHF, torchtune PyTorch-native).
- **Multi-adapter serving.** Why: economically viable per-tenant specialization. Foundations: one base model, many routed adapters (vLLM LoRA, LoRAX, SGLang). Senior signal: designs multi-tenant SaaS around it.
- **Synthetic data generation.** Why: data quality gates fine-tuning ROI. Senior signal: generates and filters synthetic preference/SFT data with eval feedback.
- **Agent orchestration patterns.** Why: distinguishing production-grade from hype. Foundations: per Anthropic's "Building Effective AI Agents," the most successful implementations use "simple, composable patterns" (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) rather than complex frameworks; orchestrator-worker multi-agent suits parallelizable breadth-first tasks. Pitfalls: multi-agent is not a fit for tightly-coupled tasks needing shared context. Senior signal: does the simplest thing that works and adds multi-agent only when it demonstrably improves performance.
- **Model Context Protocol (MCP).** Why: the de facto 2026 tool-calling standard ("USB-C for AI"), adopted by Anthropic, OpenAI, Google, and major frameworks/IDEs. Foundations: JSON-RPC client-server tool exposure, the 2026 stateless-core revision aligning auth with OAuth/OIDC. Senior signal: builds versioned, network-accessible MCP servers with proper authorization.
- **Agent evaluation (trajectory eval, tool-call accuracy, task completion, τ-bench pass^k).** Why: agents are non-deterministic and errors compound. Foundations: trajectory match vs LLM-as-judge over the message/tool-call sequence; end-state evaluation for state-mutating agents; τ-bench's pass^k reliability metric (SOTA function-callers succeed on <50% of tasks with pass^8 <25% in retail). Senior signal: evaluates the trajectory and the end state, and reports reliability across repeated runs, not single-shot pass rates.
- **LLM-as-judge and eval-driven development.** Why: scalable quality measurement. Foundations: rubric-based scoring, judge-model selection, CI gating. Pitfalls: prompt-eval mismatch where the eval prompt diverges from the production prompt. Senior signal: validates judges against human labels and gates deploys on regressions.
- **Observability (LangSmith, Langfuse, Braintrust, Arize Phoenix, DeepEval, promptfoo).** Why: close the loop between offline eval and production traces. Foundations: tracing spans, online scoring, golden-set regression. Senior signal: runs a CI eval framework (DeepEval/RAGAS/promptfoo) plus a tracing/annotation platform (Langfuse/LangSmith/Braintrust).
- **Context engineering.** Why: long-horizon agents degrade as context grows. Foundations: context rot (recall degrades as tokens grow — the attention budget), compaction (summarize-and-reinitialize), structured note-taking/memory, sub-agent context isolation (subagents return ~1–2k-token distilled summaries), just-in-time retrieval. Senior signal: manages the context window as a scarce, curated resource.
- **Failure modes.** Why: senior judgment is built on them. Foundations: error compounding over long horizons (one bad step derails the trajectory), prompt injection and excessive agency (OWASP LLM06), retrieval failure, hallucination, non-determinism. Senior signal: designs deterministic safeguards, checkpoints, least-privilege tools, and human-in-the-loop gates.
</exhaustive_topics>
<production_tech_stack>
Core: Python, PyTorch, HuggingFace Transformers/PEFT. Data manipulation/storage/query: document stores, pgvector/Qdrant/Weaviate/Milvus, embedding pipelines, Parquet. Experimentation/tracking: W&B, fine-tuning configs, eval golden sets. Training/compute/acceleration: TRL, Axolotl, Unsloth, torchtune, QLoRA on A100/H100, FlashAttention-3, FP8. Serving/deployment/orchestration: vLLM, SGLang, TensorRT-LLM, Ray Serve, NVIDIA Dynamo/llm-d; LangGraph + MCP for agents; multi-adapter serving (LoRAX). Monitoring/observability/quality: RAGAS, DeepEval, promptfoo for offline eval; LangSmith/Langfuse/Braintrust/Arize Phoenix for tracing and online eval; guardrails for injection/PII. Cloud/infra: GPU clusters, Kubernetes, object store. Each belongs because M6 is the LLM/agentic systems layer and demands current-generation tooling.
</production_tech_stack>
<strategic_capstone_project deploy="true">
**Project Title:** Production RAG + agent system with self-hosted serving and full evaluation.
**Why this belongs at this transition:** This is the third and final capstone — the move into LLM/agentic systems ownership. It forces the engineer to integrate serving, retrieval, agents, and eval into one accountable system.
**Problem Statement:** Build an enterprise knowledge agent that retrieves over private documents, calls tools via MCP, and answers reliably under latency, cost, and safety constraints.
**Users/stakeholders:** Internal users relying on correct answers; a compliance/security owner.
**Data sources and contracts:** A private document corpus with a documented ingestion and chunking contract; an evaluation golden set of 50–200 queries with known-correct chunks and reference answers.
**End-to-end architecture:** ingestion and chunking → embedding + hybrid (BM25 + dense) index → retrieval + cross-encoder reranking → LLM served on vLLM (FP8, prefix caching, continuous batching) → agent loop (MCP tools, orchestrator-worker pattern, context compaction) → offline evaluation (RAGAS) + online tracing (Langfuse) → drift and answer-quality monitoring with a feedback loop.
**Hard engineering constraints:** explicit TTFT/ITL latency budgets; a cost-per-1k-queries ceiling; a minimum faithfulness/grounding threshold; prompt-injection defenses; least-privilege tool scoping with human-in-the-loop on consequential actions.
**Failure modes to expect:** retrieval failure is the dominant error mode — Barnett et al. (arXiv 2401.05856) established the canonical 7-point RAG failure taxonomy across three real-world domains, and empirical failure analysis (arXiv 2603.02473) found "Retrieval failure is the dominant error mode, accounting for 11–46% of all questions," while "hallucinations [stay] at 0.4–1.4% regardless of configuration" — so the engineer must instrument retrieval first. Also expect context rot over long sessions, error compounding across agent steps, and excessive agency (OWASP LLM06).
**Exact implementation stack:** vLLM, Qdrant or pgvector, Cohere/Voyage/BGE reranker, LangGraph + MCP, RAGAS + DeepEval + Langfuse, guardrails/PII redaction.
**Evidence of a strong engineering standard:** retrieval recall@k and RAGAS faithfulness tracked separately from generation quality; trajectory and tool-call-accuracy evals on the agent; documented red-team results for injected prompt-injection and retrieval-failure cases; a latency/cost dashboard meeting the stated budgets; a demonstrated regression gate in CI.
</strategic_capstone_project>
</milestone>

<milestone number="7" title="Senior System Design, Platform Thinking & Leadership Judgment">
<domain_scope>
By the end of M7 the learner can architect ML/LLM platforms end-to-end, make defensible cost/reliability/security/governance tradeoffs, and exercise the technical-leadership judgment that only accrues after systems fail in production. This is the capstone capability layer: the engineer is no longer building a system but designing the platform and the decision frameworks others build on — choosing what to build vs buy, modeling cost and reliability against SLOs, designing governance and security into systems rather than retrofitting them, and mentoring teams through tradeoffs. The defining senior skill here is judgment under ambiguity informed by hard-won failure experience: cost explosions, observability blind spots, governance retrofits, and benchmark gaming.
</domain_scope>
<competency_targets>
- By the end of this milestone, the engineer should be able to design a multi-tenant ML/LLM platform with clear build-vs-buy decisions, component boundaries, and swappable interfaces.
- By the end of this milestone, the engineer should be able to model and control cost — GPU utilization, token economics, multi-agent token blowup, disaggregation economics — against explicit SLOs.
- By the end of this milestone, the engineer should be able to design governance and compliance (EU AI Act, NIST AI RMF, OWASP LLM Top 10, ISO 42001) into the platform from the start.
- By the end of this milestone, the engineer should be able to design security defenses for agents: prompt injection, excessive agency, least privilege, downstream authorization, and human-in-the-loop gates.
- By the end of this milestone, the engineer should be able to lead incident response, design rollback strategies at platform scale, and run blameless postmortems.
- By the end of this milestone, the engineer should be able to make eval-driven, data-driven architecture decisions, read benchmarks critically, and mentor others on tradeoffs.
</competency_targets>
<exhaustive_topics>
- **Platform architecture and the platform/MLOps boundary.** Why: scale demands shared substrate. Foundations: registry-as-service, golden-path pipelines, self-serve infrastructure. Pitfalls: building a custom platform from scratch instead of composing off-the-shelf components. Senior signal: defines the integration layer as the team's real product.
- **Build-vs-buy and component swappability.** Why: lock-in is expensive and migrations cost engineering quarters. Foundations: interface contracts, vendor risk. Senior signal: chooses infrastructure that can be migrated away from.
- **Cost engineering.** Why: cost explosions are a defining production failure. Foundations: token economics, disaggregation economics, spot/on-demand, model routing. Datapoint: multi-agent systems carry roughly a 15× token-cost multiplier per Anthropic's June 2025 engineering post "How we built our multi-agent research system," which states that "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats" — and yet the same multi-agent system beat single-agent Claude Opus 4 by 90.2% on their internal research eval, which is exactly why the senior tradeoff is "use multi-agent only when task value justifies the token cost." Senior signal: models cost per outcome and ties architecture to it.
- **Reliability engineering.** Why: platforms carry many services. Foundations: SLOs, error budgets, graceful degradation, fallback models. Senior signal: designs for partial failure.
- **Rollout/rollback strategy at platform scale.** Why: stateful, in-flight agents complicate rollback. Foundations: rainbow/staged deployments, checkpointing. Senior signal: never disrupts in-flight stateful work on deploy.
- **Security (OWASP LLM01 prompt injection, LLM06 excessive agency).** Why: agents act in the world. Foundations: authorization in downstream systems rather than trusting the LLM; least-privilege tools; scoped tokens; AIBOM. Senior signal: enforces complete mediation and human sign-off on high-impact actions.
- **Governance and regulation.** Why: legally binding in 2026. Foundations: the EU AI Act (Regulation (EU) 2024/1689) Article 99 penalty tiers — up to €35M or 7% of total worldwide annual turnover (whichever is higher) for prohibited practices under Article 5; up to €15M or 3% for high-risk/operator obligations; and up to €7.5M or 1% for supplying incorrect information, with the prohibited-practice provisions enforceable since 2 February 2025 — plus NIST AI RMF (Govern/Map/Measure/Manage), ISO 42001, and Article 12-style automatic logging. Senior signal: builds audit trails and human-oversight controls into the architecture rather than retrofitting them.
- **Data governance and lineage.** Why: trust and auditability. Foundations: catalogs, lineage, access control. Senior signal: governance attaches to data, not tools.
- **Model/prompt versioning and reproducibility at scale.** Why: many models, many prompts. Senior signal: any production output traces to model + prompt + data versions.
- **Benchmark literacy.** Why: benchmarks mislead. Foundations: scaffold sensitivity (the same model swings tens of points across harnesses), exploitable benchmarks, pass^k vs pass@1. Senior signal: checks methodology before trusting a number and treats outcome scores as measuring the system, not the model.
- **Organizational design and on-call.** Why: platforms are sociotechnical. Senior signal: designs ownership and on-call that scale.
- **Technical writing and decision records.** Why: judgment must be communicated. Senior signal: writes ADRs that capture tradeoffs and reversibility.
- **Ethics, bias, and fairness governance.** Why: harm and liability. Senior signal: bakes fairness checks into the lifecycle.
</exhaustive_topics>
<production_tech_stack>
Core: the entire prior stack composed into a coherent platform. Data manipulation/storage/query: governed lakehouse + catalog (Unity/Polaris), lineage tooling (DataHub/OpenMetadata). Experimentation/tracking: org-wide registry (MLflow/W&B), unified eval store. Training/compute/acceleration: shared GPU scheduling, model routing, spot management. Serving/deployment/orchestration: multi-tenant serving (vLLM/KServe), staged/rainbow rollouts. Monitoring/observability/quality: unified traces + evals + drift on OpenTelemetry GenAI conventions. Governance/security primitives: NeMo Guardrails, policy engines (OPA/Cedar), AIBOM tooling, PII redaction, audit logging. Cloud/infra: Kubernetes, Terraform, IAM, cost-management tooling. Each belongs because M7 is platform-level judgment that integrates and governs everything below it.
</production_tech_stack>
<strategic_capstone_project deploy="false"></strategic_capstone_project>
</milestone>
