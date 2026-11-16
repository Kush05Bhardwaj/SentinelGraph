# SentinelGraph Product Requirements Document (PRD)

**Document ID:** SG-PRD-001  
**Version:** 1.0 (draft)  
**Status:** Draft  
**Applies to:** v1.0 (offline-first release)  
**Owner:** Product / Engineering

---

## 1. Problem statement and motivation

Network intrusion detection today relies heavily on signature-based and rule-based
systems that struggle with novel, zero-day, and low-and-slow attacks. Network
traffic is inherently relational: hosts communicate through flows, forming a
graph of entities and edges. Most existing tooling treats each flow record as an
independent row, discarding the structural relationships that surface coordinated
campaigns, lateral movement, and distributed threats.

SentinelGraph addresses this gap by modeling network traffic as a graph and
applying graph neural networks (GNNs) alongside classical ML models for binary
and multiclass threat detection, edge-level threat classification, anomaly
detection, and explainable AI. The product targets security operations teams that
need measurable, interpretable threat signals without depending on cloud-only
inference or proprietary training pipelines.

## 2. Target users and intended use cases

### 2.1 Target users

| User | Role | Primary need |
|------|------|--------------|
| SOC analyst | Triage and investigate network alerts | Prioritize incidents, inspect explanations, correlate related events |
| Security engineer | Deploy and maintain detection models | Train, evaluate, and version models offline; integrate predictions via API |
| ML engineer | Experiment with graph-based detection | Build and compare GNN and classical models on supported datasets |
| Auditor / compliance | Review detection decisions | Trace predictions to inputs, model version, and explanation artifacts |

### 2.2 Intended use cases

- **UC-1:** Ingest network flow records from supported dataset sources and construct per-flow and per-graph representations.
- **UC-2:** Classify each flow as benign or malicious (binary intrusion detection).
- **UC-3:** Classify each flow into an attack family (multiclass intrusion detection).
- **UC-4:** Classify each graph edge as threatening or benign (edge-level graph threat classification).
- **UC-5:** Detect anomalous flows that deviate from learned behavior (anomaly detection).
- **UC-6:** Produce explainable AI outputs (feature importance / subgraph explanations) for individual predictions.
- **UC-7:** Correlate related alerts into incidents and assign a measurable risk score.
- **UC-8:** Serve predictions through a versioned REST API and view them in a dashboard.
- **UC-9:** Train and evaluate models offline using Kaggle as the primary model-training workflow.

## 3. Project objectives

- **OBJ-1:** Deliver an offline-first v1.0 that trains and evaluates detection models without external network access after dependency installation.
- **OBJ-2:** Support binary and multiclass network intrusion detection on the supported dataset sources.
- **OBJ-3:** Support edge-level graph threat classification.
- **OBJ-4:** Provide explainable AI outputs linked to actual predictions.
- **OBJ-5:** Expose predictions and incidents through a FastAPI backend and a React dashboard.
- **OBJ-6:** Keep all requirements testable and measurable; do not ship unpublished performance targets.
- **OBJ-7:** Never commit datasets, model checkpoints, weights, or credentials to the repository.

## 4. Functional requirements

### 4.1 Data ingestion (implemented in later issues)

**FR-1.1:** The system SHALL ingest network flow records from CSV/Parquet exports of CIC-IDS2017, CSE-CIC-IDS2018, and UNSW-NB15.
**FR-1.2:** The system SHALL validate schema conformity for each supported dataset source and reject records missing required fields (timestamp, source IP, destination IP, source port, destination port, protocol, byte/packet counts, label).
**FR-1.3:** The system SHALL record the source dataset, file path, and ingest timestamp for every ingested batch (measurable: metadata persisted per batch).

### 4.2 Preprocessing and feature engineering (implemented in later issues)

**FR-2.1:** The system SHALL normalize numeric features per dataset source (measurable: zero mean / unit variance or min-max, configurable).
**FR-2.2:** The system SHALL encode categorical features (protocol, flags, service) deterministically.
**FR-2.3:** The system SHALL handle missing values with a documented strategy (drop / impute) configurable per pipeline.
**FR-2.4:** The system SHALL produce a reproducible feature matrix with a fixed random seed for train/validation/test splits.

### 4.3 Graph construction (implemented in later issues)

**FR-3.1:** The system SHALL construct a graph where nodes represent network entities (IP addresses) and edges represent communication flows.
**FR-3.2:** The system SHALL attach node features (aggregated flow statistics) and edge features (per-flow features) to the graph.
**FR-3.3:** The system SHALL support directed edges preserving flow direction (src -> dst).

### 4.4 Model training (planned; Kaggle-first strategy, implemented in later issues)

**FR-4.1:** The system SHALL train classical ML models (e.g., Random Forest, Logistic Regression, XGBoost) on tabular flow features.
**FR-4.2:** The system SHALL train graph neural network models (e.g., GCN, GAT, GraphSAGE) on flow graphs.
**FR-4.3:** The system SHALL support Kaggle as the primary model-training workflow: training scripts run offline, artifacts stored under `artifacts/`, never committed.
**FR-4.4:** The system SHALL version every trained model artifact with a training run ID, dataset source, hyperparameters, and timestamp.

### 4.5 Binary intrusion detection (planned)

**FR-5.1:** The system SHALL output a binary label (benign / malicious) per flow.
**FR-5.2:** The system SHALL output a calibrated probability score in [0, 1] for the malicious class.
**FR-5.3:** The system SHALL expose a configurable decision threshold (default 0.5).

### 4.6 Multiclass intrusion detection (planned)

**FR-6.1:** The system SHALL output an attack-family label per flow from the dataset's label vocabulary (e.g., Benign, DoS, PortScan, Bot, Brute Force, Web Attack, Infiltration, Heartbleed for CIC-IDS2017; equivalent vocabularies for CSE-CIC-IDS2018 and UNSW-NB15).
**FR-6.2:** The system SHALL output a probability distribution over classes.
**FR-6.3:** The system SHALL reject flows whose predicted class is not in the supported label vocabulary for the active dataset source.

### 4.7 Edge-level graph threat classification (planned)

**FR-7.1:** The system SHALL output a binary threat label per graph edge.
**FR-7.2:** The system SHALL output an edge threat score in [0, 1].
**FR-7.3:** The system SHALL link every edge classification to the underlying flow record and source IP / destination IP pair.

### 4.8 Anomaly detection (planned)

**FR-8.1:** The system SHALL flag flows whose reconstruction error or density estimate exceeds a dataset-specific threshold.
**FR-8.2:** The system SHALL distinguish anomaly-detection flags from supervised intrusion-detection labels in the output schema.
**FR-8.3:** The system SHALL support unsupervised operation (no labels required) for anomaly scoring.

### 4.9 Explainable AI (planned)

**FR-9.1:** For every prediction served via API, the system SHALL produce an explanation artifact (feature importance for classical models; SHAP values or GNNExplainer subgraph for GNNs).
**FR-9.2:** The explanation SHALL be linkable to a specific prediction ID, model version, and input record.
**FR-9.3:** The system SHALL expose explanation artifacts through the API and dashboard.

### 4.10 Threat correlation and risk scoring (planned)

**FR-10.1:** The system SHALL group related predictions into incidents based on shared IPs, time window, and attack family.
**FR-10.2:** The system SHALL compute a risk score per incident in [0, 100] using a documented, deterministic formula (weighted combination of prediction confidence, frequency, and severity).
**FR-10.3:** The system SHALL persist incidents with start time, end time, involved entities, and risk score.

### 4.11 API capabilities (implemented in later issues)

**FR-11.1:** The system SHALL expose a REST API (FastAPI) with at minimum:
- `POST /predict` — input flow record, output prediction + explanation.
- `GET /health` — service health (implemented in SG-001).
- `GET /incidents` — list persisted incidents.
- `GET /incidents/{id}` — incident detail.
**FR-11.2:** The API SHALL return structured JSON responses with HTTP status codes and error messages for invalid input.
**FR-11.3:** The API SHALL be callable without network access to external services (offline-first).

### 4.12 Dashboard capabilities (implemented in later issues)

**FR-12.1:** The dashboard SHALL display attack category distribution from backend-sourced metrics.
**FR-12.2:** The dashboard SHALL display model prediction metrics (counts per class, threshold).
**FR-12.3:** The dashboard SHALL display anomaly score distribution.
**FR-12.4:** The dashboard SHALL display incident severity distribution.
**FR-12.5:** The dashboard SHALL display SHAP explanations and GNN explanation subgraphs linked to predictions.
**FR-12.6:** The dashboard SHALL provide an incident detail view with time-based filtering where timestamps are available.
**FR-12.7:** The dashboard SHALL handle empty and error states gracefully.

### 4.13 Persistence (implemented in later issues)

**FR-13.1:** The system SHALL persist network event metadata, prediction records, threat alerts, incidents, model metadata, and explanation references in PostgreSQL.
**FR-13.2:** The system SHALL maintain relationships between events, models, and incidents.
**FR-13.3:** Predictions and incidents SHALL persist across service restarts.

## 5. Non-functional requirements

**NFR-1 (Offline-first):** After `pip install` / `npm install`, all training, inference, and dashboard operation SHALL work without access to external network services. No internet-dependent inference endpoints.
**NFR-2 (Determinism):** Given the same dataset, split seed, hyperparameters, and code version, training SHALL produce identical model artifacts and metrics (measurable: hash-compare artifacts across repeat runs).
**NFR-3 (API latency):** Single-flow inference via `/predict` SHALL complete within a measurable SLO to be defined during SG-006–SG-010 (no target invented here).
**NFR-4 (Throughput):** Batch inference SHALL process at least one flow per measurable unit to be defined during implementation (no target invented here).
**NFR-5 (Observability):** The API SHALL emit structured logs for every request (request ID, path, status, latency).
**NFR-6 (Security):** No secrets, credentials, or keys SHALL be committed to the repository. Runtime configuration SHALL use environment variables or `.env` (gitignored).
**NFR-7 (Testability):** Every functional requirement SHALL be verifiable by at least one automated test (unit, integration, or regression).
**NFR-8 (Types):** All public Python functions SHALL include type hints and pass `mypy --strict`-equivalent checks configured in `pyproject.toml`.
**NFR-9 (Frontend build):** `npm run build` in `apps/dashboard` SHALL succeed with `tsc --noEmit` clean.
**NFR-10 (Data exclusion):** Datasets, checkpoints, and model weights SHALL never be committed (enforced by `.gitignore`).

## 6. Supported dataset sources

| Dataset | Source | Format | Labels | Notes |
|---------|--------|--------|--------|-------|
| CIC-IDS2017 | Canadian Institute for Cybersecurity | CSV | Benign + attack families | DoS, PortScan, Bot, Brute Force, Web Attack, Infiltration, Heartbleed |
| CSE-CIC-IDS2018 | Canadian Institute for Cybersecurity | CSV | Benign + attack families | Bot, Brute Force, DoS, Heartbleed, Infiltration, PortScan, Web Attack |
| UNSW-NB15 | UNSW Canberra / Australian Centre for Cyber Security | CSV | Benign + attack categories | Analysis, Backdoor, Fuzzers, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |

**FR-6.x (dataset support):** The system SHALL declare, per dataset source, the expected feature schema, label vocabulary, and supported task set (binary / multiclass / anomaly / edge classification). A dataset not on this list SHALL be explicitly unsupported until a schema declaration is added.

Dataset files themselves are NOT part of the repository and are gitignored under `data/raw/`.

## 7. Binary and multiclass network intrusion detection

**Binary (planned):**
- Input: one network flow record.
- Output: label ∈ {Benign, Malicious} + probability ∈ [0, 1] + threshold used.
- Measurable via: confusion matrix (TP, FP, TN, FN), precision, recall, F1, ROC-AUC computed on the held-out test split.

**Multiclass (planned):**
- Input: one network flow record.
- Output: label ∈ dataset label vocabulary + probability distribution over classes.
- Measurable via: per-class precision/recall/F1, macro-averaged F1, confusion matrix on the held-out test split.

No numeric performance targets are asserted in this PRD; targets are to be agreed upon during SG-008 and recorded separately.

## 8. Edge-level graph threat classification

**FR-8.x (planned):** For every communication edge in the constructed graph, output:
- threat label ∈ {Benign, Threatening}
- edge threat score ∈ [0, 1]
- reference to source flow record and endpoint IPs

Measurable via: edge-level precision/recall/F1 and ROC-AUC on labeled graph edges in the test split.

## 9. Anomaly detection

**FR-9.x (planned):** 
- Unsupervised anomaly score per flow (reconstruction error, isolation score, or density estimate).
- Configurable threshold; flows above threshold flagged as anomalous.
- Output schema distinct from supervised labels so anomaly flags can be audited separately.
- Measurable via: precision@k, false positive rate at a fixed threshold, and AUC-PR on labeled anomalies where ground truth exists.

## 10. Explainable AI

**FR-10.x (planned):**
- Classical models: feature importance (e.g., permutation importance or SHAP) per prediction.
- GNN models: SHAP values or GNNExplainer subgraph per prediction.
- Every explanation linked to prediction ID, model version, input record.
- Measurable via: explanation completeness (explanation produced for N/N predictions), and sanity checks (explanation non-empty, feature attribution sums approximately to output difference where applicable).

## 11. Threat correlation and risk scoring

**FR-11.x (planned):**
- Incident grouping: predictions sharing src/dst IP and attack family within a configurable time window (default 1 hour) are grouped.
- Risk score per incident ∈ [0, 100], deterministic formula documented in `docs/`.
- Incident record: ID, start time, end time, involved entities, attack families, risk score, linked prediction IDs.
- Measurable via: incident count, risk score distribution, and reproducibility (same input → same incidents and scores).

## 12. API and dashboard capabilities

**API (planned beyond `/health`):** `/predict`, `/incidents`, `/incidents/{id}`, structured JSON, HTTP error handling, offline operation. See FR-11.
**Dashboard (planned):** views and filters per FR-12; empty/error states; linked explanations.

## 13. Kaggle-first model training strategy

**FR-13.x (planned):**
- All model training executed via local scripts runnable on Kaggle Kernels or locally with identical code.
- Training artifacts (model weights, metrics, explanations) written to `artifacts/`, never committed.
- A single training script per model family (classical, GNN) accepting dataset source, split seed, and hyperparameters as arguments.
- Model metadata (run ID, dataset, hyperparameters, metrics, timestamp) persisted for every run.

## 14. Offline-first v1.0 scope

v1.0 is offline-first: after dependency installation, ingestion, preprocessing, training, inference, API, and dashboard run without external network services. Cloud training, managed inference endpoints, and SaaS integrations are out of scope for v1.0.

## 15. Explicitly excluded v1.0 capabilities

The following are NOT part of v1.0 and will not be tested or supported in v1.0:

- Real-time packet capture / live network interface ingestion (pcap/NIC).
- Streaming inference (Kafka, Kinesis, etc.).
- Multi-tenant authentication and authorization (SSO, OAuth, RBAC).
- Model deployment to Kubernetes / managed model serving.
- Scheduled retraining pipelines (cron/airflow).
- Email / SMS / Slack alerting.
- Packet-level (L2/L3) analysis beyond NetFlow-style flow records.
- Cloud object storage (S3, GCS, Azure Blob) as primary artifact store.
- Mobile or desktop client applications.
- Any capability requiring internet access at inference time.

## 16. Performance and evaluation criteria

Evaluation is performed on a fixed held-out test split per dataset source. The following metrics SHALL be computed and recorded for every model run:

| Task | Metric(s) |
|------|-----------|
| Binary intrusion detection | Precision, Recall, F1, ROC-AUC, confusion matrix |
| Multiclass intrusion detection | Per-class precision/recall/F1, macro F1, confusion matrix |
| Edge-level threat classification | Edge precision/recall/F1, ROC-AUC |
| Anomaly detection | AUC-PR, false positive rate at threshold, precision@k |
| Explainability | Explanation coverage (predictions with explanation / total predictions) |

**No numeric performance targets are asserted in this PRD.** Targets (e.g., "macro F1 ≥ 0.90") are not agreed upon at this stage and will be defined during SG-008 / SG-009 and recorded in a separate evaluation baseline document. Claims about accuracy must be backed by recorded test-split metrics; unpublished targets are not acceptance criteria.

## 17. Security, privacy and ethical limitations

**SEC-1:** Network flow data may contain IP addresses, ports, and traffic volumes that can be sensitive. Datasets are treated as potentially sensitive and never committed.
**SEC-2:** The API SHALL not log full request payloads containing potentially sensitive flow data by default; log metadata only (request ID, path, status, latency).
**SEC-3:** No authentication in v1.0; the API SHALL be deployed behind infrastructure-level access control (not part of v1.0 scope).
**PRV-1:** IP addresses are pseudo-identifiers; the system SHALL support IP masking/anonymization as a configurable preprocessing option.
**ETH-1:** Model predictions are decision support, not authoritative verdicts; a human analyst SHALL review high-severity incidents before action.
**ETH-2:** Model performance may vary across dataset sources and network populations; metrics SHALL be reported per dataset source, not aggregated into a single headline number.
**ETH-3:** Explainability outputs are approximations and SHALL not be treated as proof of malicious intent.

## 18. Success criteria and release definition

### 18.1 v1.0 release definition

v1.0 is released when all of the following hold:

- Source dataset ingestion implemented and tested for CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15 (schema validation + metadata).
- Preprocessing + feature engineering pipeline deterministic and tested.
- Graph construction implemented and tested.
- At least one classical model and one GNN model trainable offline with recorded metrics.
- Binary and multiclass prediction implemented with `/predict` endpoint.
- Edge-level threat classification implemented.
- Anomaly detection implemented.
- Explainability artifacts produced and exposed via API/dashboard.
- Incident grouping + deterministic risk scoring implemented.
- PostgreSQL persistence with migrations; predictions/incidents survive restart.
- FastAPI backend + React dashboard operational end-to-end (Docker Compose per SG-047).
- Automated test suite passes (unit/integration/regression per SG-046).
- No datasets, checkpoints, weights, or credentials in git history.
- PRD (this document) reviewed and agreed; evaluation baseline targets recorded separately.

### 18.2 Success criteria (measurable)

- Every functional requirement in Section 4 has ≥1 automated test.
- `pytest` suite passes with exit code 0.
- `ruff check`, `ruff format --check`, `mypy` pass on configured paths.
- `npm run build` succeeds and `tsc --noEmit` is clean.
- `docker compose up` starts backend, frontend, and PostgreSQL; `/health` returns 200.
- Repeat training run on identical inputs produces byte-identical model artifacts (determinism check).
- No committed files under `data/raw/`, `data/processed/`, `data/graphs/`, `artifacts/`, matching `.gitignore` patterns `*.pkl`, `*.pt`, `*.pth`, `*.csv`, `*.parquet`.

### 18.3 Distinction: implemented vs planned

| Area | SG-001 (implemented) | Planned (later issues) |
|------|----------------------|------------------------|
| Repository structure | ✅ Complete | — |
| Backend package + `/health` | ✅ Implemented | Remaining endpoints |
| Frontend scaffold | ✅ Implemented | Dashboard views |
| ML packages (placeholders) | ✅ Skeleton modules | Preprocessing, features, GNN, training |
| Tests | ✅ conftest fixtures only | Unit/integration/regression (SG-046) |
| Docker / CI | — | SG-047 |
| PRD | — | This document (SG-002) |

## Appendix A: Requirement traceability

| ID | Requirement | Implemented in | Verified by |
|----|-------------|----------------|-------------|
| FR-1.x | Data ingestion | SG-002+ | Dataset schema tests |
| FR-2.x | Preprocessing | SG-003 | Pipeline unit tests |
| FR-3.x | Graph construction | SG-004/005 | Graph construction tests |
| FR-4.x | Model training | SG-006/007 | Training run tests |
| FR-5.x | Binary detection | SG-006–010 | Prediction tests |
| FR-6.x | Multiclass detection | SG-006–010 | Prediction tests |
| FR-7.x | Edge classification | SG-004–010 | Edge classification tests |
| FR-8.x | Anomaly detection | SG-009 | Anomaly tests |
| FR-9.x | Explainability | SG-010 | Explanation tests |
| FR-10.x | Correlation / risk | SG-011+ | Incident tests |
| FR-11.x | API | SG-011 | API integration tests |
| FR-12.x | Dashboard | SG-043–045 | Frontend tests |
| FR-13.x | Persistence | SG-042 | Database tests |
| NFR-1..10 | Non-functional | SG-001–048 | Lint/type/test/CI |

---

*End of PRD. Direct changes to this document through the product owner; implementation questions map to the linked SG issues.*
