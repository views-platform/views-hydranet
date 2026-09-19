<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>


# **HydraNet**: Spatiotemporal Conflict Forecasting Model 👾  

> **Part of the [VIEWS Platform](https://github.com/views-platform) ecosystem for large-scale conflict forecasting.**

---

## 📚 Table of Contents  

1. [Overview](#overview)  
2. [Role in the VIEWS Pipeline](#role-in-the-views-pipeline)  
3. [Features](#features)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Configuration & Stability](#configuration--stability)
7. [Runtime Expectations](#runtime-expectations)
8. [Architecture](#architecture)  
9. [Project Structure](#project-structure)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Acknowledgements](#acknowledgements)  

---

## 🧠 Overview  

**HydraNet** is an advanced machine learning model designed for **spatiotemporal forecasting of violent conflict** at high granularity. It predicts three types of violence—state-based, non-state-based, and one-sided—by solving regression and classification tasks concurrently.  

The model provides:  
- **Probabilistic Outputs**: Enables uncertainty quantification through posterior distributions.  
- **Temporospatial Learning**: Leverages convolutional layers for spatial dependencies and LSTMs for temporal patterns.  
- **Multi-Tasking**: Simultaneously predicts probabilities and magnitudes of conflict.  

HydraNet is **FAIR-compliant** (Findable, Accessible, Interoperable, Reusable), ensuring transparency and ease of use for researchers and policymakers.  

---

## 🌍 Role in the VIEWS Pipeline  

HydraNet is a core component of the **Violence & Impacts Early Warning System (VIEWS)** pipeline, working alongside other repositories:  

- **[views-pipeline-core](https://github.com/views-platform/views-pipeline-core):** Manages data ingestion, preprocessing, and pipeline orchestration.  
- **[views-models](https://github.com/views-platform/views-models):** Provides interfaces to train, test, and deploy HydraNet.  
- **[views-evaluation](https://github.com/views-platform/views-evaluation):** Evaluates model predictions and performs calibration tasks.  
- **[docs](https://github.com/views-platform/docs):** Organization/pipeline level documentation.

### Integration Workflow  

HydraNet fits into the VIEWS pipeline as follows:  
1. **Data Input:** Preprocessed PRIO grid-cell-level conflict data is retrieved from **views-pipeline-core**.  
2. **Model Execution:** HydraNet generates probabilistic forecasts for multiple violence types across a 36-month horizon.  
3. **Evaluation and Calibration:** Outputs are passed to **views-evaluation** for ensembling and alignment with other models.  

---

## ✨ Features  

- **Multi-Task Learning:** Simultaneous prediction of probabilities and magnitudes for three conflict types.  
- **Uncertainty Quantification:** Generates posterior distributions for robust decision-making.  
- **Hybrid Architecture:** Combines CNNs for spatial dependencies, LSTMs for temporal patterns, and U-net for precision.  
- **Minimal Manual Engineering:** Relies solely on past conflict history, simplifying input requirements.  
- **Scalable Design:** Adaptable for new features and forecasting tasks.  

---

## ⚙️ Installation  

### Prerequisites  

- Python >= 3.8  
- GPU support recommended (e.g., NVIDIA CUDA).   

### Steps  

See the organization/pipeline level [docs](https://github.com/views-platform/docs)  

---

## 🚀 Usage  

### 1. Run Training Locally  

See the organization/pipeline level [docs](https://github.com/views-platform/docs)  

### 2. Use in the VIEWS Pipeline  

HydraNet seamlessly integrates into the broader VIEWS pipeline. After training and prediction, the outputs can be passed into the **views-evaluation** repository for further analysis and calibration.  

---

## ⚙️ Configuration & Stability

HydraNet uses a **Strict Handshake** protocol to ensure production predictability. All hyperparameters are validated against a Pydantic schema (`HydraNetConfig`) at the start of every task.

- **Fail-Fast:** If a required field is missing or a value is invalid (e.g., a typo in `transform`), the program will halt immediately with a detailed error report.
- **Safe-Mode:** The system uses internal state protection to ensure that critical paths (like model saving and data loading) are robust against partial initializations or mocked environments.

### Feature Lifecycle (ADR 046)
HydraNet employs a **Symmetric Feature Lifecycle** governed by an "Instructional Blueprint."
- **Transformations:** Mathematical scaling (e.g., `log1p`) applied to raw inputs.
- **Derivations:** Manufacturing instructions for targets (e.g., `binary` thresholding) applied consistently during training and evaluation.
- **Handshake:** The model manager "sanctifies" ground-truth data for evaluation, ensuring that targets like `by_sb_best` are derived on-the-fly and bit-perfect with training logic.

### Prediction Output Format (ADR 047)
HydraNet adopts the **PredictionFrame** interface mandated by `views-pipeline-core` (ADR-033).

- **What:** `_evaluate_model_artifact()` returns `dict[str, list[PredictionFrame]]` and `_forecast_model_artifact()` returns `dict[str, PredictionFrame]` — one entry per target signal.
- **Why:** Target-keyed output enables multi-target dispatch, enforces a validated `(N, S)` shape contract, and unlocks automatic parity auditing between the PF and legacy DataFrame paths.
- **How:** The config declares `"prediction_format": "prediction_frame"`. The manager converts its internal DataFrames via `_to_pf_dict()` before returning. The upstream pipeline reads the flag and routes through `PredictionFrameDispatcher`.
- **Guide:** See [`reports/guides/prediction_frame.md`](reports/guides/prediction_frame.md) for a self-contained implementation guide, including how to adopt this pattern in other model repos.

---

## ⏱ Runtime Expectations

How long a roster model takes, **measured**, per machine. Add a row when you run on a new machine;
never estimate one — and add a new **workload** row if any knob below changes, because the
machine rows only mean something against a fixed workload.

**Workload A — the roster standard (every row below).** These are the knobs that set the compute;
anything else in the config is irrelevant to time.

| knob | value | what it scales |
|---|---|---|
| `model` | `HydraBNUNet06_LSTM4`, `total_hidden_channels` 32, `input_channels` 3 (`sb`, `ns`, `os`), `output_channels` 1 | cost per step |
| training volume | `total_lessons` 300 × `windows_per_lesson` 3 × 395 months = 355,500 forward/backward steps, batch 1 | **training, linearly** |
| `window_dim` | 32 × 32 spatial crop per training window | cost per step |
| `time_steps` | 36-month rollout horizon | length of every inference rollout |
| region / grid | `africa_me_legacy`, 13,110 cells on the 180 × 180 model grid | inference cost per origin |
| posterior | `n_posterior_samples` (D) 4 × `n_head_samples` (K) 4 = 16 draws per cell | **evaluation, linearly in D** (K is cheap) |
| origins | 13 validation origins | **evaluation, linearly** |
| targets | 3 regression + 3 classification (`*_sb`, `*_ns`, `*_os`) | both, mildly |
| `diagnostic_visualizations` | `True` — about 12 figures per lesson | ≈ +1 h on training |
| BatchNorm recalibration | `bn_recalibrate: True`, 30 forward-only windows after training | minutes |

A global run (`region="land"`, 360 × 720) or a bigger `n_posterior_samples` is a **different
workload** — start a new table, do not overwrite a row.

| machine | GPU · driver | torch | train (workload A) | lessons / h | evaluate (workload A) | measured |
|---|---|---|---|---|---|---|
| laptop — i9-13900H, 31 GB, Linux Mint 21.1 | RTX 4070 Laptop 8 GB · 535 | 2.6.0+cu124 | **3 h 00 m – 4 h 55 m** | **76–99** | **17–24 min** | 2026-09-16, four models from the PyPI wheel: heavy_freighter 3 h 01 m / 99 per h (overnight, laptop idle), bold_comet 3 h 23 m / 89, purple_alien 4 h 56 m / 76 and violet_visitor 4 h 53 m / 77 (laptop in use). Same config took 3 h 15 m on 2026-09-08. |
| same laptop, **torch on CPU** | *(CUDA unavailable — torch 2.14+cu130 vs driver 535)* | 2.14.0+cu130 | 6 h 46 m | 44 | 65 min | 2026-09-16 (`violet_visitor`) — see #377 |
| server | *(to be measured)* | | | | | |

**Read the CPU row as a warning, not a data point.** A fresh `pip install` resolves the newest torch,
and torch's CUDA build moves faster than drivers get updated; if the driver cannot run it, torch
falls back to CPU and the only trace is a `UserWarning` at import and a DEBUG log line. Everything
else looks healthy and the run is merely 2× slower — on a server with an older driver it would be
far worse. Check `torch.cuda.is_available()` in the env before a long run, or pin torch to a build
your driver supports (`--index-url https://download.pytorch.org/whl/cu124`). Tracked in #377.

**What the time is made of** (laptop, GPU): pure training steps run at ~55 months/s over 355,500
months (≈1.8 h); the rest is per-lesson diagnostics, forensics and W&B logging (≈1–1.5 h).
`diagnostic_visualizations: False` buys back roughly an hour per run. The spread across the four
measured runs is the laptop being used at the same time, not the models: the fastest ran overnight.

**Training is deterministic here.** All four models retrained on 2026-09-16 from the PyPI wheel
produced artifacts **byte-identical** to the ones trained from the checkout on 2026-09-07/08 — every
tensor, BatchNorm buffers included (`reports/2026-09-15_pypi_smoke/`). Same seed, same data, same
torch build, same GPU ⇒ same weights. A different torch build or GPU is not expected to reproduce
them bit for bit.

---

## 🏗 Architecture  

HydraNet employs a **probabilistic recurrent U-net** architecture optimized for spatiotemporal conflict forecasting.  

### Key Components  

- **CNNs (Convolutional Neural Networks):** Capture intricate spatial patterns in grid-cell data.  
- **LSTMs (Long Short-Term Memory networks):** Model temporal dependencies and trends.  
- **Dropout Layers:** Enable Monte Carlo sampling to quantify model uncertainty.  
- **Multi-Decoder Design:** Outputs six distinct forecasts (probabilities and magnitudes for three violence types).  

### Workflow  

1. **Input:** Historical conflict fatalities categorized as state-based, non-state, and one-sided.  
2. **Data Processing:** Converts historical data into z-stacks of monthly grids with three channels (one for each violence type).  
3. **Prediction:** Generates six outputs: probabilities and magnitudes for each type of violence.  

For a detailed explanation of the architecture, refer to the **[HydraNet Paper](link-to-paper)**.  

---

## 🗂 Project Structure  

```plaintext
views_hydranet/
├── README.md            # Documentation
├── tests                # Unit and integration tests
├── views_hydranet       # Main source code
│   ├── architecture     # Model definitions (CNN + LSTM + U-net)
│   ├── evaluate         # Evaluation scripts
│   ├── forecast         # Forecasting utilities
│   ├── manager          # Workflow management
│   ├── train            # Training logic
│   ├── utils            # Helper functions (logging, metrics, etc.)
│   ├── __init__.py      # Package initialization
├── .gitignore           # Git ignore rules
├── pyproject.toml       # Poetry project file
├── poetry.lock          # Dependency lock file
```  

---

## 🤝 Contributing  

We welcome contributions to HydraNet! Please follow the contribution guidelines outlined in the [organization-level documentation](https://github.com/views-platform/docs).  

---

## 📜 License  

...

---

## 💬 Acknowledgements  

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

HydraNet builds upon:  

- [UCDP Georeferenced Event Dataset (GED)](https://ucdp.uu.se/) for conflict data.  
- [PRIO Grid](https://grid.prio.org/#/) for spatial resolution.  
- Concepts from Hegre et al. (2019), Hegre et al. (2021), and Vesco et al. (2022).  
- Funding from the **European Research Council** and the **Danish Research Council**.  

Special thanks to the **VIEWS MD&D Team** for their collaboration, guidance, and efforts.  
