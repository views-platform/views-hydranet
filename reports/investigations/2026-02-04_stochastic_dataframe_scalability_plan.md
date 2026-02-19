# Research Plan: Scalable Stochastic DataFrame Construction

| Plan Info           | Details |
|---------------------|---------|
| Subject             | Solving the "List-in-Cell" RAM Bottleneck |
| Status              | Proposed |
| Target Artifact     | ADR for Stochastic DataFrame Construction |
| Date                | 04.02.2026 |

## 1. Context & Problem Statement
The HydraNet pipeline operates efficiently in the Tensor domain (Time × Space × Channels × Samples). However, the final "Handshake" with the Evaluation Library requires converting this efficient tensor into a **pandas DataFrame** where each cell contains a Python list of 128 float samples.

### The Physics of Failure
*   **The Bottleneck:** It is not the *data size* (bytes) but the *object count* (metadata) that kills the process. Creating millions of individual Python `list` objects and hundreds of millions of `float` objects triggers massive heap fragmentation and overhead (The "Object Tax").
*   **Current State:** A 3-target run requires ~30GB RAM, which is disproportionate to the actual information content (~2-3GB).
*   **Constraint:** The output format (Pandas DataFrame with list-columns) is a fixed constraint from the `eval_dataframe_describtion.md`. We cannot change the Consumer; we must fix the Producer.

## 2. Research Objective
Identified a "Boring", robust, and scalable mechanism to bridge the gap between `numpy.ndarray` (5D) and `pandas.DataFrame` (List-Col) without incurring the Python Object Tax.

**Success Metric:** Reduce RAM overhead by >70% (target <8GB for a standard run) while keeping the code simple and readable.

## 3. Research Avenues (Candidate Architectures)

### Avenue A: The Modern Standard (PyArrow-Backed Pandas)
Pandas 2.0+ introduced formal support for PyArrow backends, which support complex types (like Lists) natively in contiguous memory.
*   **Hypothesis:** Using `pd.Series(..., dtype="list[float64][pyarrow]")` will eliminate the Python Object creation entirely, keeping the data in a zero-copy Arrow buffer while presenting a Pandas API.
*   **"Boring" Factor:** High. It's just a dtype change.

### Avenue B: The Polars Bridge
Polars handles list-columns natively and efficiently.
*   **Hypothesis:** Constructing the table in Polars first, then converting to Pandas using `to_pandas(use_pyarrow_extension_array=True)`, might bypass the slow Python interpreter loop and the object overhead.
*   **"Boring" Factor:** Medium. Introduces a dependency (Polars) but removes complex custom logic.

### Avenue C: Chunked Sequential Construction
If RAM is strictly limited, we bypass "The Big DF" entirely.
*   **Hypothesis:** We define a generator that yields monthly DataFrames, writes them to a Parquet stream, and then reads the whole set back. 
*   **"Boring" Factor:** Low. This is a fallback if in-memory solutions (A/B) fail.

## 4. Execution Steps (The Method)

### Step 1: Baseline Micro-Benchmark
I will extend `tests/test_memory_fingerprint.py` to create a rigorous "Object Tax" profile.

**Findings from Step 1 (04.02.2026):**
*   **Pandas (List-in-Cell):** ~16x RAM overhead (Object Tax).
*   **Pandas (NumPy-in-Cell):** ~20x RAM overhead.
*   **Conclusion:** Storing individual NumPy arrays in cells is *more* expensive than Python lists due to the overhead of millions of small NumPy headers. Fragmentation is the primary enemy. We must move to a **contiguous list representation** (Arrow/Polars native).

### Step 2: Feasibility Spike (PyArrow/Polars)
I will write a probe script `research/probe_reconstruction_efficiency.py` to test the actual conversion speed and RAM peak.

**Findings from Step 2 (04.02.2026):**
*   **Both Avenue A (Arrow-Pandas) and Avenue B (Polars) are highly efficient.**
*   **Memory Result:** Both show ~2.1x RAM overhead (down from 16x).
*   **Speed Result:** Polars is **instantaneous (0.02s)** while Pandas-Arrow takes **~1.5s** per column.
*   **Conclusion:** Polars is the technically superior engine for construction, but Arrow-backed Pandas is the best "Boring" interface for the final handshake.

## 5. Implementation Roadmap
1.  **Intermediate Polars Bridge**: Use Polars to construct the table from Arrow arrays (for speed/memory safety).
2.  **Zero-Copy Handshake**: Convert the final Polars table to an Arrow-backed Pandas DataFrame using `to_pandas(use_pyarrow_extension_array=True)`.
3.  **Compatibility Check**: Ensure the resulting DataFrame passes the `validate_contract_dataframes` check in `HydranetManager`.
4.  **ADR 023**: Document this "Polars Bridge" as the standard for all spatiotemporal-to-dataframe transitions.

