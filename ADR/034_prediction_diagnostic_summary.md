# ADR 034: Automated Prediction Diagnostic Summary

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Human-Parsable Validation of Outbound Data |
| ADR Number          | 034               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
Spatiotemporal forecasts produce high-dimensional DataFrames that are difficult to inspect manually. Silent data corruption (e.g., all values becoming 0.0, or specific channels becoming `NaN` due to inverse-transform failures) often went unnoticed until the data reached the final evaluation stage, leading to wasted research cycles.

## 2. Decision: The "Super Diagnostic" Printout
We implement a mandatory, permanent diagnostic summary that executes at the end of every prediction path (`Evaluation` and `Forecasting`).

### 2.1 Technical Specifications
The diagnostic must provide a column-wise statistical breakdown for every sequence in the output list:
*   **Boundary Detection:** Min, Max, and Mean values for every feature.
*   **Numerical Sanity:** Explicit count of `NaN` and `Inf` values per column.
*   **Stochastic Awareness:** For stochastic runs (`evaluation_mode: stochastic`), the diagnostic must automatically flatten cell-lists to provide global statistics for the posterior distribution.
*   **Visual Indicators:** Stochastic columns must be marked with an asterisk (`*`) for clarity.

### 2.2 Placement
The summary is handled by the `HydranetManager._log_prediction_summary` method, ensuring that no prediction DataFrame leaves the HydraNet domain without being described in the terminal.

## 3. Consequences

**Positive Effects:**
- **Instant Feedback:** Researchers see immediately if a forecast has "collapsed" or "exploded."
- **Data Integrity:** Catching `NaNs` before they hit the evaluation library prevents opaque downstream crashes.
- **Transprency:** Provides a "Boring" verification that the 12-feature schema (ADR 032) is correctly populated.

**Negative Effects:**
- **Terminal Noise:** Increases the amount of text in the console (mitigated by clean table formatting).

## 4. Rationale
In a Boring Architecture, we trust but verify. By making the data's "Vital Signs" visible by default, we eliminate the need for ad-hoc debugging scripts and ensure that every prediction is mathematically plausible.
