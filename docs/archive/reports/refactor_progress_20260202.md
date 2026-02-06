# Refactor Progress Report: Spatiotemporal Integrity (ADR 007 & 009)

**Date:** 02-02-2026  
**Status:** 100% Core Foundation Restored

---

## 1. Verified Components (Locked Down)

The "Boring Architecture" is now fully established for the data bridge. All components are part of the permanent CI suite.

| Component | Status | Spec | Key Falsification Overcome |
| :--- | :--- | :--- | :--- |
| `VolumeHandler` | **LOCKED** | ADR 007 | Topographic drift in reconstruction. |
| `VolumeSampler` | **LOCKED** | ADR 013 | Geographic context loss in windows. |
| `Curriculum (Transparency)` | **LOCKED** | ADR 012 | Added available cell count to progress bar. |
| `Mixed Salad (Oscillation)` | **LOCKED** | ADR 011 | Implemented window-level target interleaving. |
| `Optimization Gate` | **LOCKED** | ADR 014 | Enforced gradient accumulation per Lesson. |
| `Stochastic Integrity`| **LOCKED** | ADR 007 | Preserved 5D samples as lists in DataFrames. |
| `Terminology Alignment` | **LOCKED** | ADR 008 | Renamed samples/batch_size to total_lessons/windows_per_lesson. |
| `Uncertainty Resolution`| **LOCKED** | ADR 008 | Renamed test_samples to n_posterior_samples. |
| `Config Handshake`     | **LOCKED** | ADR 008 | Enforced Fail-Fast validation (no magic migration). |
| `Zero-Magic Defaults` | **LOCKED** | ADR 008 | Purged all hidden defaults; explicit config mandatory. |
| `Lens`                 | **LOCKED** | ADR 013 | Pure geometric tool (no training knowledge). |
| `Sentinel`             | **LOCKED** | ADR 018 | DataSniffer enforces spatiotemporal contracts. |
| `Normalizer`           | **LOCKED** | ADR 019 | Stateful FeatureScaler with bit-perfect reversibility. |
| `Utilities Migration`  | **IN PROGRESS** | N/A | Isolating 11 legacy files to utils/legacy/. |
| `Geometric (Flip/Permute)` | **LOCKED** | ADR 007 | Augmentation regressions in training. |

### VolumeSampler Breakthroughs:
*   **The Mini-Custodian:** Windows are no longer naked tensors; they are mini-`VolumeHandler` objects.
*   **Absolute Slicing:** Window offsets are mathematically calculated to preserve absolute geographic coordinates.
*   **Handshake Rigor:** The sampler now validates spatial bounds at initialization.

---

## 2. Recovery Context
*   **Handled:** Inbound, Transformation, Outbound, and Stochastic Windowing.
*   **Next Horizon:** Trainer integration. Ensuring the `train()` loop consumes these mini-handlers without magic indices.
