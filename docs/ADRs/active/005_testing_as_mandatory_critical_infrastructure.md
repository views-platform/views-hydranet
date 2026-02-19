# ADR 005: Testing as Mandatory Critical Infrastructure

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The Red/Beige/Green Testing Taxonomy |
| ADR Number          | 005               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
This repository supports high-stakes spatiotemporal forecasting systems. Failure in such systems is not limited to crashes but includes **silent semantic drift** and **brittle behavior** under realistic conditions. Testing is therefore not a quality signal; it is **critical infrastructure**.

---

## 2. Decision: The Triple-Team Taxonomy
We enforce a strict, three-perspective testing strategy. No component is considered "verified" until it has passed all three perspectives.

### 2.1 🟩 Green Team (Resilience & Accuracy)
*   **Goal:** Ensure the system works as intended under expected conditions.
*   **Mindset:** "How do we make this solid?"
*   **Typical Tests:** Bit-perfect round-trips (`DF -> Volume -> DF`), training convergence, and forecast alignment.
*   **Standard:** These must pass continuously and form the backbone of the CI pipeline.

### 2.2 🟫 Beige Team (Realistic Misuse)
*   **Goal:** Catch failures caused by normal, but potentially dangerous, human behavior.
*   **Mindset:** "What will regular users actually do?"
*   **Typical Tests:** Missing configuration keys, mismatched spatial resolutions, and ambiguous column naming.
*   **Standard:** The system must respond with "Loud and Proud" errors, never silent defaults or "best-effort" execution.

### 2.3 🟥 Red Team (Adversarial)
*   **Goal:** Expose failure modes and vulnerabilities by assuming worst-case or hostile behavior.
*   **Mindset:** "How could this go wrong?"
*   **Typical Tests:** Shuffling input data to test topological stability, providing non-finite values (`NaN`, `Inf`) to test stability guards, and extreme "Out-of-Distribution" configurations.
*   **Standard:** These tests must demonstrate that the system is **Invincible** against identity corruption or data leakage.

---

## 3. Enforcement Rules
*   **No Test, No Merge:** Code that affects behavior must not be merged without corresponding tests.
*   **Happy-Path Only is Failure:** Tests that only cover the Green Team perspective are insufficient.
*   **Loud Failure:** If a failure mode is known, it must be tested and the system must fail explicitly (ADR 042).

---

## 4. Consequences
*   **Scientific Confidence:** Ensures that research results are not artifacts of silent data corruption.
*   **Developer Joy:** Catching errors at the "Handshake" boundary prevents opaque downstream crashes.
*   **Trustworthiness:** Provides a verifiable record of the system's robustness and safety.
