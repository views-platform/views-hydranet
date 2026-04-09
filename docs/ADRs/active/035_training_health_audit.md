# ADR 035: Training Health Audit and Spectral Monitoring

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Mathematical Validation of the Learning Process |
| ADR Number          | 035               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
Training Recurrent U-Nets (HydraNet) is numerically hazardous. Vanishing gradients can lead to "Dead Layers," while spectral radii > 1.0 can cause exponential weight explosion. Previously, training runs were often considered "successful" if they completed without crashing, even if the resulting weights were mathematically broken.

## 2. Decision: Post-Training Mathematical Audit
We implement a mandatory health audit that evaluates the internal state of the model immediately after the `training_loop` completes.

### 2.1 Loss Convergence Metrics
The audit must report the Final, Minimum, and Maximum Lesson Loss to verify that the optimizer successfully descended the gradient slope.

### 2.2 Spectral Health (L2 Norms)
The audit must calculate and display the **L2 Norm** of the weight parameters for every layer.
*   **The Sweet Spot (✅):** Norms between 0.01 and 100.0 indicate healthy, active neurons.
*   **Vanishing Gradients (💀):** Norms close to 0.0 indicate "Dead Layers" that have stopped learning.
*   **Exploding Gradients (⚠️):** Norms > 100.0 indicate instability and imminent numerical collapse.

### 2.3 Final Verdict
The audit provides a binary "Health Verdict." If any weight or loss value is non-finite (`NaN` or `Inf`), the run is flagged as a **CRITICAL FAILURE**, regardless of whether a model artifact was saved.

## 3. Consequences

**Positive Effects:**
- **Scientific Confidence:** Confirms that the model's "brain" is mathematically functional.
- **Educational Value:** Helps developers understand the internal dynamics of U-Net convergence.
- **Early Detection:** Identifies "Zombie Models" (models that exist but predict noise) immediately.

**Negative Effects:**
- **Compute Overhead:** Minor delay at the end of training to calculate norms (neglegible compared to training time).

## 4. Rationale
A saved `.pt` file is not a proof of success. In a Boring Architecture, we prioritize mathematical health over process completion. By auditing the spectral weight of the model, we ensure that HydraNet remains within a stable "Performance Corridor."
