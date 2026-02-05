# ADR 029: Geographic Anchors for Autoregressive Stability

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Positional Embeddings vs Hidden State Freezing |
| ADR Number          | 029               |
| Status              | Proposed          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## 1. Problem Statement: The Hallucination Feedback Loop

HydraNet currently relies on a Recurrent U-Net (LSTM) architecture to capture spatiotemporal dynamics. During autoregressive forecasting (Month 2+), the model consumes its own previous predictions as input. 

**The Collapse:** 
In an autoregressive state, the hidden state ($h$) begins to drift. Because standard CNNs are **Translation Invariant**, the model has no fixed geographic reference point. Without Ground Truth to 'anchor' the hidden state, errors accumulate. The model effectively 'forgets' where it is geographically, leading to a total performance collapse between Months 10 and 13.

**Current Mitigation:** 
The `execute_freeze_h_option` acts as a tactical intervention, manually preventing parts of the hidden state from updating. This preserves the 'Map of the World' at the cost of temporal sensitivity.

---

## 2. Conceptual Solution: Fixed Geographic Anchors

The proposal is to inject **Static Positional Embeddings** directly into the input tensor. By providing the model with a permanent, non-recurrent sense of "Where," we reduce the burden on the hidden state to "remember" geography.

### Proposed Implementation (CoordConv Pattern):
Instead of just feeding $[T, 32, 32, 3]$ (Conflict Features), we feed $[T, 32, 32, 5]$. The two new channels are static:
1. **Normalized Latitude ($y$):** A linear gradient from -1.0 (South) to 1.0 (North).
2. **Normalized Longitude ($x$):** A linear gradient from -1.0 (West) to 1.0 (East).

---

## 3. Topological Constraints: The Land/Water Mask

While coordinates ($x, y$) provide a geographic address, they do not explicitly encode the physical reality of terrestrial vs. aquatic space.

### The "Water Problem"
Expecting a model to learn coastline topology from $(x, y)$ gradients is inefficient. In sparse conflict datasets, the model may not see enough "Ocean" samples to learn that violence is physically impossible there. This leads to "Coastline Hallucination," where autoregressive noise bleeds into the sea.

### The Land Mask Channel
We propose adding a third static channel: a **Binary Land Mask** (Land=1, Water=0).
* **Source:** Derived directly from `priogrid_gid > 0`.
* **Impact:** Provides an absolute topological barrier. It allows CNN kernels to learn that conflict propagation terminates at the water's edge. 
* **Autoregressive Stability:** Acts as a "Sanity Check" for the model's own guesses at $t=20+$.

---

## 4. Critical Assessment: Pros and Cons

### Pros
* **Geographic Stability:** Provides an absolute frame of reference that noise cannot distort.
* **Topological Rigidity:** Hardwires the physical boundaries of the world, preventing ocean hallucinations.
* **Boring Math:** Linear gradients and binary masks are computationally cheap and easy for early-layer filters to digest.
* **Translation Awareness:** Allows the model to learn localized geographic behaviors.

### Cons / Risks
* **The "Shortcut" Risk:** The model might stop learning dynamics and overfit to geographic averages.
* **Channel Dilution:** Adding static channels could drown out sparse conflict features.
* **Model Size:** Increasing input channels from 3 to 6 requires a minor update to the first layer of the architecture.

---

## 5. Implementation Considerations: The Boring Path

1. **Gate Placement:** All context channels (x, y, mask) should be generated inside the `VolumeHandler` at construction time.
2. **Deterministic Augmentation:** If a spatial flip occurs, the coordinate gradients and land mask must flip in perfect synchrony with the features.
3. **Training Synergy:** Use **Input Dropout** on the context channels to force the model to prioritize conflict signals while still acknowledging geographic constraints.

---

## 6. Conclusion: Towards Autoregressive Autonomy

By combining **Identity (x, y)** with **Topography (mask)**, we provide the model with a Permanent Frame of Reference. This infrastructure shift aims to eliminate the need for manual hidden-state "Brain Locks," allowing the model to evolve its temporal understanding without losing its geographic anchor. 

---

## 7. References & Further Reading

To deepen the understanding of CoordConv and spatial awareness in CNNs, the following papers are recommended:

1. **"An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"** (Liu et al., 2018). 
   * *Significance:* The seminal paper introducing the CoordConv layer. It demonstrates that standard CNNs struggle with simple coordinate transforms and shows how adding extra coordinate channels solves this failing.
2. **"How Much Position Information Do Convolutional Neural Networks Encode?"** (Islam et al., 2020). 
   * *Significance:* An ICLR study proving that while CNNs implicitly learn some position information via zero-padding (the "border effect"), explicit positional embeddings significantly improve performance on tasks requiring spatial precision.
