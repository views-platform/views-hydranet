# ADR-NNN: <Concise Decision Title>

**Status:** Proposed | Accepted | Superseded | Deprecated  
**Date:** YYYY-MM-DD  
**Deciders:** <Names/Roles>  
**Informed:** <Names/Roles>  

---

## 1. Context
**Why are we doing this now?**
- **Problem:** Describe the current pain point or scientific constraint.
- **Assumptions:** What was previously assumed that is no longer true?
- **Urgency:** Why is a decision required immediately?

---

## 2. Decision
**The new Law of the Land.**
- **Statement:** "We will [Action] to achieve [Outcome]."
- **In-Scope:** Explicitly define what is covered.
- **Out-of-Scope:** Explicitly define what is NOT covered.

---

## 3. Rationale & Integrity Impact
**The logic behind the choice.**
- **Logic:** Why this over Alternative X? (Prioritize *Correctness > Convenience*).
- **Fortress State:** How does this improve **Reproducibility** or **Numerical Stability**?
- **Fail-Loud:** Does this align with the mandate to crash immediately on contract violation?

---

## 4. Consequences
**The honest trade-off.**

### ✅ Positive (Benefits)
- [ ] Simplification of [Component]
- [ ] Reduced risk of [Failure Mode]
- [ ] Improved traceability for [Role]

### ⚠️ Negative (Costs)
- [ ] Increased boilerplate in [Layer]
- [ ] Breaking change requiring migration of [Artifacts]

---

## 5. Validation
**How do we prove it works?**
- **Invariants:** What must remain bit-perfect?
- **Tests:** Which Red/Beige/Green tests (ADR-005) verify this?
- **Failure Mode:** What observation would trigger a reconsideration of this ADR?

---

## 6. Implementation Notes
- **Location:** Where is this enforced (Code, Config, or CI)?
- **References:** PRs, Issues, or Research Papers.

---

### 💡 How to use:
1. Copy to `docs/ADRs/[active|proposed]/NNN_snake_case_title.md`.
2. Keep it **assertive and literal**.
3. If you can't define the **Validation**, the decision is too vague.
