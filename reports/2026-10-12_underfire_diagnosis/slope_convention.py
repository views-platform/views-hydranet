"""State the body's slope defect in the GATE's own convention, so the two are comparable.

GLOSSARY 5b fits  logit(observed rate) = a + b*logit(gate)  over gate bins; b > 1 = under-dispersed.
The body analogue is  log(mu_empirical) = a + b*log(mu_model)  over cells.
Both regress TRUTH on PREDICTION, so b > 1 means the same thing in both: truth moves faster
than the prediction does.  Reported both ways so neither direction can be quoted as the other.
"""
from pathlib import Path
import numpy as np

t3 = np.load(Path("reports/2026-10-12_underfire_diagnosis") / "t3_result.npz")
EPS = 1e-6
mu_model = t3["MDL"][..., 0].mean(0)
mu_emp = t3["SCOREY"].mean(1)
lm, le = np.log(mu_model + EPS), np.log(mu_emp + EPS)
r = np.corrcoef(lm, le)[0, 1]
b_truth_on_pred = np.polyfit(lm, le, 1)[0]
b_pred_on_truth = np.polyfit(le, lm, 1)[0]
print("  n cells %d" % len(lm))
print("  pearson r                                  %.4f" % r)
print("  sd log mu_model %.4f   sd log mu_emp %.4f  ratio %.4f" % (lm.std(), le.std(), lm.std()/le.std()))
print("  b  (TRUTH on PREDICTION, glossary 5b convention)   %.4f   <- compare to the gate's 1.146-1.456" % b_truth_on_pred)
print("  b' (PREDICTION on TRUTH, the other direction)      %.4f" % b_pred_on_truth)
print("  check r^2 == b*b' :  %.4f vs %.4f" % (b_truth_on_pred * b_pred_on_truth, r ** 2))
print("\n  errors-in-variables note: OLS of truth on prediction is ATTENUATED by noise in the")
print("  predictor, so b=%.2f is a LOWER bound on the compression, exactly as glossary 5b says" % b_truth_on_pred)
print("  for the gate.  The deming/total-least-squares slope (equal-noise assumption) is %.4f."
      % (le.std() / lm.std()))
