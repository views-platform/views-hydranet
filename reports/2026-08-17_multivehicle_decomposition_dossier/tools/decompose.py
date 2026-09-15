import csv, statistics as S
from pathlib import Path
R = Path("reports/2026-08-17_multivehicle_decomposition_dossier/results")
CTL = {}
for r in csv.DictReader(open("reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv")):
    if r["target"] == "sb":
        CTL.setdefault(r["model"], {})[int(r["h"])] = r
for r in csv.DictReader(open("reports/2026-08-17_placement_intervention_dossier/results/score_blue_stranger_rollout.csv")):
    if r["target"] == "sb":
        CTL.setdefault("blue_stranger", {})[int(r["h"])] = r
V = ["purple_alien", "blue_stranger", "blazing_meteor"]
ARMS = ["use_real", "spatial_scramble", "occurrence_real_magnitude_model",
        "occurrence_model_magnitude_real", "thin_0.75"]

def sc(m, n):
    return {int(r["h"]): r for r in csv.DictReader(open(R / f"score_{m}_{n}.csv")) if r["target"] == "sb"}

def fed(m, n):
    rows = [r for r in csv.DictReader(open(R / f"fedfield_{m}_{n}.csv")) if r["target_idx"] == "0"]
    g = lambda k: S.mean(float(r[k]) for r in rows if float(r["n_active"]) > 0)
    return g("active_fraction"), g("neighbour_pairs_per_active"), g("mean_magnitude_on_active")

print("FALSIFIERS, recorded before predictions")
print()
for m in V:
    probs = []
    ctl = CTL[m]
    h1 = [float(sc(m, a)[1]["AP"]) for a in ARMS] + [float(ctl[1]["AP"])]
    if max(h1) - min(h1) > 1e-6:
        probs.append("F1 h1 spread %.2e" % (max(h1) - min(h1)))
    for a in ARMS:
        for h, r in sc(m, a).items():
            if int(r["N"]) != 170430:
                probs.append("F2 %s h%d N=%s" % (a, h, r["N"]))
    au, nu, mu = fed(m, "use_real")
    a2, n2, _ = fed(m, "spatial_scramble")
    if abs(au - a2) > 1e-6:
        probs.append("F3 scramble af")
    if n2 / nu >= 0.5:
        probs.append("F3 clustering %.3f" % (n2 / nu))
    a3, _, m3 = fed(m, "occurrence_real_magnitude_model")
    if abs(au - a3) > 1e-6:
        probs.append("F3 E4a af")
    if abs(m3 - mu) / mu <= 0.05:
        probs.append("F3 E4a magnitudes")
    a4, _, _ = fed(m, "thin_0.75")
    if abs(a4 / au - 0.25) / 0.25 > 0.05:
        probs.append("F3 thin %.4f" % (a4 / au))
    gap = float(sc(m, "use_real")[18]["AP"]) - float(ctl[18]["AP"])
    if gap < 0.05:
        probs.append("F4 gap %.4f" % gap)
    print("  %-16s %s   (gap %.4f, thin ratio %.4f)"
          % (m, "ALL PASS" if not probs else "FIRED: " + "; ".join(probs), gap, a4 / au))

print()
print("DECOMPOSITION — share of the oracle gap at h18, target sb")
print()
print("  %-16s %11s %10s %10s %10s" % ("vehicle", "occurrence", "magnitude", "thin:0.75", "scrambled"))
print("  %-16s %11s %10s %10s %10s" % ("violet_visitor", "95.3%", "1.4%", "95.5%", "-93.7%"))
for m in V:
    c = float(CTL[m][18]["AP"])
    g = float(sc(m, "use_real")[18]["AP"]) - c
    f = lambda n: "%.1f%%" % (100 * (float(sc(m, n)[18]["AP"]) - c) / g)
    print("  %-16s %11s %10s %10s %10s" % (m, f("occurrence_real_magnitude_model"),
          f("occurrence_model_magnitude_real"), f("thin_0.75"), f("spatial_scramble")))
