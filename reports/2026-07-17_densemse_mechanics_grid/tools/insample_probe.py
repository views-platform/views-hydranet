"""In-sample T=0 body-magnitude probe — dense-mse grid (2026-07-17).

Teacher-forced ONE-step forward on a REAL in-sample volume (month 440 -> predict 441; no 36-step
rollout -> no C-113 bloom). E[y]=expm1(reg) (standard emit=identity log1p->count). Per arm/target:
frac_alive (E[y]>0.1 on positive-truth cells), ratio_med (median E[y]/truth on positives), pred_max
(count-divergence check), mean E[y] on positives. Answers: softplus revive vs relu dead? count diverge?
Retrain-free. Writes summary txt + comparison plot.
"""

from __future__ import annotations
import importlib.util
import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, "/home/simon/Documents/scripts/views_platform/views-hydranet")
from views_hydranet.utils.utils import choose_model
from views_hydranet.utils.data_fetcher import DataFetcher
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.volume_handler import VolumeHandler

ARTS = "/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/artifacts"
SC = (
    "/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/"
    "faf1c146-3d1e-4018-bf86-28d563362784/scratchpad"
)
PARQ = (
    "/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/"
    "data/raw/calibration_viewser_df.parquet"
)
MONTH0_ID = 121  # month index = month_id - 121
IN_MONTH = 440  # in-sample input month (<=456); predict IN_MONTH+1
FEAT = slice(5, 8)  # channel_map indices of lr_sb/ns/os (log1p)
TARGETS = ["sb", "ns", "os"]

ARMS = {
    "sp_log_s42": "calibration_model_20260717_022510",
    "sp_log_s43": "calibration_model_20260717_024214",
    "relu_log_s42": "calibration_model_20260717_025908",
    "relu_log_s43": "calibration_model_20260717_031602",
    "sp_count_s42": "calibration_model_20260717_033339",
    "sp_count_s43": "calibration_model_20260717_035124",
    "relu_count_s42": "calibration_model_20260717_040858",
    "relu_count_s43": "calibration_model_20260717_042642",
}


def load_cfg(label):
    spec = importlib.util.spec_from_file_location("c_" + label, f"{SC}/config_dm_{label}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.get_hp_config()
    cfg["run_type"] = "calibration"
    return cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = load_cfg("sp_log_s42")
    df = pd.read_parquet(PARQ)
    df = DataFetcher.standardize_raw_df(df, cfg0)
    df = FeatureScaler(cfg0).fit_transform(df)
    vol = VolumeHandler.from_df(df, cfg0).data  # (T,H,W,C)
    i = IN_MONTH - MONTH0_ID
    x_np = vol[i, :, :, FEAT].astype("float32")  # (H,W,3) log1p features at t
    truth_np = np.expm1(vol[i + 1, :, :, FEAT].astype("float32"))  # (H,W,3) counts at t+1
    x = torch.from_numpy(np.transpose(x_np, (2, 0, 1)))[None].to(device)  # [1,3,H,W]

    rows = []
    for label, ts in ARMS.items():
        cfg = load_cfg(label)
        model = choose_model(cfg, device)
        sd = torch.load(f"{ARTS}/{ts}.pt", map_location=device)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            h = model.init_hTtime(hidden_channels=model.base, H=x.shape[-2], W=x.shape[-1]).to(
                device
            )
            reg = model(x, h).reg[0].cpu().numpy()  # (3,H,W) log-space body
        ey = np.expm1(np.clip(reg, None, 40))  # count-space E[y]; clip only to avoid inf in print
        for ti, tg in enumerate(TARGETS):
            tr = truth_np[:, :, ti]
            e = ey[ti]
            pos = tr > 0
            if pos.sum() == 0:
                continue
            r = e[pos] / tr[pos]
            rows.append(
                dict(
                    arm=label,
                    tgt=tg,
                    npos=int(pos.sum()),
                    frac_alive=float((e[pos] > 0.1).mean()),
                    ratio_med=float(np.median(r)),
                    ey_mean_pos=float(e[pos].mean()),
                    ey_max=float(e.max()),
                    reg_max=float(reg[ti].max()),
                )
            )
    res = pd.DataFrame(rows)
    res.to_csv(f"{SC}/insample_probe.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    with open(f"{SC}/insample_probe.txt", "w") as fh:
        fh.write(
            f"IN-SAMPLE T=0 BODY PROBE (month {IN_MONTH}->{IN_MONTH + 1}, teacher-forced 1-step)\n\n"
        )
        for tg in TARGETS:
            fh.write(f"=== lr_{tg}_best ===\n")
            sub = res[res.tgt == tg].copy()
            fh.write(
                sub[
                    ["arm", "npos", "frac_alive", "ratio_med", "ey_mean_pos", "ey_max", "reg_max"]
                ].to_string(index=False)
                + "\n\n"
            )
    print(open(f"{SC}/insample_probe.txt").read())

    # plot: frac_alive (alive vs dead) + ey_max (divergence) per arm, target sb
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = list(ARMS.keys())
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for j, tg in enumerate(TARGETS):
            sub = res[res.tgt == tg].set_index("arm").reindex(order)
            xpos = np.arange(len(order))
            ax[j].bar(
                xpos,
                sub["frac_alive"].values,
                color=["#2a7" if s.startswith("sp") else "#a33" for s in order],
            )
            ax[j].set_xticks(xpos)
            ax[j].set_xticklabels(order, rotation=90, fontsize=7)
            ax[j].set_title(f"lr_{tg}: frac of positive cells ALIVE (E[y]>0.1)")
            ax[j].set_ylim(0, 1)
            ax[j].axhline(0, color="k", lw=0.5)
        fig.tight_layout()
        fig.savefig(f"{SC}/insample_probe_frac_alive.png", dpi=110)
        print("plot ->", f"{SC}/insample_probe_frac_alive.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    main()
