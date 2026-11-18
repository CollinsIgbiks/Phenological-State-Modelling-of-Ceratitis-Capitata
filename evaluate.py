"""
evaluate.py
-----------
Shared walk-forward evaluation, metrics and plots. Every model goes through
exactly the same loop, so the comparison is fair.

Walk-forward: starting at TEST_START, every STEP days we
  - retrain the model on all samples whose 7-day target is already in the past,
  - forecast the next 7 days from each origin in the following STEP days.
No future information ever leaks into training.

A model is any object with  fit(X, Y)  and  predict(X)
  X: (n, lookback, n_features)   Y: (n, horizon) in log1p space
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import load_daily, make_windows, HORIZON

TEST_START = "2020-09-01"   # train on July-Aug (1st generation) first, test on Sep-Dec (2nd)
STEP = 7                    # retrain once a week
RESULTS_DIR = "results"


def walk_forward(model_factory, X, Y, origin, horizon=HORIZON,
                 test_start=TEST_START, step=STEP, verbose=True):
    """Returns a DataFrame: one row per forecast origin, columns pred_h1..h7, true_h1..h7 (raw counts)."""
    test_idx = np.where(origin >= pd.Timestamp(test_start))[0]
    rows = []
    for k in range(0, len(test_idx), step):
        block = test_idx[k:k + step]
        o = origin[block[0]]
        # a training sample is usable only if its whole target window ends on/before today
        train_mask = (origin + pd.Timedelta(days=horizon)) <= o
        model = model_factory()
        model.fit(X[train_mask], Y[train_mask])
        P = model.predict(X[block])
        for j, i in enumerate(block):
            row = {"origin": origin[i]}
            for h in range(horizon):
                row[f"pred_h{h+1}"] = max(np.expm1(P[j, h]), 0.0)
                row[f"true_h{h+1}"] = np.expm1(Y[i, h])
            rows.append(row)
        if verbose:
            print(f"  origin {o.date()}: trained on {train_mask.sum()} samples, forecast {len(block)} windows")
    return pd.DataFrame(rows).set_index("origin")


def metrics(preds, horizon=HORIZON):
    """MAE / RMSE per horizon day, plus error on the 7-day total (the generation signal)."""
    out = {}
    for h in range(1, horizon + 1):
        e = preds[f"pred_h{h}"] - preds[f"true_h{h}"]
        out[f"MAE_d{h}"] = e.abs().mean()
    p = preds[[f"pred_h{h}" for h in range(1, horizon + 1)]].to_numpy()
    t = preds[[f"true_h{h}" for h in range(1, horizon + 1)]].to_numpy()
    out["MAE_all"] = np.abs(p - t).mean()
    out["RMSE_all"] = np.sqrt(((p - t) ** 2).mean())
    out["MAE_7day_total"] = np.abs(p.sum(axis=1) - t.sum(axis=1)).mean()
    # correlation of 7-day totals: does the model see the rise/fall of the generation?
    out["Corr_7day_total"] = np.corrcoef(p.sum(axis=1), t.sum(axis=1))[0, 1] if p.sum(axis=1).std() > 0 else np.nan
    return pd.Series(out)


def plot_predictions(preds, daily, name, horizon=HORIZON, out_dir=RESULTS_DIR):
    """Two panels: 1-day-ahead forecast vs actual, and 7-day total forecast vs actual."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].bar(daily.index, daily["new"], color="#c9c9c9", label="Actual new catches")
    ax[0].plot(preds.index + pd.Timedelta(days=1), preds["pred_h1"], color="#1f6fb4", lw=2,
               label=f"{name}: 1-day-ahead forecast")
    ax[0].axvline(pd.Timestamp(TEST_START), color="k", ls="--", lw=1)
    ax[0].text(pd.Timestamp(TEST_START), ax[0].get_ylim()[1] * 0.9, "  test period", fontsize=9)
    ax[0].set_ylabel("Flies / day"); ax[0].legend(frameon=False)
    ax[0].set_title(f"{name} — walk-forward forecasts")

    pcols = [f"pred_h{h}" for h in range(1, horizon + 1)]
    tcols = [f"true_h{h}" for h in range(1, horizon + 1)]
    ax[1].plot(preds.index, preds[tcols].sum(axis=1), color="#444", lw=2, label="Actual next-7-day total")
    ax[1].plot(preds.index, preds[pcols].sum(axis=1), color="#d9480f", lw=2, label=f"{name}: predicted next-7-day total")
    ax[1].set_ylabel("Flies in next 7 days"); ax[1].set_xlabel("Forecast made on")
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_forecast.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def run_model(name, model_factory):
    """Full pipeline for one model: data -> walk-forward -> metrics -> plot -> saved CSVs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    daily = load_daily()
    X, Y, origin = make_windows(daily)
    print(f"[{name}] walk-forward evaluation")
    preds = walk_forward(model_factory, X, Y, origin)
    m = metrics(preds)
    preds.to_csv(os.path.join(RESULTS_DIR, f"preds_{name.lower().replace(' ', '_')}.csv"))
    plot = plot_predictions(preds, daily, name)
    print(m.round(3).to_string())
    print(f"[{name}] plot saved to {plot}")
    return preds, m
