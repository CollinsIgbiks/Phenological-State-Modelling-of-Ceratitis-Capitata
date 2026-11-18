"""
compare.py
----------
Comparative analysis: runs the two baselines and the three models through the
same walk-forward evaluation, then writes a metrics table and comparison plots.

Baselines (a model is only useful if it beats these):
  - Persistence:   tomorrow..day 7 = today's catch
  - Moving average: next 7 days = mean of the last 7 days

Run:  python compare.py            (re-uses saved model predictions if present)
      python compare.py --rerun    (retrains everything)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import load_daily, make_windows, HORIZON
from evaluate import walk_forward, metrics, plot_predictions, RESULTS_DIR, TEST_START
from model_svm import SVMForecaster
from model_rnn import RNNForecaster
from model_lstm import LSTMForecaster


class Persistence:
    def fit(self, X, Y): return self
    def predict(self, X): return np.repeat(X[:, -1:, 0], HORIZON, axis=1)     # feature 0 = new_log


class MovingAverage:
    def fit(self, X, Y): return self
    def predict(self, X):
        m = np.log1p(np.expm1(X[:, -7:, 0]).mean(1, keepdims=True))
        return np.repeat(m, HORIZON, axis=1)


MODELS = {
    "Persistence": Persistence,
    "Moving average": MovingAverage,
    "SVM": SVMForecaster,
    "RNN": RNNForecaster,
    "LSTM": LSTMForecaster,
}
COLORS = {"Persistence": "#9e9e9e", "Moving average": "#616161",
          "SVM": "#2a9d8f", "RNN": "#e76f51", "LSTM": "#264653"}


def main(rerun=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    daily = load_daily()
    X, Y, origin = make_windows(daily)

    all_preds, table = {}, {}
    for name, factory in MODELS.items():
        path = os.path.join(RESULTS_DIR, f"preds_{name.lower().replace(' ', '_')}.csv")
        if os.path.exists(path) and not rerun:
            print(f"[{name}] loading saved predictions")
            preds = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            print(f"[{name}] running walk-forward")
            preds = walk_forward(factory, X, Y, origin, verbose=False)
            preds.to_csv(path)
            plot_predictions(preds, daily, name)
        all_preds[name] = preds
        table[name] = metrics(preds)

    table = pd.DataFrame(table).T
    table.to_csv(os.path.join(RESULTS_DIR, "comparison_metrics.csv"))
    print("\n=== Comparison (test period from", TEST_START, ") ===")
    print(table[["MAE_all", "RMSE_all", "MAE_7day_total", "Corr_7day_total"]].round(3).to_string())

    # 1) error by forecast day
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in MODELS:
        ax.plot(range(1, HORIZON + 1), [table.loc[name, f"MAE_d{h}"] for h in range(1, HORIZON + 1)],
                marker="o", color=COLORS[name], label=name,
                ls="--" if name in ("Persistence", "Moving average") else "-")
    ax.set_xlabel("Days ahead"); ax.set_ylabel("MAE (flies / day)")
    ax.set_title("Forecast error by horizon"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "compare_mae_by_horizon.png"), dpi=150); plt.close(fig)

    # 2) predicted vs actual 7-day totals, all models on one chart
    fig, ax = plt.subplots(figsize=(11, 4.5))
    any_p = next(iter(all_preds.values()))
    tcols = [f"true_h{h}" for h in range(1, HORIZON + 1)]
    pcols = [f"pred_h{h}" for h in range(1, HORIZON + 1)]
    ax.plot(any_p.index, any_p[tcols].sum(axis=1), color="black", lw=2.5, label="Actual")
    for name in ("SVM", "RNN", "LSTM"):
        ax.plot(all_preds[name].index, all_preds[name][pcols].sum(axis=1), color=COLORS[name], lw=1.8, label=name)
    ax.set_ylabel("Flies in next 7 days"); ax.set_xlabel("Forecast made on")
    ax.set_title("Predicted vs actual 7-day catches (new-generation signal)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "compare_7day_totals.png"), dpi=150); plt.close(fig)

    # 3) overall bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.index, table["MAE_all"], color=[COLORS[n] for n in table.index])
    ax.set_ylabel("MAE (flies / day)"); ax.set_title("Overall forecast error (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "compare_overall_mae.png"), dpi=150); plt.close(fig)
    print(f"\nTables and figures written to ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main(rerun="--rerun" in sys.argv)
