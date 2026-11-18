"""
data_prep.py
------------
Turns the raw hourly file (bug_ml.csv) into a clean DAILY table that every
model uses, so all three models see exactly the same data.

What it does
  1. Un-scales the temperatures back to deg C (file values were divided by 44.743).
  2. Aggregates hourly rows to one row per day (mean / min / max temperature).
  3. Computes degree-days from the hourly temperatures (the empty Deg_Day column).
  4. Converts the CUMULATIVE trap count (bug_cnt) into NEW catches per day.
  5. Builds sliding windows: last LOOKBACK days of features -> next HORIZON days of catches.

Run it on its own to check the data:  python data_prep.py
"""
import numpy as np
import pandas as pd

# ---------------- configuration (change here, everything else follows) ----------------
DATA_PATH = "bug_ml.csv"
SCALE_FACTOR = 44.743        # from the Scl_fac_avg cell in the file
LOWER_THRESHOLD = 15.0       # deg C, from the file (literature values are ~10-13 C, try them!)
UPPER_THRESHOLD = 35.0       # deg C, from the file
END_DATE = "2020-12-12"      # count jumps 518 -> 665 on 13 Dec then stays flat:
                             # looks like trap removal / data entry, so it is excluded.
                             # Set to None to keep the whole series.
LOOKBACK = 14                # days of history the models see
HORIZON = 7                  # days ahead to predict (the 7-day window in the brief)

FEATURES = ["new_log", "t_mean", "t_min", "t_max", "dd", "cum_dd"]


def load_daily(path=DATA_PATH, lower=LOWER_THRESHOLD, upper=UPPER_THRESHOLD, end_date=END_DATE):
    raw = pd.read_csv(path)
    raw = raw[["c_year", "c_month", "c_day", "c_hours", "Temp_avg", "Temp_min", "Temp_max", "bug_cnt"]]
    raw["date"] = pd.to_datetime(dict(year=raw.c_year, month=raw.c_month, day=raw.c_day))

    for c in ["Temp_avg", "Temp_min", "Temp_max"]:
        raw[c] = raw[c] * SCALE_FACTOR

    # Hourly degree-days with a horizontal cut-off at the upper threshold:
    # each hour contributes (T - lower)/24, limited to the range [0, upper - lower]/24.
    raw["dd_hour"] = (raw.Temp_avg - lower).clip(lower=0, upper=upper - lower) / 24.0

    daily = raw.groupby("date").agg(
        t_mean=("Temp_avg", "mean"),
        t_min=("Temp_min", "min"),
        t_max=("Temp_max", "max"),
        dd=("dd_hour", "sum"),
        cum_count=("bug_cnt", "max"),   # one reading per day (at 22:00)
    )

    if end_date is not None:
        daily = daily.loc[:end_date]

    daily["cum_count"] = daily["cum_count"].ffill().fillna(0)
    daily["new"] = daily["cum_count"].diff().fillna(daily["cum_count"].iloc[0]).clip(lower=0)
    daily["new_log"] = np.log1p(daily["new"])        # counts are spiky; log makes them easier to learn
    daily["cum_dd"] = daily["dd"].cumsum()
    return daily


def make_windows(daily, features=FEATURES, lookback=LOOKBACK, horizon=HORIZON):
    """
    Sample i uses days [t-lookback+1 .. t] to predict new catches on days [t+1 .. t+horizon].
    Returns
      X      (n, lookback, n_features)
      Y      (n, horizon)  in log1p space
      origin DatetimeIndex of day t for every sample (the "forecast made on" date)
    """
    F = daily[features].to_numpy(dtype=float)
    y = daily["new_log"].to_numpy(dtype=float)
    X, Y, origin = [], [], []
    for t in range(lookback - 1, len(daily) - horizon):
        X.append(F[t - lookback + 1: t + 1])
        Y.append(y[t + 1: t + 1 + horizon])
        origin.append(daily.index[t])
    return np.array(X), np.array(Y), pd.DatetimeIndex(origin)


if __name__ == "__main__":
    d = load_daily()
    print(d.head(10).round(2).to_string())
    print("\nDays:", len(d), "| total new catches:", int(d["new"].sum()),
          "| total degree-days:", round(d["dd"].sum(), 1))
    X, Y, o = make_windows(d)
    print("Windows:", X.shape, "->", Y.shape, "| first origin", o[0].date(), "| last", o[-1].date())
