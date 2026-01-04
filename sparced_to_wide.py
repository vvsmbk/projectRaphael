import pandas as pd

df = pd.read_csv("saw.csv")

df["time"] = pd.to_datetime(df["time"])
df["time_1s"] = df["time"].dt.floor("s")   # .dt.round("s") for rounding

value_cols = [c for c in df.columns if c not in ("time", "time_1s")]

out = (
    df.sort_values("time")
      .groupby("time_1s")[value_cols]
      .last()                      # last not-NaN for each column
      .reset_index()
      .rename(columns={"time_1s": "time"})
)

out.to_csv("saw_wide_1s.csv", index=False)
