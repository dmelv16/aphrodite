"""One power parquet: DC1/DC2 28V voltage+current per station/save, labeled with
status, transients, test case and test run. Samples outside test case windows are dropped.
Usage: python power_tca_to_parquet.py <main_folder>   Requires pandas + pyarrow."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"<path_to_main_folder>") / "data"
OUT = Path("power.parquet")
UNIT_ID = ""
OFP = ""

ON_V = 22.0      # first sample above this = energized; steady-state floor
OFF_V = 5.0      # below this = fully de-energized
STAB_N = 20      # first N energized samples = stabilizing
TRANS_V = 18.0   # MIL-STD-704 28VDC undervoltage transient floor
TRANS_S = 0.1    # max dip duration the envelope allows (approx, tune)
DATE_FMTS = ["%m/%d/%Y", "%Y-%m-%d", "%Y-%b-%d", "%d-%b-%Y"]
TCA_FMT = "%Y-%b-%d %H:%M:%S.%f"
EPOCH, SEC = pd.Timestamp("1970-01-01"), pd.Timedelta(1, "s")
COLS = ["unit_id", "station", "save", "dc1_status", "dc1_trans", "dc2_status", "dc2_trans", "timestamp",
        "voltage_28v_dc1_cal", "current_28v_dc1_cal", "voltage_28v_dc2_cal", "current_28v_dc2_cal",
        "test_case", "test_run", "ofp"]


def read_tca(d):
    t = pd.read_csv(d / "CSV" / "tca" / "tca_data.csv", dtype=str)
    ts = lambda i: (pd.to_datetime(t.iloc[:, i].str.strip(), format=TCA_FMT) - EPOCH) / SEC
    return pd.DataFrame({"test_case": t.iloc[:, 0], "start": ts(2), "end": ts(4)})


def read_dc(d, n):
    rd = lambda q: pd.read_csv(next(d.rglob(f"{q}_28v_dc{n}*.csv")), engine="pyarrow", usecols=["Date", "Seconds", "Value"],
                               dtype={"Date": str, "Seconds": str, "Value": "float64"}
                               ).rename(columns={"Value": f"{q}_28v_dc{n}_cal"})
    return rd("voltage").merge(rd("current"), on=["Date", "Seconds"])


def date_s(s):  # parse each unique date once -> epoch seconds at midnight
    u = pd.Series(s.unique())
    dt = pd.concat([pd.to_datetime(u, format=f, errors="coerce") for f in DATE_FMTS], axis=1).bfill(axis=1).iloc[:, 0]
    return s.map(dict(zip(u, (dt - EPOCH) / SEC)))


def label(df, n):
    v, ts = df[f"voltage_28v_dc{n}_cal"].to_numpy(), df["timestamp"].to_numpy()
    on = pd.Series(np.where(v > ON_V, 1.0, np.where(v < OFF_V, 0.0, np.nan))).ffill().fillna(0).to_numpy()
    seg = np.cumsum(np.r_[False, (on[1:] == 1) & (on[:-1] == 0)])
    k = pd.Series(seg).groupby(seg).cumcount().to_numpy()
    dip = (on == 1) & (k >= STAB_N) & (v < ON_V)
    g = pd.DataFrame({"v": v, "t": ts, "nx": np.r_[on[1:], 1]}).groupby(np.cumsum(np.r_[True, dip[1:] != dip[:-1]]))
    # dip is a real transient if it breaks the 704 envelope and recovers (a dip into OFF is a power-down)
    real = ((g.v.transform("min") < TRANS_V) | (g.t.transform("max") - g.t.transform("min") > TRANS_S)) \
        & (g.nx.transform("last") == 1)
    df[f"dc{n}_status"] = np.select([on == 0, k < STAB_N], ["de-energized", "stabilizing"], "steady state")
    df[f"dc{n}_trans"] = dip & real.to_numpy()
    return df


saves = [(d, d.parent.name.replace("_", ""), int(d.name.rsplit("save", 1)[1]))
         for d in sorted(ROOT.glob("DAM_Data_*/_8_*_RT_*/*_save*"))]

# test_run: per test case, windows overlapping across stations = one run; numbered by start time
tca = pd.concat([read_tca(d).assign(station=st, save=sv) for d, st, sv in saves]).sort_values(["test_case", "start"], ignore_index=True)
prev_end = tca.groupby("test_case")["end"].cummax().groupby(tca["test_case"]).shift()
r = (tca["start"] > prev_end).groupby(tca["test_case"]).cumsum()
tca["test_run"] = tca.groupby(["test_case", r])["start"].transform("min").rank(method="dense").astype("int32")

w, rows = None, 0
try:
    for d, st, sv in saves:
        df = read_dc(d, 1).merge(read_dc(d, 2), on=["Date", "Seconds"])
        df["timestamp"] = date_s(df["Date"]) + df["Seconds"].astype("float64")
        df = label(label(df.sort_values("timestamp", ignore_index=True), 1), 2)
        win = tca[(tca.station == st) & (tca.save == sv)].sort_values("start")[["start", "end", "test_case", "test_run"]]
        df = pd.merge_asof(df, win, left_on="timestamp", right_on="start")
        df = df[df["timestamp"] <= df["end"]].assign(unit_id=UNIT_ID, station=st, save=sv, ofp=OFP)
        df["test_run"] = df["test_run"].astype("int32")
        t = pa.Table.from_pandas(df[COLS], preserve_index=False)
        w = w or pq.ParquetWriter(OUT, t.schema, compression="zstd")
        w.write_table(t)
        rows += len(t)
        print(f"{st} save{sv}: {len(t):,} rows")
finally:
    if w: w.close()
print(f"{rows:,} rows -> {OUT}")
