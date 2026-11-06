#!/usr/bin/env python3
"""
MMASH pipeline: overnight heart-rate / HRV stability features vs. sleep quality.

Reads the MMASH data directly from the downloaded .zip (no need to unzip) or
from an already-extracted folder, then:
    1. cleans each participant's beat-to-beat (RR) data,
    2. keeps only the sleep period (from "Onset" to "Out Bed" in sleep.csv),
    3. extracts "physiological stability" features from that night (all sleep
       episodes of the night are combined; time out of bed between them is excluded),
    4. joins them with sleep-quality measures (sleep.csv + PSQI),
    5. reports correlations and a small cross-validated Ridge model.

Install (inside your venv):  pip install pandas matplotlib      (scipy comes with scikit-learn)

Run:
    python mmash_pipeline.py path\\to\\mmash-1.0.0.zip
    python mmash_pipeline.py path\\to\\MMASH.zip --with-activity --out-dir mmash_out

Notes
- With ~22 participants this is EXPLORATORY. Read the numbers as hints, not proof.
- Main features use only heart data. The sleep metrics in sleep.csv are probably
  scored from the wrist accelerometer, so accelerometer features are OFF by
  default (--with-activity) to avoid a circular comparison.
"""

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAY = 86400
USER_RE = re.compile(r"(?:^|[\\/])user_(\d+)[\\/]([^\\/]+\.csv)$", re.IGNORECASE)
SURGE_BPM = 8      # a 1-min HR this far above the night's median counts as a surge
MIN_BEATS_1MIN = 20


# --------------------------------------------------------------------------- #
# 1. Data access (zip, nested zip, or folder)
# --------------------------------------------------------------------------- #
def index_zip(zf: zipfile.ZipFile, index: dict) -> None:
    for name in zf.namelist():
        m = USER_RE.search(name)
        if m:
            index.setdefault(int(m.group(1)), {})[m.group(2).lower()] = \
                (lambda z=zf, n=name: z.read(n))
        elif name.lower().endswith(".zip"):  # e.g. MMASH.zip inside the PhysioNet zip
            inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
            index_zip(inner, index)


def build_index(path: Path) -> dict:
    index: dict = {}
    if path.is_dir():
        for f in path.rglob("*.csv"):
            m = USER_RE.search(str(f))
            if m:
                index.setdefault(int(m.group(1)), {})[m.group(2).lower()] = \
                    (lambda p=f: p.read_bytes())
    elif path.suffix.lower() == ".zip" and path.is_file():
        index_zip(zipfile.ZipFile(path), index)
    else:
        sys.exit(f"[error] Not a zip file or folder: {path}")
    if not index:
        sys.exit("[error] No user_*/ folders with CSV files found. "
                 "Is this the MMASH download?")
    return index


def read_csv(files: dict, name: str) -> pd.DataFrame | None:
    reader = files.get(name.lower())
    if reader is None:
        return None
    df = pd.read_csv(io.BytesIO(reader()))
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


# --------------------------------------------------------------------------- #
# 2. Small helpers
# --------------------------------------------------------------------------- #
def find_col(df: pd.DataFrame, must: list, exclude: list = ()) -> str | None:
    for c in df.columns:
        low = str(c).lower()
        if all(k in low for k in must) and not any(k in low for k in exclude):
            return c
    return None


def to_seconds(series: pd.Series) -> pd.Series:
    """'HH:MM' or 'HH:MM:SS' -> seconds since midnight."""
    parts = series.astype(str).str.strip().str.split(":", expand=True)
    parts = parts.apply(pd.to_numeric, errors="coerce")
    h = parts[0]
    m = parts[1] if 1 in parts.columns else 0
    s = parts[2] if 2 in parts.columns else 0
    return h * 3600 + m * 60 + s


def abs_seconds(df: pd.DataFrame, time_col: str, day_col: str | None) -> pd.Series:
    secs = to_seconds(df[time_col])
    if day_col is not None:
        secs = secs + (pd.to_numeric(df[day_col], errors="coerce") - 1) * DAY
    return secs


def fmt_t(sec: float) -> str:
    """Absolute seconds -> 'day2 00:46' (day 1 = first recording day)."""
    if not np.isfinite(sec):
        return "NaN"
    sec = int(sec)
    return f"day{sec // DAY + 1} {(sec % DAY) // 3600:02d}:{(sec % 3600) // 60:02d}"


def sleep_episodes(sleep: pd.DataFrame) -> list:
    """Absolute (onset, out-of-bed) seconds for EVERY row of sleep.csv.

    MMASH participants can have several sleep episodes in one night
    (e.g. 00:46-03:31 and 03:57-07:30), one row each.
    """
    t_on, d_on = find_col(sleep, ["onset", "time"]), find_col(sleep, ["onset", "date"])
    t_out, d_out = find_col(sleep, ["out", "time"]), find_col(sleep, ["out", "date"])
    if t_on is None or t_out is None:
        raise ValueError("no 'Onset Time' / 'Out Bed Time' column in sleep.csv")
    on = abs_seconds(sleep, t_on, d_on)
    out = abs_seconds(sleep, t_out, d_out)
    out = out.where(out > on, out + DAY)          # episode crossing midnight
    episodes = [(a, b) for a, b in zip(on, out) if np.isfinite(a) and np.isfinite(b)]
    if not episodes:
        raise ValueError("no valid sleep episode in sleep.csv")
    return sorted(episodes)


def in_episodes(t: pd.Series, episodes: list) -> np.ndarray:
    """Boolean mask: which timestamps fall inside any sleep episode."""
    mask = np.zeros(len(t), dtype=bool)
    for a, b in episodes:
        mask |= ((t >= a) & (t <= b)).to_numpy()
    return mask


# --------------------------------------------------------------------------- #
# 3. Cleaning and feature extraction
# --------------------------------------------------------------------------- #
def clean_rr(rr: pd.DataFrame, by_wrap: bool = False) -> pd.DataFrame:
    """Clean beat-to-beat data.

    by_wrap=True ignores the file's 'day' column (some participants have unusable
    day numbers) and rebuilds the timeline from the clock: the recording is in
    order, so every time the clock jumps back by >12 h it has passed midnight and
    the day number goes up by one. The first beat is taken as day 1, which is the
    convention sleep.csv uses (day 2 = after the first midnight).
    """
    ibi_col = find_col(rr, ["ibi"])
    time_col = find_col(rr, ["time"])
    day_col = find_col(rr, ["day"])
    if by_wrap or day_col is None:
        secs = to_seconds(rr[time_col])
        t = secs + (secs.diff() < -DAY / 2).cumsum() * DAY
    else:
        t = abs_seconds(rr, time_col, day_col)
    df = pd.DataFrame({
        "ibi": pd.to_numeric(rr[ibi_col], errors="coerce").values,
        "t": t.values,
    }).reset_index(drop=True)
    ok = df["ibi"].between(0.3, 2.0)                       # physiological range
    med = df["ibi"].where(ok).rolling(5, center=True, min_periods=2).median()
    ok &= (df["ibi"] - med).abs() / med < 0.20             # drop sudden jumps (artifacts)
    df["valid"] = ok
    df["idx"] = np.arange(len(df))
    return df


def hrv_features(night: pd.DataFrame, onset: float, hours: float) -> dict:
    """`hours` = total time inside sleep episodes (gaps between episodes excluded)."""
    v = night[night["valid"]]
    ibi = v["ibi"].to_numpy()
    hr = 60.0 / ibi
    # successive differences only between truly adjacent beats
    adj = v["idx"].diff().to_numpy() == 1
    d = np.diff(ibi)[adj[1:]] if len(ibi) > 1 else np.array([])

    f = {
        "valid_beat_frac": len(v) / max(len(night), 1),
        "hr_mean": hr.mean(),
        "hr_std": hr.std(),
        "hr_p5": np.percentile(hr, 5),
        "hr_p95": np.percentile(hr, 95),
        "sdnn_ms": ibi.std() * 1000,
        "rmssd_ms": np.sqrt(np.mean(d ** 2)) * 1000 if len(d) else np.nan,
        "pnn50": np.mean(np.abs(d) > 0.05) * 100 if len(d) else np.nan,
    }

    # per-minute and per-5-minute HR (night "stability" over time)
    v = v.assign(minute=((v["t"] - onset) // 60).astype(int),
                 block5=((v["t"] - onset) // 300).astype(int))
    g1 = v.groupby("minute")["ibi"].agg(["mean", "size"])
    hr_min = (60.0 / g1.loc[g1["size"] >= MIN_BEATS_1MIN, "mean"])
    if len(hr_min) >= 10:
        f["hr_surges_per_hour"] = (hr_min > hr_min.median() + SURGE_BPM).sum() / max(hours, 1e-9)
        third = max(len(hr_min) // 3, 1)
        f["hr_trend_last_minus_first_third"] = hr_min.iloc[-third:].mean() - hr_min.iloc[:third].mean()
    g5 = v.groupby("block5")["ibi"]
    hr5 = 60.0 / g5.mean()
    if len(hr5) >= 3:
        f["hr_5min_cv"] = hr5.std() / hr5.mean()
        f["rmssd_5min_cv"] = np.nan
        rm = []
        for _, blk in v.groupby("block5"):
            b_ibi = blk["ibi"].to_numpy()
            b_adj = blk["idx"].diff().to_numpy() == 1
            b_d = np.diff(b_ibi)[b_adj[1:]] if len(b_ibi) > 1 else np.array([])
            if len(b_d) >= 30:
                rm.append(np.sqrt(np.mean(b_d ** 2)) * 1000)
        if len(rm) >= 3:
            f["rmssd_5min_cv"] = np.std(rm) / np.mean(rm)
    return f


def activity_features(act: pd.DataFrame, episodes: list) -> dict:
    vm_col = find_col(act, ["vector"])
    time_col = find_col(act, ["time"])
    day_col = find_col(act, ["day"])
    if None in (vm_col, time_col):
        return {}
    t = abs_seconds(act, time_col, day_col)
    vm = pd.to_numeric(act[vm_col], errors="coerce")[in_episodes(t, episodes)]
    return {"act_vm_mean": vm.mean(), "act_vm_p90": vm.quantile(0.9)}


def num(df: pd.DataFrame, must: list, exclude: list = (), prefer: str | None = None) -> pd.Series:
    """First numeric column whose name contains all `must` words."""
    cands = [c for c in df.columns
             if all(k in str(c).lower() for k in must)
             and not any(k in str(c).lower() for k in exclude)]
    if prefer:  # e.g. prefer 'Sleep Fragmentation Index' over 'Fragmentation Index'
        cands = sorted(cands, key=lambda c: prefer not in str(c).lower())
    if not cands:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[cands[0]], errors="coerce")


def wavg(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan


def sleep_targets(sleep: pd.DataFrame, quest: pd.DataFrame | None) -> dict:
    """Combine ALL sleep episodes of the night into one value per participant.

    Durations and counts are summed; efficiency and the fragmentation index are
    averaged weighted by time in bed (efficiency then equals total sleep time /
    total time in bed, which matches the per-episode values in sleep.csv).
    """
    tib = num(sleep, ["total minutes in bed"])
    tst = num(sleep, ["total sleep"])
    awk = num(sleep, ["number", "awakening"])
    tib_sum = tib.sum(min_count=1)
    awk_sum = awk.sum(min_count=1)
    out = {
        "n_episodes": len(sleep),
        "time_in_bed_min": tib_sum,
        "total_sleep_min": tst.sum(min_count=1),
        "num_awakenings": awk_sum,                       # raw count (not a target)
        "sleep_efficiency": wavg(num(sleep, ["efficiency"]), tib),
        "waso_min": num(sleep, ["waso"]).sum(min_count=1),
        "awakenings_per_hour": awk_sum / (tib_sum / 60) if tib_sum and tib_sum > 0 else np.nan,
        "sleep_fragmentation_idx": wavg(
            num(sleep, ["fragmentation"], ["movement"], prefer="sleep"), tib),
        "psqi": np.nan,
    }
    if quest is not None:
        c = find_col(quest, ["psqi"]) or find_col(quest, ["pittsburgh"])
        if c is not None:
            out["psqi"] = pd.to_numeric(quest[c].iloc[0], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# 4. Analysis
# --------------------------------------------------------------------------- #
def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def correlations(df: pd.DataFrame, feats: list, targets: list) -> pd.DataFrame:
    rows = []
    for t in targets:
        for f in feats:
            sub = df[[f, t]].dropna()
            if len(sub) < 8 or sub[f].nunique() < 3 or sub[t].nunique() < 3:
                continue
            rho, p = spearmanr(sub[f], sub[t])
            rows.append({"feature": f, "target": t, "n": len(sub), "rho": rho, "p": p})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["q_fdr"] = bh_qvalues(res["p"].values)
        res = res.reindex(res["rho"].abs().sort_values(ascending=False).index)
    return res


def loo_models(df: pd.DataFrame, feats: list, targets: list) -> pd.DataFrame:
    rows = []
    for t in targets:
        sub = df.dropna(subset=[t])
        if len(sub) < 10 or sub[t].nunique() < 3:
            continue
        X, y = sub[feats], sub[t]
        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              RidgeCV(alphas=np.logspace(-1, 3, 15)))
        base = DummyRegressor(strategy="mean")
        p_r = cross_val_predict(ridge, X, y, cv=LeaveOneOut())
        p_b = cross_val_predict(base, X, y, cv=LeaveOneOut())
        rows.append({"target": t, "n": len(sub),
                     "MAE_ridge": mean_absolute_error(y, p_r),
                     "MAE_baseline": mean_absolute_error(y, p_b),
                     "R2_ridge": r2_score(y, p_r)})
    return pd.DataFrame(rows)


def heatmap(corr: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping chart (pip install matplotlib).")
        return
    piv = corr.pivot(index="feature", columns="target", values="rho")
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4, 0.38 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(piv.values[i, j]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Spearman rho")
    ax.set_title("Night physiology features vs. sleep-quality measures")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping chart (pip install matplotlib).")
        return None


def scatter_top_features(df: pd.DataFrame, corr: pd.DataFrame, target: str,
                         path: Path, k: int = 4) -> None:
    """One dot per participant: top-k features (by |Spearman rho|) vs. the primary outcome."""
    plt = get_plt()
    top = corr[corr["target"] == target].head(k)
    if plt is None or top.empty:
        return
    n = len(top)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.3 * nrows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        sub = df[["user", r["feature"], target]].dropna()
        x, y = sub[r["feature"]].to_numpy(float), sub[target].to_numpy(float)
        ax.scatter(x, y, s=38, color="steelblue", edgecolor="white", zorder=3)
        for u, xi, yi in zip(sub["user"], x, y):
            ax.annotate(str(u), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if len(sub) >= 3 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="darkorange", lw=1.8, zorder=2)
        ax.set_xlabel(r["feature"])
        ax.set_ylabel(target)
        ax.set_title(f"rho = {r['rho']:.2f}, q_fdr = {r['q_fdr']:.3f}, n = {int(r['n'])}", fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Top features vs. {target} (each dot = one participant, label = user ID)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {path}")


HRV_FEATS = {"sdnn_ms", "rmssd_ms", "pnn50", "rmssd_5min_cv"}


def ablation(df: pd.DataFrame, feats: list, target: str) -> pd.DataFrame:
    """Leave-one-out error of Ridge using different groups of input signals."""
    act = [c for c in feats if c.startswith("act_")]
    hrv = [c for c in feats if c in HRV_FEATS]
    hr = [c for c in feats if c not in act and c not in HRV_FEATS]
    sets = {"HR only": hr, "HRV only": hrv, "HR + HRV": hr + hrv}
    if act:
        sets["Activity only"] = act
        sets["HR + HRV + activity"] = hr + hrv + act

    sub = df.dropna(subset=[target])
    y = sub[target]
    base_pred = cross_val_predict(DummyRegressor(strategy="mean"), sub[feats], y, cv=LeaveOneOut())
    base_mae = mean_absolute_error(y, base_pred)
    rows = []
    for name, cols in sets.items():
        if not cols:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              RidgeCV(alphas=np.logspace(-1, 3, 15)))
        pred = cross_val_predict(model, sub[cols], y, cv=LeaveOneOut())
        rows.append({"input_signals": name, "n_features": len(cols), "n": len(sub),
                     "MAE": mean_absolute_error(y, pred), "MAE_baseline": base_mae,
                     "R2": r2_score(y, pred)})
    return pd.DataFrame(rows)


def ablation_chart(res: pd.DataFrame, target: str, path: Path) -> None:
    plt = get_plt()
    if plt is None or res.empty:
        return
    base = res["MAE_baseline"].iloc[0]
    fig, ax = plt.subplots(figsize=(1.6 * len(res) + 3, 4.8))
    colors = ["seagreen" if m < base else "gray" for m in res["MAE"]]
    bars = ax.bar(res["input_signals"], res["MAE"], color=colors, zorder=3)
    ax.axhline(base, color="firebrick", ls="--", lw=1.6, label=f"Baseline: predict the mean ({base:.2f})")
    for b, r2 in zip(bars, res["R2"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"R2={r2:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"Leave-one-out MAE of {target}\n(lower is better)")
    ax.set_title(f"Which input signals help predict {target}?\n"
                 "green = beats the baseline", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="MMASH overnight physiology pipeline")
    ap.add_argument("source", type=Path, help="MMASH .zip (or extracted folder)")
    ap.add_argument("--out-dir", type=Path, default=Path("mmash_out"))
    ap.add_argument("--primary", default="sleep_efficiency",
                    choices=["sleep_efficiency", "waso_min", "awakenings_per_hour",
                             "sleep_fragmentation_idx", "psqi"],
                    help="primary outcome (y), chosen BEFORE looking at results")
    ap.add_argument("--with-activity", action="store_true",
                    help="also add accelerometer features (see circularity note)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(args.source)
    print(f"Found {len(index)} participants.")

    rows, skipped = [], []
    for uid in sorted(index):
        files = index[uid]
        try:
            rr, sleep = read_csv(files, "RR.csv"), read_csv(files, "sleep.csv")
            missing = [n for n, d in (("RR.csv", rr), ("sleep.csv", sleep))
                       if d is None or d.empty]
            if missing:
                raise ValueError(f"missing/empty {', '.join(missing)}; "
                                 f"files present: {sorted(files)}")
            episodes = sleep_episodes(sleep)
            sleep_hours = sum(b - a for a, b in episodes) / 3600
            beats = clean_rr(rr)
            night = beats[in_episodes(beats["t"], episodes)]
            if night["valid"].sum() < 1000:
                beats2 = clean_rr(rr, by_wrap=True)
                night2 = beats2[in_episodes(beats2["t"], episodes)]
                if night2["valid"].sum() >= 1000:
                    print(f"[note] user_{uid}: RR 'day' column unusable "
                          f"(values {[int(v) for v in sorted(rr[find_col(rr, ['day'])].dropna().unique())[:6]]}); "
                          "day numbers rebuilt from the clock (midnight wrap-around).")
                    beats, night = beats2, night2
            if night["valid"].sum() < 1000:
                t_ok = beats["t"].dropna()
                raise ValueError(
                    f"only {int(night['valid'].sum())} valid beats during sleep | "
                    f"sleep window: " + ", ".join(f"{fmt_t(a)}-{fmt_t(b)}" for a, b in episodes) +
                    f" | RR file: {len(rr)} rows, covers "
                    f"{fmt_t(t_ok.min()) if len(t_ok) else 'n/a'} to "
                    f"{fmt_t(t_ok.max()) if len(t_ok) else 'n/a'}, "
                    f"{int(beats['t'].isna().sum())} unparsed times, "
                    f"{int(beats['valid'].sum())} valid beats overall")
            feats = hrv_features(night, episodes[0][0], sleep_hours)
            if args.with_activity:
                act = read_csv(files, "Actigraph.csv")
                if act is not None:
                    feats.update(activity_features(act, episodes))
            targets = sleep_targets(sleep, read_csv(files, "questionnaire.csv"))
            rows.append({"user": uid, "sleep_hours_window": sleep_hours,
                         **feats, **targets})
        except Exception as exc:
            skipped.append((uid, str(exc)))

    for uid, why in skipped:
        print(f"[skip] user_{uid}: {why}")
    if len(rows) < 8:
        sys.exit("[error] Too few usable participants to analyse.")

    df = pd.DataFrame(rows)
    all_targets = ["sleep_efficiency", "waso_min", "awakenings_per_hour",
                   "sleep_fragmentation_idx", "psqi"]
    target_cols = [c for c in all_targets if c in df and df[c].notna().sum() >= 8]
    for c in all_targets:
        if c in df and c not in target_cols:
            print(f"[warn] outcome '{c}' has too few values and is skipped "
                  "(it is NOT used as an input feature).")
    # bookkeeping columns: never used as model inputs (X)
    meta = ["user", "sleep_hours_window", "valid_beat_frac", "n_episodes",
            "time_in_bed_min", "total_sleep_min", "num_awakenings"]
    feat_cols = [c for c in df.columns if c not in meta + all_targets]

    print("\n" + "=" * 60 + f"\nUSABLE NIGHTS: {len(df)}\n" + "=" * 60)
    print("Sleep-quality measures (describe):")
    print(df[target_cols].describe().loc[["count", "mean", "std", "min", "max"]].round(2).T.to_string())
    print(f"\nMedian valid-beat fraction during sleep: {df['valid_beat_frac'].median():.2f}")

    df.to_csv(args.out_dir / "mmash_features.csv", index=False, encoding="utf-8-sig")

    corr = correlations(df, feat_cols, target_cols)
    if not corr.empty:
        corr.to_csv(args.out_dir / "mmash_correlations.csv", index=False, encoding="utf-8-sig")
        print("\n" + "=" * 60 + "\nTOP SPEARMAN CORRELATIONS (q_fdr corrects for many tests)\n" + "=" * 60)
        print(corr.head(12).round(3).to_string(index=False))
        n_sig = int((corr["q_fdr"] < 0.10).sum())
        print(f"\nPairs with q_fdr < 0.10: {n_sig} of {len(corr)} tested.")
        heatmap(corr, args.out_dir / "mmash_summary.png")

    if not corr.empty and args.primary in set(corr["target"]):
        print("\n" + "=" * 60 + f"\nPRIMARY OUTCOME: {args.primary}\n" + "=" * 60)
        print(corr[corr["target"] == args.primary].head(6).round(3).to_string(index=False))
        print("(all other outcomes are secondary / exploratory)")
        scatter_top_features(df, corr, args.primary, args.out_dir / "mmash_scatter_primary.png")

    if args.primary in target_cols:
        abl = ablation(df, feat_cols, args.primary)
        if not abl.empty:
            abl.to_csv(args.out_dir / "mmash_ablation.csv", index=False, encoding="utf-8-sig")
            print("\n" + "=" * 60 + f"\nABLATION on {args.primary} (leave-one-out Ridge)\n" + "=" * 60)
            print(abl.round(3).to_string(index=False))
            if not args.with_activity:
                print("(activity rows need --with-activity; see circularity note at the top)")
            ablation_chart(abl, args.primary, args.out_dir / "mmash_ablation.png")

    res = loo_models(df, feat_cols, target_cols)
    if not res.empty:
        res.to_csv(args.out_dir / "mmash_loo_models.csv", index=False, encoding="utf-8-sig")
        print("\n" + "=" * 60 + "\nLEAVE-ONE-OUT RIDGE vs. 'predict the mean' BASELINE\n" + "=" * 60)
        print(res.round(3).to_string(index=False))
        print("\nRidge only helps if MAE_ridge < MAE_baseline and R2_ridge > 0.")

    print(f"\nFiles written to {args.out_dir}/")


if __name__ == "__main__":
    main()
