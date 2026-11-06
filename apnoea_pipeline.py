#!/usr/bin/env python3
"""
Simulated-apnoea dataset (PhysioNet respiratory-oximetry-apnoea): can a NECK-WORN
optical sensor (wearable-like) detect a breath-hold, using heart, breathing and
optical features? Evaluated across people (leave-one-subject-out).

Install (inside your venv):    pip install pandas matplotlib

Run (path to the downloaded .zip or the extracted folder):
    python apnoea_pipeline.py path\\to\\respiratory-oximetry-apnoea-1.0.0.zip
    python apnoea_pipeline.py path\\to\\folder --peeps 0,4,8

Data layout used (from the dataset page and the first rows of two files)
    Inline_PQ_Data/SubjectN/SubjectN_<peep>cmH2O_<normal|apnea1|apnea2>.csv
        Time [s], Gauge Pressure [cmH2O], Inspiratory differential pressure [cmH2O]
        (airway sensor behind a face mask: NOT wearable; used only as ground truth)
    Neck_Pulse_Oximeter_Data/SubjectN/SubjectN_<peep>cmH2O_<cond>_pulse.csv
        Time [s], PD1..PD4 (assumed 660 nm), PD1_9..PD4_9 (assumed 940 nm); ~250 Hz
    Demographic_Data.csv  (optional: used to compare our heart rate with the finger reference)
    normal = 60 s of relaxed breathing; apnea1 = 3 breaths then a 10 s breath-hold;
    apnea2 = 3 breaths then a 20 s breath-hold.

Prediction target: trial contains a breath-hold (apnea1/apnea2) vs normal breathing.
Inputs: features computed from the NECK sensor only.
"""

import argparse
import io
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, medfilt, sosfiltfilt, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 42
HOLD_GAP_S = 8.0          # no inspiration for this long = breath-hold (normal gaps are ~3-6 s)
FILE_RE = re.compile(r"Subject(\d+)_(\d+)cmH2O_(normal|apn(?:o)?ea[12])(_pulse)?\.csv$", re.I)
PRE, POST = 8, 30         # event-locked window (seconds before / after hold start)


# --------------------------------------------------------------------------- #
# 1. Finding and reading files
# --------------------------------------------------------------------------- #
def index_source(path: Path):
    index, demo = {}, []

    def add(name: str, reader):
        base = name.replace("\\", "/").split("/")[-1]
        m = FILE_RE.search(base)
        if m:
            subj, peep = int(m.group(1)), int(m.group(2))
            cond = m.group(3).lower().replace("apnoea", "apnea")
            kind = "neck" if (m.group(4) or "neck" in name.lower()) else "inline"
            index.setdefault((subj, peep, cond), {})[kind] = reader
        elif base.lower() == "demographic_data.csv":
            demo.append(reader)

    if path.is_dir():
        for f in path.rglob("*.csv"):
            add(str(f), lambda p=f: p.read_bytes())
    elif path.is_file() and path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            if n.lower().endswith(".csv"):
                add(n, lambda zz=z, nn=n: zz.read(nn))
    else:
        sys.exit(f"[error] Not a zip file or folder: {path}")
    if not index:
        sys.exit("[error] No Subject*_cmH2O_*.csv files found. Is this the apnoea dataset?")
    return index, (demo[0] if demo else None)


def read_csv(reader) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(reader()), encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 2. Inline airway sensor -> ground truth for the breath-hold
# --------------------------------------------------------------------------- #
def inline_analysis(df: pd.DataFrame, fs: int = 100) -> dict:
    cols = list(df.columns)
    dp_col = next((c for c in cols if "differential" in c.lower()), cols[2])
    t = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy()
    dp = pd.to_numeric(df[dp_col], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(dp)
    t, dp = t[ok], dp[ok]
    t, first = np.unique(t, return_index=True)
    dp = dp[first]
    grid = np.arange(t[0], t[-1], 1 / fs)
    dp_i = uniform_filter1d(np.interp(grid, t, dp), int(0.2 * fs))

    base, top = np.percentile(dp_i, 5), np.percentile(dp_i, 95)
    amp = max(top - base, 1e-6)
    peaks, _ = find_peaks(dp_i, prominence=0.3 * amp, distance=int(1.2 * fs))
    times = grid[peaks] - grid[0]
    out = {"grid": grid - grid[0], "dp": dp_i, "peak_times": times,
           "inline_n_breaths": len(peaks), "inline_max_gap_s": np.nan,
           "inline_resp_rate": np.nan, "hold_onset_s": np.nan, "hold_end_s": np.nan}
    if len(times) >= 3:
        gaps = np.diff(times)
        k = int(np.argmax(gaps))
        out["inline_max_gap_s"] = float(gaps[k])
        out["hold_onset_s"], out["hold_end_s"] = float(times[k]), float(times[k + 1])
        rest = np.delete(gaps, k)
        out["inline_resp_rate"] = 60.0 / np.median(rest) if len(rest) else np.nan
    out["inline_hold_detected"] = int(out["inline_max_gap_s"] >= HOLD_GAP_S) \
        if np.isfinite(out["inline_max_gap_s"]) else 0
    return out


# --------------------------------------------------------------------------- #
# 3. Neck optical sensor -> features (the "wearable" inputs)
# --------------------------------------------------------------------------- #
def sos_band(x, lo, hi, fs, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def beat_intervals(sig, fs):
    peaks, _ = find_peaks(sig, prominence=0.3 * np.std(sig), distance=int(0.35 * fs))
    t = peaks / fs
    ibi = np.diff(t)
    ok = (ibi > 0.33) & (ibi < 1.5)
    med = pd.Series(np.where(ok, ibi, np.nan)).rolling(5, center=True, min_periods=2).median().to_numpy()
    ok &= np.abs(ibi - med) / med < 0.25
    idx = np.flatnonzero(ok)
    return t[1:][ok], ibi[ok], np.diff(idx) == 1


def neck_analysis(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    t = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy()
    X = df[cols[1:9]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(X).all(1)
    t, X = t[ok], X[ok]
    fs = int(round(1 / np.median(np.diff(t))))
    if np.std(np.diff(t)) > 0.1 / fs:                       # uneven timestamps -> resample
        grid = np.arange(t[0], t[-1], 1 / fs)
        X = np.column_stack([np.interp(grid, t, X[:, k]) for k in range(X.shape[1])])
        t = grid
    X = np.column_stack([medfilt(X[:, k], 9) for k in range(X.shape[1])])   # remove short spikes
    n_sec = int(len(X) // fs)

    pulse = sos_band(X, 0.7, 3.5, fs)
    dc = X.mean(0)
    resp = sos_band(X / dc, 0.1, 0.6, fs)                   # breathing-related change, relative to DC

    # best pulse channel = most concentrated spectral peak in the pulse band
    snr = []
    for k in range(pulse.shape[1]):
        f, p = welch(pulse[:, k], fs=fs, nperseg=min(len(X), 8 * fs))
        b = (f >= 0.7) & (f <= 3.5)
        snr.append(p[b].max() / p[b].sum() if p[b].sum() > 0 else 0)
    kp = int(np.argmax(snr))

    feats = {"pulse_snr": float(snr[kp]), "pulse_channel": kp + 1}
    bt, ibi, adj = beat_intervals(pulse[:, kp], fs)
    if len(ibi) >= 8:
        hr = 60 / ibi
        d = np.diff(ibi)[adj] if len(ibi) > 1 else np.array([])
        feats.update({"hr_mean": hr.mean(), "hr_std": hr.std(), "hr_min": hr.min(),
                      "hr_max": hr.max(), "hr_range": hr.max() - hr.min(),
                      "sdnn_ms": ibi.std() * 1000,
                      "rmssd_ms": np.sqrt(np.mean(d ** 2)) * 1000 if len(d) >= 5 else np.nan})
        sec = np.arange(n_sec) + 0.5
        hr_1hz = np.interp(sec, bt, hr)
    else:
        hr_1hz = np.full(n_sec, np.nan)

    # breathing seen by the neck sensor
    kr = int(np.argmax(resp.std(0)))
    r = resp[:, kr]
    f, p = welch(r, fs=fs, nperseg=len(r))
    band = (f >= 0.1) & (f <= 0.6)
    peak = np.argmax(p[band])
    feats["resp_rate_bpm"] = float(f[band][peak] * 60)
    feats["resp_peak_ratio"] = float(p[band][peak] / p[band].sum())
    feats["resp_amp"] = float(r.std())
    env = uniform_filter1d(np.abs(hilbert(r)), 2 * fs)
    quiet = env < 0.3 * np.median(env)
    feats["resp_quiet_frac"] = float(quiet.mean())
    feats["resp_quiet_max_s"] = longest_run(quiet) / fs
    env_1hz = env[: n_sec * fs].reshape(n_sec, fs).mean(1)

    # optical features: ratio-of-ratios (uncalibrated oxygenation proxy) and pulse amplitude
    w = 3 * fs
    nw = len(X) // w
    ror, dcr = [], []
    for i in range(nw):
        sl = slice(i * w, (i + 1) * w)
        rr_k, dd_k = [], []
        for k in range(4):
            ac_r, ac_i = pulse[sl, k].std(), pulse[sl, k + 4].std()
            dc_r, dc_i = X[sl, k].mean(), X[sl, k + 4].mean()
            if min(ac_r, ac_i, dc_r, dc_i) > 1e-6:
                rr_k.append((ac_r / dc_r) / (ac_i / dc_i))
                dd_k.append(np.log(dc_r / dc_i))
        ror.append(np.median(rr_k) if rr_k else np.nan)
        dcr.append(np.mean(dd_k) if dd_k else np.nan)
    ror, dcr = np.array(ror), np.array(dcr)
    good = np.isfinite(ror)
    if good.sum() >= 6:
        feats["ror_median"] = float(np.median(ror[good]))
        feats["ror_cv"] = float(np.std(ror[good]) / abs(np.mean(ror[good])))
        feats["ror_slope_per_min"] = float(np.polyfit(np.flatnonzero(good) * 3 / 60, ror[good], 1)[0])
        g2 = np.isfinite(dcr)
        feats["dc_ratio_slope_per_min"] = float(np.polyfit(np.flatnonzero(g2) * 3 / 60, dcr[g2], 1)[0])
    ac_w = np.array([pulse[i * w:(i + 1) * w, kp].std() for i in range(nw)])
    if nw >= 6 and np.median(ac_w) > 0:
        feats["ac_cv"] = float(ac_w.std() / ac_w.mean())
        feats["ac_min_rel"] = float(ac_w.min() / np.median(ac_w))
    ac_1hz = np.array([pulse[i * fs:(i + 1) * fs, kp].std() for i in range(n_sec)])

    series = {"hr": hr_1hz, "resp_env": env_1hz, "ac": ac_1hz}
    feats["neck_series"] = series
    feats["resp_ds"] = (r[:: fs // 25], 25)            # downsampled breathing trace for plots
    return feats


FEATURE_GROUPS = {
    "Cardiac": ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "sdnn_ms", "rmssd_ms"],
    "Respiration": ["resp_rate_bpm", "resp_peak_ratio", "resp_amp", "resp_quiet_frac", "resp_quiet_max_s"],
    "Optical": ["ror_median", "ror_cv", "ror_slope_per_min", "dc_ratio_slope_per_min", "ac_cv", "ac_min_rel"],
}
INLINE_FEATS = ["inline_max_gap_s", "inline_n_breaths", "inline_resp_rate"]


# --------------------------------------------------------------------------- #
# 4. Models and evaluation (leave-one-subject-out)
# --------------------------------------------------------------------------- #
def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1))


def cluster_ci(y, p, groups, reps=500):
    """Bootstrap over SUBJECTS (not trials) for an AUC confidence interval."""
    rng = np.random.default_rng(SEED)
    subs = np.unique(groups)
    idx = {s: np.flatnonzero(groups == s) for s in subs}
    vals = []
    for _ in range(reps):
        pick = rng.choice(subs, len(subs), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], p[ii]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if len(vals) > 50 else (np.nan, np.nan)


def ablation(df, y, groups):
    sets = {name: cols for name, cols in FEATURE_GROUPS.items()}
    sets["Cardiac + Optical"] = FEATURE_GROUPS["Cardiac"] + FEATURE_GROUPS["Optical"]
    sets["Cardiac + Respiration"] = FEATURE_GROUPS["Cardiac"] + FEATURE_GROUPS["Respiration"]
    sets["All neck signals"] = [c for g in FEATURE_GROUPS.values() for c in g]
    sets["Inline airway (reference, not wearable)"] = INLINE_FEATS
    rows = []
    for name, cols in sets.items():
        cols = [c for c in cols if c in df.columns and df[c].notna().any()]
        if not cols:
            continue
        X = df[cols].to_numpy(float)
        for mname, maker in (("Logistic", make_lr), ("RandomForest", make_rf)):
            p = cross_val_predict(maker(), X, y, groups=groups, cv=LeaveOneGroupOut(),
                                  method="predict_proba")[:, 1]
            lo, hi = cluster_ci(y, p, groups)
            rows.append({"signals": name, "n_features": len(cols), "model": mname,
                         "AUC": roc_auc_score(y, p), "AUC_lo": lo, "AUC_hi": hi,
                         "AP": average_precision_score(y, p), "AP_chance": y.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Charts
# --------------------------------------------------------------------------- #
def get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping charts (pip install matplotlib).")
        return None


def event_locked(store: pd.DataFrame):
    """Average neck-sensor signals around the start of the longest no-inspiration gap."""
    offs = np.arange(-PRE, POST + 1)
    rows = []
    for _, r in store.iterrows():
        on = r["hold_onset_s"]
        if not np.isfinite(on):
            continue
        s, o = r["series"], int(round(on))
        rec = {"subject": r["subject"], "cond": r["cond"]}
        for name in ("hr", "resp_env", "ac"):
            x = s[name]
            seg = np.full(len(offs), np.nan)
            for j, off in enumerate(offs):
                if 0 <= o + off < len(x):
                    seg[j] = x[o + off]
            base = np.nanmean(seg[(offs >= -6) & (offs <= -1)])
            if not np.isfinite(base) or base == 0:
                continue
            rec[name] = seg - base if name == "hr" else seg / base
        if all(k in rec for k in ("hr", "resp_env", "ac")):
            rows.append(rec)
    return offs, rows


def event_chart(store, path: Path):
    plt = get_plt()
    if plt is None:
        return
    offs, rows = event_locked(store)
    if not rows:
        return
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    spec = [("hr", "Heart rate change (bpm)"), ("resp_env", "Breathing signal (x baseline)"),
            ("ac", "Pulse amplitude (x baseline)")]
    colors = {"normal": "gray", "apnea1": "steelblue", "apnea2": "firebrick"}
    labels = {"normal": "Normal breathing (aligned at longest gap)",
              "apnea1": "10 s breath-hold", "apnea2": "20 s breath-hold"}
    for a, (key, ylabel) in zip(ax, spec):
        for cond in ("normal", "apnea1", "apnea2"):
            per_subj = {}
            for r in rows:
                if r["cond"] == cond:
                    per_subj.setdefault(r["subject"], []).append(r[key])
            if len(per_subj) < 3:
                continue
            m = np.array([np.nanmean(v, 0) for v in per_subj.values()])
            mean, sem = np.nanmean(m, 0), np.nanstd(m, 0) / np.sqrt(len(m))
            a.plot(offs, mean, color=colors[cond], label=f"{labels[cond]} (n={len(m)})")
            a.fill_between(offs, mean - sem, mean + sem, color=colors[cond], alpha=0.18)
        a.axvline(0, color="k", ls=":", lw=1)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Neck-sensor signals around the start of a breath-hold (mean +/- SEM across people)")
    ax[-1].set_xlabel("Seconds from the last breath before the pause")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def example_chart(store, path: Path):
    plt = get_plt()
    if plt is None:
        return
    pairs = store.pivot_table(index=["subject", "peep"], columns="cond", values="hold_onset_s",
                              aggfunc="first")
    pairs = pairs.dropna(subset=["normal", "apnea2"]) if {"normal", "apnea2"} <= set(pairs.columns) else pairs.iloc[0:0]
    if pairs.empty:
        return
    subj, peep = pairs.index[0]
    fig, ax = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for j, cond in enumerate(("normal", "apnea2")):
        r = store[(store.subject == subj) & (store.peep == peep) & (store.cond == cond)].iloc[0]
        ax[0, j].plot(r["inline_t"], r["inline_dp"], color="seagreen", lw=0.8)
        for t in r["peak_times"]:
            ax[0, j].axvline(t, color="seagreen", alpha=0.25, lw=0.8)
        rs, rfs = r["resp_ds"]
        ax[1, j].plot(np.arange(len(rs)) / rfs, rs, color="teal", lw=0.9)
        ax[2, j].plot(np.arange(len(r["series"]["hr"])) + 0.5, r["series"]["hr"], color="crimson", lw=1)
        ax[0, j].set_title(f"Subject {subj}, {peep} cmH2O, {'normal breathing' if cond == 'normal' else '20 s breath-hold'}")
        ax[2, j].set_xlabel("Seconds")
    ax[0, 0].set_ylabel("Airway flow signal\n(reference)")
    ax[1, 0].set_ylabel("Neck sensor:\nbreathing component")
    ax[2, 0].set_ylabel("Neck sensor:\nheart rate (bpm)")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def ablation_chart(abl: pd.DataFrame, path: Path):
    plt = get_plt()
    if plt is None or abl.empty:
        return
    order = list(dict.fromkeys(abl["signals"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    for j, (m, color) in enumerate((("Logistic", "steelblue"), ("RandomForest", "darkorange"))):
        sub = abl[abl.model == m].set_index("signals").reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * width
        err = np.vstack([sub["AUC"] - sub["AUC_lo"], sub["AUC_hi"] - sub["AUC"]])
        err = np.clip(np.nan_to_num(err), 0, None)          # interval can be narrower than the point value
        ax.bar(x, sub["AUC"], width, color=color, label=m, zorder=3,
               yerr=err, error_kw={"elinewidth": 1, "capsize": 2})
    ax.axhline(0.5, color="firebrick", ls="--", lw=1.5, label="Chance (0.5)")
    ax.set_xticks(np.arange(len(order)), order, rotation=20, ha="right")
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUC (leave-one-subject-out)")
    ax.set_title("Which neck-sensor signals detect a breath-hold? (bars = 95% CI over people)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #
def check_against_finger(demo_reader, trials: pd.DataFrame) -> None:
    """Compare our per-person mean heart rate with the finger-oximeter value in Demographic_Data.csv."""
    if demo_reader is None:
        return
    d = None
    try:
        d = read_csv(demo_reader)
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")] if d.shape[1] > 2 else d
        hr_col = next((c for c in d.columns
                       if "heart" in str(c).lower() or re.search(r"\bhr\b", str(c).lower())), None)
        if hr_col is None:
            print(f"[info] No heart-rate column found in Demographic_Data.csv. Columns: {list(d.columns)}")
            return
        id_col = next((c for c in d.columns
                       if re.search(r"subject|participant|^id$", str(c).lower())), None)
        if id_col is not None:
            ids = pd.to_numeric(d[id_col].astype(str).str.strip(), errors="coerce")   # 1, 2, 3 ...
            if ids.notna().sum() < 5:                                                  # e.g. "Subject3"
                ids = pd.to_numeric(d[id_col].astype(str).str.extract(r"^\s*Subject\s*(\d+)\s*$", flags=re.I)[0],
                                    errors="coerce")
        else:
            ids = pd.Series(np.arange(1, len(d) + 1), index=d.index)
        ref = pd.to_numeric(d[hr_col].astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
        ref = pd.Series(ref.to_numpy(), index=ids.to_numpy())
        ref = ref[ref.index.notna()]
        ref.index = ref.index.astype(int)
        ref = ref[~ref.index.duplicated(keep="first")]
        ours = trials.groupby("subject")["hr_mean"].mean()
        j = pd.concat([ours, ref], axis=1, keys=["neck", "finger"]).dropna()
        if len(j) >= 5:
            print(f"Heart-rate check vs finger oximeter (id column '{id_col}', HR column '{hr_col}'): "
                  f"n={len(j)} people, correlation = {np.corrcoef(j.neck, j.finger)[0, 1]:.2f}, "
                  f"mean difference (neck - finger) = {np.mean(j.neck - j.finger):+.1f} bpm")
        else:
            print(f"[info] Only {len(j)} people matched between neck and finger heart rate "
                  f"(id column '{id_col}', HR column '{hr_col}').")
    except Exception as exc:
        print(f"[info] Could not compare with Demographic_Data.csv ({exc}).")
        if d is not None:
            print("       Columns:", list(d.columns))
            print(d.head(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Neck-sensor breath-hold detection")
    ap.add_argument("source", type=Path, help="dataset .zip or extracted folder")
    ap.add_argument("--out-dir", type=Path, default=Path("apnoea_out"))
    ap.add_argument("--peeps", default="4,8",
                    help="PEEP levels to use (default 4,8: the levels that have BOTH normal and "
                         "breath-hold trials; 0 cmH2O has no breath-hold trials)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    peeps = {int(p) for p in args.peeps.split(",")}

    index, demo = index_source(args.source)
    keys = sorted(k for k in index if k[1] in peeps and {"inline", "neck"} <= set(index[k]))
    print(f"Found {len(index)} trials; using {len(keys)} with both sensors at PEEP {sorted(peeps)}.")

    rows = []
    for subj, peep, cond in keys:
        try:
            inl = inline_analysis(read_csv(index[(subj, peep, cond)]["inline"]))
            nk = neck_analysis(read_csv(index[(subj, peep, cond)]["neck"]))
        except Exception as exc:
            print(f"[skip] Subject{subj} {peep}cmH2O {cond}: {exc}")
            continue
        rows.append({"subject": subj, "peep": peep, "cond": cond, "hold": int(cond != "normal"),
                     **{k: v for k, v in inl.items() if k not in ("grid", "dp", "peak_times")},
                     **{k: v for k, v in nk.items() if k not in ("neck_series", "resp_ds")},
                     "series": nk["neck_series"], "resp_ds": nk["resp_ds"],
                     "inline_t": inl["grid"][::10], "inline_dp": inl["dp"][::10],
                     "peak_times": inl["peak_times"]})
    store = pd.DataFrame(rows)
    if len(store) < 20 or store["hold"].nunique() < 2:
        sys.exit("[error] Too few usable trials.")

    flat = store.drop(columns=["series", "resp_ds", "inline_t", "inline_dp", "peak_times"])
    flat.to_csv(args.out_dir / "apnoea_trial_features.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60 + "\nDATA CHECKS\n" + "=" * 60)
    print(f"People: {store['subject'].nunique()} | trials: {len(store)} | "
          f"normal: {(store['hold'] == 0).sum()} | breath-hold: {(store['hold'] == 1).sum()}")
    print("Longest gap between inhalations (s) by condition, from the airway sensor "
          "(expect: normal ~3-6, apnea1 ~12-14, apnea2 ~22-24):")
    print(store.groupby("cond")["inline_max_gap_s"].describe()[["count", "mean", "min", "max"]].round(1).to_string())
    acc = (store["inline_hold_detected"] == store["hold"]).mean()
    print(f"Airway-sensor rule (gap >= {HOLD_GAP_S:g} s) agrees with the trial label in {acc:.0%} of trials.")
    print(f"Neck pulse quality (spectral peak share, 1 = clean): median {store['pulse_snr'].median():.2f}; "
          f"HR extracted in {store['hr_mean'].notna().mean():.0%} of trials.")
    check_against_finger(demo, flat)

    y = store["hold"].to_numpy(int)
    groups = store["subject"].to_numpy()
    abl = ablation(store, y, groups)
    abl.to_csv(args.out_dir / "apnoea_ablation.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60 + "\nDETECTING A BREATH-HOLD (leave-one-subject-out)\n" + "=" * 60)
    print(abl.round(3).to_string(index=False))
    print("AUC 0.5 = chance. CI = bootstrap over people. AP_chance = share of breath-hold trials.")

    ablation_chart(abl, args.out_dir / "apnoea_ablation.png")
    event_chart(store, args.out_dir / "apnoea_event_locked.png")
    example_chart(store, args.out_dir / "apnoea_example.png")

    print("\n[caution] Healthy volunteers holding their breath on purpose, awake, with a lab sensor: "
          "this shows which signals react to a pause in breathing, NOT that the sensor detects sleep "
          "apnoea. The page also reports no oxygen desaturation in any participant, and the neck "
          "sensor is uncalibrated (the 660/940 nm channel order is assumed).")
    print(f"Files written to {args.out_dir}/")


if __name__ == "__main__":
    main()
