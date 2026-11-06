#!/usr/bin/env python3
"""
SHHS (PhysioNet shhpsgdb) pipeline: can wearable-type signals detect moments of
physiological INSTABILITY during sleep?

The PhysioNet copy contains ONE overnight recording (record 0000), so this is a
within-night, epoch-level (30 s) analysis of a single person. It shows which
signals react when the night is disturbed; it cannot show how well a model
generalises to new people.

Install (inside your venv):
    pip install wfdb pandas matplotlib          (scipy / scikit-learn already there)

Run (path to the downloaded .zip, the extracted folder, or the 0000.hea file):
    python shhs_pipeline.py path\\to\\shhpsgdb-1.0.0.zip
    python shhs_pipeline.py path\\to\\shhpsgdb --target resp_event --context

Per 30-second epoch the script extracts
    SpO2 (SaO2) : mean, min, std, drop below the recent baseline
    HR   (PR)   : mean, std, min, max, range, rise above the recent baseline
    HRV  (ECG)  : mean RR, SDNN, RMSSD   (R-peaks detected from the ECG)
    Resp        : airflow / chest / abdomen amplitude, relative amplitude,
                  respiratory rate, effort-vs-flow ratio
and uses the annotation files as ground truth (labels are never used as inputs):
    .hypn sleep stage | .arou EEG arousals | .resp apneas/hypopneas | .oart oximeter artifacts

Targets (--target):
    disturbed   epoch contains an arousal or a respiratory event (sleep epochs only)  [default]
    resp_event  epoch overlaps an apnea/hypopnea (sleep epochs only)
    arousal     epoch overlaps an EEG arousal (sleep epochs only)
    wake        epoch is scored Wake (all epochs)
"""

import argparse
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPOCH = 30          # seconds per scoring epoch
GAP = 4             # epochs removed on each side of a test block (limits leakage)
N_FOLDS = 5         # contiguous time blocks for cross-validation
SEED = 42
STAGE_LEVEL = {"W": 5, "R": 4, "1": 3, "2": 2, "3": 1, "4": 0}   # for plotting
SLEEP_STAGES = {"1", "2", "3", "4", "R"}
DUR_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------- #
# 1. Loading (the only part that needs the wfdb package)
# --------------------------------------------------------------------------- #
def resolve_record(path: Path, work_dir: Path) -> Path:
    """Return the record base path (no extension) from a zip, folder or .hea file."""
    if path.is_file() and path.suffix.lower() == ".zip":
        target = work_dir / "_shhs_extracted"
        if not target.exists():
            print(f"Extracting {path.name} to {target} ...")
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
        path = target
    if path.is_dir():
        heas = sorted(path.rglob("*.hea"))
        if not heas:
            sys.exit(f"[error] No .hea file found under {path}")
        return heas[0].with_suffix("")
    if path.suffix.lower() in {".hea", ".dat", ".edf"}:
        return path.with_suffix("")
    sys.exit(f"[error] Cannot find an SHHS record at {path}")


def read_annotation(wfdb, base: str, ext: str):
    try:
        a = wfdb.rdann(base, ext)
    except Exception as exc:
        print(f"[warn] could not read .{ext}: {exc}")
        return None
    notes = [str(n).strip("\x00 \t\r\n") for n in (a.aux_note or [""] * len(a.sample))]
    text = [n if n else str(s) for n, s in zip(notes, a.symbol)]
    return pd.DataFrame({"tick": np.asarray(a.sample, float), "text": text})


def load_shhs(record_base: Path):
    try:
        import wfdb
    except ImportError:
        sys.exit("[error] The 'wfdb' package is missing. Run:  pip install wfdb")
    base = str(record_base)
    rec = wfdb.rdrecord(base, smooth_frames=False)   # keep each channel's own sampling rate

    signals = {}
    if getattr(rec, "e_p_signal", None) is not None:
        spf = rec.samps_per_frame or [1] * len(rec.sig_name)
        for name, arr, k in zip(rec.sig_name, rec.e_p_signal, spf):
            signals[name.strip().upper()] = (np.asarray(arr, float), rec.fs * k)
    else:
        for i, name in enumerate(rec.sig_name):
            signals[name.strip().upper()] = (np.asarray(rec.p_signal[:, i], float), rec.fs)

    meta = {"record": record_base.name, "comments": " ".join(rec.comments or [])}

    ann = {ext: read_annotation(wfdb, base, ext) for ext in ("hypn", "arou", "resp", "oart")}
    # Annotation times are in 'ticks'. Infer ticks/second from the hypnogram (1 label per 30 s).
    hyp = ann["hypn"]
    if hyp is None or len(hyp) < 3:
        sys.exit("[error] .hypn (sleep stages) is required but could not be read.")
    tps = float(np.median(np.diff(hyp["tick"]))) / EPOCH
    meta["ticks_per_sec"] = tps
    return signals, ann, meta


def to_hypnogram(df: pd.DataFrame, tps: float) -> pd.Series:
    """Epoch index -> stage ('W','1','2','3','4','R'). A label applies to the FOLLOWING 30 s."""
    stage = df["text"].str.strip().str.upper().str[:1]
    idx = np.round(df["tick"].to_numpy() / tps / EPOCH).astype(int)
    s = pd.Series(stage.to_numpy(), index=idx)
    return s[~s.index.duplicated(keep="first")]


def to_events(df, tps: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["t0", "t1", "type"])
    t0 = df["tick"].to_numpy() / tps
    dur, typ = [], []
    for txt in df["text"]:
        m = DUR_RE.search(txt)
        dur.append(int(m.group(1)) * 60 + float(m.group(2)) if m else 10.0)
        parts = txt.split()
        typ.append(parts[0] if parts else "")
    return pd.DataFrame({"t0": t0, "t1": t0 + np.array(dur), "type": typ})


# --------------------------------------------------------------------------- #
# 2. Epoch features
# --------------------------------------------------------------------------- #
def get_channel(signals: dict, name: str):
    return signals.get(name.upper())


def epochs_of(x: np.ndarray, fs: float, n_ep: int) -> np.ndarray:
    length = int(round(fs * EPOCH))
    need = n_ep * length
    x = x[:need]
    if len(x) < need:
        x = np.concatenate([x, np.full(need - len(x), np.nan)])
    return x.reshape(n_ep, length)


def baseline(series: pd.Series, window: int = 5) -> pd.Series:
    """Median of the previous `window` epochs (excludes the current one)."""
    return series.shift(1).rolling(window, min_periods=2).median()


def spo2_features(sao2, fs, n_ep) -> pd.DataFrame:
    x = sao2.copy()
    x[(x < 50) | (x > 100)] = np.nan
    e = epochs_of(x, fs, n_ep)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = pd.DataFrame({"sao2_mean": np.nanmean(e, 1), "sao2_min": np.nanmin(e, 1),
                          "sao2_std": np.nanstd(e, 1)})
    f["sao2_drop"] = baseline(f["sao2_mean"]) - f["sao2_min"]
    return f


def hr_features(pr, fs, n_ep) -> pd.DataFrame:
    x = pr.copy()
    x[(x < 30) | (x > 200)] = np.nan
    e = epochs_of(x, fs, n_ep)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = pd.DataFrame({"pr_mean": np.nanmean(e, 1), "pr_std": np.nanstd(e, 1),
                          "pr_min": np.nanmin(e, 1), "pr_max": np.nanmax(e, 1)})
    f["pr_range"] = f["pr_max"] - f["pr_min"]
    f["pr_rise"] = f["pr_max"] - baseline(f["pr_mean"])
    return f


def hrv_features(ecg, fs, n_ep) -> pd.DataFrame:
    """R-peak detection (band-pass, squared, smoothed, adaptive threshold) -> RR statistics."""
    out = pd.DataFrame(np.nan, index=range(n_ep), columns=["rr_mean_ms", "sdnn_ms", "rmssd_ms"])
    x = np.nan_to_num(ecg - np.nanmedian(ecg))
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    env = filtfilt(b, a, x) ** 2
    w = max(int(0.12 * fs), 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")

    blk = int(10 * fs)
    nb = len(env) // blk
    if nb < 3:
        return out
    thr_blk = 0.4 * np.percentile(env[: nb * blk].reshape(nb, blk), 98, axis=1)
    thr = np.concatenate([np.repeat(thr_blk, blk), np.full(len(env) - nb * blk, thr_blk[-1])])
    peaks, _ = find_peaks(env, height=thr, distance=int(0.3 * fs))
    if len(peaks) < 50:
        return out

    t = peaks / fs
    rr, t_rr = np.diff(t), t[1:]
    ok = (rr > 0.3) & (rr < 2.0)
    med = pd.Series(np.where(ok, rr, np.nan)).rolling(5, center=True, min_periods=2).median().to_numpy()
    ok &= np.abs(rr - med) / med < 0.20
    ep = (t_rr // EPOCH).astype(int)
    df = pd.DataFrame({"ep": ep, "rr": rr})[ok & (ep < n_ep)]
    for e, g in df.groupby("ep"):
        r, idx = g["rr"].to_numpy(), g.index.to_numpy()
        if len(r) < 10:
            continue
        d = np.diff(r)[np.diff(idx) == 1]
        out.loc[e, "rr_mean_ms"] = r.mean() * 1000
        out.loc[e, "sdnn_ms"] = r.std() * 1000
        if len(d) >= 5:
            out.loc[e, "rmssd_ms"] = np.sqrt(np.mean(d ** 2)) * 1000
    return out


def resp_features(air, thor, abdo, fs, n_ep) -> pd.DataFrame:
    A, T, B = (epochs_of(v, fs, n_ep) for v in (air, thor, abdo))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = pd.DataFrame({"airflow_std": np.nanstd(A, 1), "thor_std": np.nanstd(T, 1),
                          "abdo_std": np.nanstd(B, 1)})
        for c in ("airflow", "thor"):
            f[f"{c}_rel"] = f[f"{c}_std"] / f[f"{c}_std"].shift(1).rolling(10, min_periods=3).median()
        f["effort_vs_flow"] = f["thor_rel"] / f["airflow_rel"].clip(lower=0.05)
        # respiratory rate: dominant frequency of the chest signal in 0.1-0.7 Hz
        Te = np.nan_to_num(T - np.nanmean(T, 1, keepdims=True))
        spec = np.abs(np.fft.rfft(Te * np.hanning(T.shape[1]), axis=1))
        freqs = np.fft.rfftfreq(T.shape[1], 1 / fs)
        band = (freqs >= 0.1) & (freqs <= 0.7)
        f["resp_rate_bpm"] = freqs[band][np.argmax(spec[:, band], 1)] * 60
    return f


def overlap_flags(events: pd.DataFrame, n_ep: int) -> np.ndarray:
    flag = np.zeros(n_ep, bool)
    for t0, t1 in zip(events["t0"], events["t1"]):
        a, b = int(max(t0, 0) // EPOCH), int(min(t1, n_ep * EPOCH - 1e-6) // EPOCH)
        if a < n_ep:
            flag[a: b + 1] = True
    return flag


GROUPS = {
    "SpO2": ["sao2_mean", "sao2_min", "sao2_std", "sao2_drop"],
    "HR": ["pr_mean", "pr_std", "pr_min", "pr_max", "pr_range", "pr_rise"],
    "HRV": ["rr_mean_ms", "sdnn_ms", "rmssd_ms"],
    "Resp": ["airflow_std", "airflow_rel", "thor_std", "thor_rel", "abdo_std",
             "effort_vs_flow", "resp_rate_bpm"],
}


def build_epoch_table(signals, ann, tps) -> tuple:
    sao2 = get_channel(signals, "SaO2")
    if sao2 is None:
        sys.exit("[error] SaO2 channel not found; channels: " + ", ".join(signals))
    n_ep = int(len(sao2[0]) / sao2[1] // EPOCH)

    hyp = to_hypnogram(ann["hypn"], tps)
    events = {k: to_events(ann[k], tps) for k in ("arou", "resp", "oart")}

    tab = pd.DataFrame({"epoch": np.arange(n_ep)})
    tab["hours"] = (tab["epoch"] * EPOCH) / 3600
    tab["stage"] = hyp.reindex(tab["epoch"]).to_numpy()
    tab["arousal"] = overlap_flags(events["arou"], n_ep)
    tab["resp_event"] = overlap_flags(events["resp"], n_ep)
    tab["artifact"] = overlap_flags(events["oart"], n_ep)

    feats = [spo2_features(sao2[0], sao2[1], n_ep)]
    spo2_cols = GROUPS["SpO2"]
    feats[0].loc[tab["artifact"].to_numpy(), spo2_cols] = np.nan   # unreliable oximeter

    for key, fn, chans in (("HR", hr_features, ["PR"]), ("HRV", hrv_features, ["ECG"])):
        ch = get_channel(signals, chans[0])
        if ch is None:
            print(f"[warn] channel {chans[0]} not found; {key} features skipped.")
        else:
            feats.append(fn(ch[0], ch[1], n_ep))
    chans = [get_channel(signals, n) for n in ("AIRFLOW", "THOR RES", "ABDO RES")]
    if all(c is not None for c in chans):
        feats.append(resp_features(chans[0][0], chans[1][0], chans[2][0], chans[0][1], n_ep))
    else:
        print("[warn] respiration channels not all found; Resp features skipped.")

    tab = pd.concat([tab] + feats, axis=1)
    tab["is_sleep"] = tab["stage"].isin(SLEEP_STAGES)
    tab["disturbed"] = tab["arousal"] | tab["resp_event"]
    tab["wake"] = tab["stage"].eq("W")
    return tab, events


# --------------------------------------------------------------------------- #
# 3. Models, blocked cross-validation, ablation, importance
# --------------------------------------------------------------------------- #
def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1))


def block_folds(ep_idx: np.ndarray):
    """Contiguous time blocks; epochs near the test block are removed from training."""
    edges = np.linspace(ep_idx.min(), ep_idx.max() + 1, N_FOLDS + 1).astype(int)
    for k in range(N_FOLDS):
        a, b = edges[k], edges[k + 1]
        test = (ep_idx >= a) & (ep_idx < b)
        train = (ep_idx < a - GAP) | (ep_idx >= b + GAP)
        yield train, test


def evaluate(X: np.ndarray, y: np.ndarray, ep_idx: np.ndarray, make_model) -> tuple:
    oof = np.full(len(y), np.nan)
    for tr, te in block_folds(ep_idx):
        if te.sum() == 0 or y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
            continue
        oof[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan, np.nan
    return roc_auc_score(y[ok], oof[ok]), average_precision_score(y[ok], oof[ok])


def add_context(tab: pd.DataFrame, cols: list) -> tuple:
    """Add previous/next-epoch copies of each feature (responses trail events by seconds)."""
    new = {}
    for c in cols:
        new[f"{c}__prev"] = tab[c].shift(1)
        new[f"{c}__next"] = tab[c].shift(-1)
    return pd.concat([tab, pd.DataFrame(new)], axis=1), new


def ablation(sub, y, ep_idx, groups: dict, context_cols: dict) -> pd.DataFrame:
    def cols(names):
        out = []
        for g in names:
            for c in groups[g]:
                out.append(c)
                out += [k for k in (f"{c}__prev", f"{c}__next") if k in context_cols]
        return out

    present = [g for g in groups if groups[g]]
    sets = {g: [g] for g in present}
    if {"HR", "HRV"} <= set(present):
        sets["HR + HRV"] = ["HR", "HRV"]
    if {"HR", "SpO2"} <= set(present):
        sets["HR + SpO2"] = ["HR", "SpO2"]
    wearable = [g for g in ("HR", "HRV", "SpO2") if g in present]
    if len(wearable) == 3:
        sets["HR + HRV + SpO2"] = wearable
    if len(present) > 1:
        sets["All signals"] = present

    rows = []
    prevalence = float(y.mean())
    for name, gs in sets.items():
        X = sub[cols(gs)].to_numpy(float)
        for mname, maker in (("Logistic", make_lr), ("RandomForest", make_rf)):
            auc, ap = evaluate(X, y, ep_idx, maker)
            rows.append({"signals": name, "n_features": X.shape[1], "model": mname,
                         "AUC": auc, "AP": ap, "AP_chance": prevalence})
    return pd.DataFrame(rows)


def importance(sub, y, ep_idx, feature_cols) -> pd.Series:
    X = sub[feature_cols].to_numpy(float)
    acc, used = np.zeros(len(feature_cols)), 0
    for tr, te in block_folds(ep_idx):
        if te.sum() == 0 or y[tr].sum() < 3 or (1 - y[tr]).sum() < 3 or len(np.unique(y[te])) < 2:
            continue
        m = make_rf().fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=5,
                                   random_state=SEED)
        acc += r.importances_mean
        used += 1
    return pd.Series(acc / max(used, 1), index=feature_cols).sort_values(ascending=False)


def group_importance(sub, y, ep_idx, groups: dict, ctx: dict) -> pd.Series:
    """AUC lost when a WHOLE sensor group is shuffled (robust to redundant features)."""
    cols_of = {g: [c for c in cs] + [k for k in ctx if k.split("__")[0] in cs]
               for g, cs in groups.items() if cs}
    all_cols = [c for cs in cols_of.values() for c in cs]
    pos = {c: i for i, c in enumerate(all_cols)}
    X = sub[all_cols].to_numpy(float)
    rng = np.random.default_rng(SEED)
    acc, used = {g: 0.0 for g in cols_of}, 0
    for tr, te in block_folds(ep_idx):
        if te.sum() == 0 or y[tr].sum() < 3 or (1 - y[tr]).sum() < 3 or len(np.unique(y[te])) < 2:
            continue
        model = make_rf().fit(X[tr], y[tr])
        base = roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])
        for g, cs in cols_of.items():
            idx = [pos[c] for c in cs]
            drops = []
            for _ in range(5):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(te.sum())][:, idx]
                drops.append(base - roc_auc_score(y[te], model.predict_proba(Xp)[:, 1]))
            acc[g] += float(np.mean(drops))
        used += 1
    return pd.Series({g: v / max(used, 1) for g, v in acc.items()}).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 4. Charts
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


def timeline_chart(tab, events, path: Path) -> None:
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(5, 1, figsize=(13, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 0.8, 1, 1, 1]})
    lvl = tab["stage"].map(STAGE_LEVEL)
    ax[0].step(tab["hours"], lvl, where="post", color="navy", lw=1)
    ax[0].set_yticks(list(STAGE_LEVEL.values()), list(STAGE_LEVEL.keys()))
    ax[0].set_ylabel("Sleep stage")
    ax[0].set_title("One night: sleep stages, events, heart rate, oxygen, airflow")
    for row, (key, color, label) in enumerate((("resp", "firebrick", "Resp. events"),
                                              ("arou", "darkorange", "Arousals"))):
        ev = events[key]
        ax[1].eventplot(ev["t0"] / 3600, lineoffsets=row, linelengths=0.8, colors=color)
    ax[1].set_yticks([0, 1], ["Resp. events", "Arousals"])
    if "pr_mean" in tab:
        ax[2].plot(tab["hours"], tab["pr_mean"], color="crimson", lw=0.9)
    ax[2].set_ylabel("Heart rate\n(bpm)")
    ax[3].plot(tab["hours"], tab["sao2_min"], color="steelblue", lw=0.9)
    ax[3].set_ylabel("SpO2 min\n(%)")
    if "airflow_std" in tab:
        ax[4].plot(tab["hours"], tab["airflow_std"], color="seagreen", lw=0.9)
    ax[4].set_ylabel("Airflow\namplitude")
    ax[4].set_xlabel("Hours since start of recording")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def zoom_chart(signals, events, path: Path, minutes: int = 20) -> None:
    """Raw signals for the stretch of the night with the most respiratory events."""
    plt = get_plt()
    resp = events["resp"]
    if plt is None or resp.empty:
        return
    span = minutes * 60
    starts = np.arange(0, max(int(resp["t0"].max()) - span, 1), 60)
    counts = [((resp["t0"] >= s) & (resp["t0"] < s + span)).sum() for s in starts]
    t0 = float(starts[int(np.argmax(counts))]) if len(starts) else 0.0

    chans = [("AIRFLOW", "Airflow", "seagreen"), ("THOR RES", "Chest effort", "teal"),
             ("SAO2", "SpO2 (%)", "steelblue"), ("PR", "Heart rate (bpm)", "crimson")]
    chans = [c for c in chans if get_channel(signals, c[0]) is not None]
    fig, ax = plt.subplots(len(chans), 1, figsize=(13, 2.2 * len(chans)), sharex=True)
    ax = np.atleast_1d(ax)
    for a, (name, label, color) in zip(ax, chans):
        x, fs = get_channel(signals, name)
        i0, i1 = int(t0 * fs), int((t0 + span) * fs)
        a.plot(np.arange(i0, min(i1, len(x))) / fs / 60, x[i0:i1], color=color, lw=0.7)
        a.set_ylabel(label)
        for _, e in resp.iterrows():
            if t0 <= e["t0"] <= t0 + span:
                a.axvspan(e["t0"] / 60, e["t1"] / 60, color="firebrick", alpha=0.15)
        for _, e in events["arou"].iterrows():
            if t0 <= e["t0"] <= t0 + span:
                a.axvspan(e["t0"] / 60, e["t1"] / 60, color="orange", alpha=0.18)
        a.grid(alpha=0.25)
    ax[0].set_title(f"{minutes} busiest minutes (red = respiratory event, orange = arousal)")
    ax[-1].set_xlabel("Minutes since start of recording")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def ablation_chart(abl: pd.DataFrame, target: str, path: Path) -> None:
    plt = get_plt()
    if plt is None or abl.empty:
        return
    piv = abl.pivot(index="signals", columns="model", values="AUC")
    piv = piv.loc[[s for s in abl["signals"].unique()]]
    ax = piv.plot.bar(figsize=(11, 5), color=["steelblue", "darkorange"], zorder=3)
    ax.axhline(0.5, color="firebrick", ls="--", lw=1.5, label="Chance (0.5)")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC (blocked cross-validation)")
    ax.set_title(f"Which signals detect '{target}' epochs?")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(signals, ann, meta, out_dir: Path, target: str, context: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tab, events = build_epoch_table(signals, ann, meta["ticks_per_sec"])

    print("=" * 60 + "\nRECORD SUMMARY\n" + "=" * 60)
    print(f"Record {meta['record']} | {meta['comments']}")
    print(f"Ticks per second inferred from the hypnogram: {meta['ticks_per_sec']:g}")
    print(f"Epochs (30 s): {len(tab)} = {len(tab) * EPOCH / 3600:.1f} h | "
          f"scored epochs: {int(tab['stage'].notna().sum())}")
    print("Stage counts:", tab["stage"].value_counts().to_dict())
    sleep_h = tab["is_sleep"].sum() * EPOCH / 3600
    print(f"Sleep time: {sleep_h:.1f} h | annotated respiratory events: {len(events['resp'])} "
          f"({len(events['resp']) / max(sleep_h, 1e-9):.1f}/sleep-hour) | arousals: {len(events['arou'])}")
    if not events["resp"].empty:
        print("Respiratory event types:", events["resp"]["type"].value_counts().to_dict())

    groups = {g: [c for c in cols if c in tab.columns] for g, cols in GROUPS.items()}
    base_cols = [c for cols in groups.values() for c in cols]
    ctx = {}
    if context:
        tab, ctx = add_context(tab, base_cols)

    sub = tab[tab["stage"].notna()] if target == "wake" else tab[tab["is_sleep"]]
    sub = sub.reset_index(drop=True)
    y = sub[target].to_numpy(int)
    ep_idx = sub["epoch"].to_numpy()
    print(f"\nTarget '{target}': {int(y.sum())} of {len(y)} epochs positive "
          f"({y.mean():.0%})")
    if y.sum() < 15 or (1 - y).sum() < 15:
        sys.exit("[error] Too few positive or negative epochs for this target.")

    tab.to_csv(out_dir / "shhs_epoch_features.csv", index=False, encoding="utf-8-sig")

    abl = ablation(sub, y, ep_idx, groups, ctx)
    abl.to_csv(out_dir / "shhs_ablation.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60 + f"\nABLATION: detecting '{target}' (blocked CV, {N_FOLDS} time blocks)\n" + "=" * 60)
    print(abl.round(3).to_string(index=False))
    print("AUC 0.5 = chance. AP_chance = share of positive epochs (the AP you get by guessing).")

    all_cols = [c for g in groups.values() for c in g]
    all_cols += [k for k in ctx if k.split("__")[0] in all_cols]
    gimp = group_importance(sub, y, ep_idx, groups, ctx)
    gimp.to_csv(out_dir / "shhs_group_importance.csv", header=["auc_drop"], encoding="utf-8-sig")
    print("\nSensor-group importance (AUC lost when the whole group is shuffled; Random Forest):")
    print(gimp.round(3).to_string())
    imp = importance(sub, y, ep_idx, all_cols)
    imp.to_csv(out_dir / "shhs_importance.csv", header=["perm_importance_auc_drop"], encoding="utf-8-sig")
    print("\nTop single features (often small because features are redundant; prefer the group view):")
    print(imp.head(10).round(4).to_string())

    timeline_chart(tab, events, out_dir / "shhs_timeline.png")
    zoom_chart(signals, events, out_dir / "shhs_zoom.png")
    ablation_chart(abl, target, out_dir / "shhs_ablation.png")

    print("\n[caution] One person, one night: adjacent epochs are strongly correlated and the "
          "result says nothing about other people. Use it to show WHICH signals react to "
          "disturbances, not to claim a generalisable detector.")
    print(f"Files written to {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="SHHS epoch-level instability analysis")
    ap.add_argument("source", type=Path, help="SHHS .zip, extracted folder, or 0000.hea")
    ap.add_argument("--out-dir", type=Path, default=Path("shhs_out"))
    ap.add_argument("--target", default="disturbed",
                    choices=["disturbed", "resp_event", "arousal", "wake"])
    ap.add_argument("--context", action="store_true",
                    help="also use the previous and next epoch's features")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = resolve_record(args.source, args.out_dir)
    print(f"Reading record {base} ...")
    signals, ann, meta = load_shhs(base)
    print("Channels:", ", ".join(f"{k} ({v[1]:g} Hz)" for k, v in signals.items()))
    for ext, df in ann.items():
        print(f"  .{ext}: {0 if df is None else len(df)} annotations"
              + ("" if df is None or df.empty else f" | first text: {df['text'].iloc[0]!r}"))
    run_analysis(signals, {k: v for k, v in ann.items()}, meta, args.out_dir, args.target, args.context)


if __name__ == "__main__":
    main()
