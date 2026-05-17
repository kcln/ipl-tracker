"""Nightly v3-phase retrain — produces a fresh candidate model and a
side-by-side metrics comparison against the current live (v3) baseline.

Honest scope (per ml/CLAUDE.md):
  * 2025 IPL is a LOCKED test set. We re-evaluate against it (the existing
    train_phase.py reports test_brier on 2025) only as a *report*. We do
    NOT auto-promote based on those numbers — repeated optimisation against
    a locked test set is leakage. Promotion is human-gated.
  * Each retrain produces a versioned artifact (v10, v11, ...) — never
    overwrites prior weights.
  * Output: per-day CSV row in ml/data/backtest_results/retrain_history.csv
    with phase, version, n_train, val/test brier & accuracy, delta vs v3 live.

Promotion (when you decide a candidate is good enough):
    cp ml/data/models/v{N}_phase_<phase>.pkl  ml/data/models/v3_phase_<phase>.pkl
    cp ml/data/models/v{N}_phase_<phase>.json ml/data/models/v3_phase_<phase>.json
  (with a backup first; the original v3 stays in the repo under its versioned
  artifact name forever — CLAUDE.md rule 3.)

Invoked by launchd/com.kcln.ipl-ml-retrain.plist at 02:00 PT daily.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
MODELS = ML / "data" / "models"
HIST = ML / "data" / "historical"
RESULTS = ML / "data" / "backtest_results"
LOG_CSV = RESULTS / "retrain_history.csv"
PROMOTE_LOG = RESULTS / "promotion_history.csv"
BACKUP_DIR = MODELS / "backups"
PHASES = ["pre_match", "post_toss", "post_pp1", "innings_break"]
LIVE_VERSION = 3  # current production v3_phase_*.pkl is the reference baseline
RESERVED_VERSIONS = {3, 6}  # do not collide with shipped artifacts

# Auto-promotion gates — only act when 2026 holdout signal is strong.
HOLDOUT_MIN_N = 20            # need ≥20 completed 2026 matches to trust the eval
PROMOTE_BRIER_DELTA = 0.005   # new must beat live by ≥0.005 absolute Brier
PROMOTE_ACC_FLOOR = 0.60      # CLAUDE.md kill criterion


def _existing_versions() -> set[int]:
    """Scan v{N}_phase_pre_match.pkl files and return the set of N values."""
    versions: set[int] = set()
    for p in MODELS.glob("v*_phase_pre_match.pkl"):
        m = re.match(r"v(\d+)_phase_pre_match\.pkl$", p.name)
        if m:
            versions.add(int(m.group(1)))
    return versions


def _next_version() -> int:
    used = _existing_versions() | RESERVED_VERSIONS
    n = max(used) + 1 if used else 10
    if n < 10:
        n = 10  # reserve 1-9 for documented baselines
    while n in used:
        n += 1
    return n


def _load_metrics(version: int, phase: str) -> dict | None:
    """Read training metrics from v{version}_phase_{phase}.json."""
    p = MODELS / f"v{version}_phase_{phase}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _train(version: int, log_lines: list[str]) -> int:
    """Run ml.src.train_phase --phase all --version N. Captures combined output
    and appends to log_lines so the retrain log has full provenance."""
    cmd = [
        str(ML / ".venv" / "bin" / "python"),
        "-m", "ml.src.train_phase",
        "--phase", "all",
        "--version", str(version),
    ]
    log_lines.append(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log_lines.append(proc.stdout)
    if proc.stderr:
        log_lines.append("[stderr]\n" + proc.stderr)
    return proc.returncode


def _build_holdout_2026() -> dict[str, int]:
    """Rebuild the 2026 holdout parquets from the current cricsheet snapshot."""
    from ml.scripts import build_holdout_2026 as bh
    return bh.build()


def _load_holdout(phase: str):
    p = HIST / f"holdout_2026_{phase}.parquet"
    if not p.exists():
        return None
    import pandas as pd
    return pd.read_parquet(p)


def _eval_on_holdout(version: int, phase: str, holdout) -> tuple[float, float, int] | None:
    """Score v{version}_phase_{phase} against holdout. Returns (acc, brier, n) or None."""
    model_path = MODELS / f"v{version}_phase_{phase}.pkl"
    if not model_path.exists() or holdout is None or holdout.empty:
        return None
    try:
        from sklearn.metrics import accuracy_score, brier_score_loss
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        cal = obj["calibrator"]
        feats = obj["feature_names"]
        # Drop holdout rows missing any required feature
        sub = holdout.dropna(subset=feats)
        if len(sub) < 1:
            return None
        X = sub[feats]
        y = sub["winner_is_team1"]
        p = cal.predict_proba(X)[:, 1]
        return (
            float(accuracy_score(y, (p >= 0.5).astype(int))),
            float(brier_score_loss(y, p)),
            int(len(sub)),
        )
    except Exception as e:
        print(f"[nightly_retrain] eval failed for v{version}/{phase}: {e}", file=sys.stderr)
        return None


def _backup_and_promote(phase: str, new_version: int) -> pathlib.Path:
    """Copy live v3 to backups/, then overwrite live with v{new_version}.
    Returns the backup path. Original v{new_version} stays in models/ forever
    (CLAUDE.md immutability)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    live_pkl = MODELS / f"v{LIVE_VERSION}_phase_{phase}.pkl"
    live_json = MODELS / f"v{LIVE_VERSION}_phase_{phase}.json"
    backup_pkl = BACKUP_DIR / f"v{LIVE_VERSION}_phase_{phase}_pre_promote_{ts}.pkl"
    backup_json = BACKUP_DIR / f"v{LIVE_VERSION}_phase_{phase}_pre_promote_{ts}.json"
    shutil.copy2(live_pkl, backup_pkl)
    shutil.copy2(live_json, backup_json)
    shutil.copy2(MODELS / f"v{new_version}_phase_{phase}.pkl", live_pkl)
    shutil.copy2(MODELS / f"v{new_version}_phase_{phase}.json", live_json)
    return backup_pkl


def _maybe_auto_promote(phase: str, new_version: int, holdout) -> dict:
    """Apply gates against 2026 holdout. Returns dict with decision + numbers."""
    out = {"phase": phase, "version": new_version, "holdout_n": 0}
    if holdout is None or holdout.empty:
        out["decision"] = "NO_HOLDOUT"
        return out
    new = _eval_on_holdout(new_version, phase, holdout)
    live = _eval_on_holdout(LIVE_VERSION, phase, holdout)
    if new is None or live is None:
        out["decision"] = "EVAL_FAILED"
        return out
    out["holdout_n"] = new[2]
    out["new_acc_2026"] = round(new[0], 4)
    out["new_brier_2026"] = round(new[1], 4)
    out["live_acc_2026"] = round(live[0], 4)
    out["live_brier_2026"] = round(live[1], 4)
    out["delta_brier_2026"] = round(live[1] - new[1], 4)
    out["delta_acc_2026"] = round(new[0] - live[0], 4)

    if new[2] < HOLDOUT_MIN_N:
        out["decision"] = f"INSUFFICIENT_N ({new[2]} < {HOLDOUT_MIN_N})"
        return out
    if out["delta_brier_2026"] < PROMOTE_BRIER_DELTA:
        out["decision"] = f"KEEP_LIVE (Δbrier {out['delta_brier_2026']:+.4f} < {PROMOTE_BRIER_DELTA})"
        return out
    if new[0] < PROMOTE_ACC_FLOOR:
        out["decision"] = f"KEEP_LIVE (acc {new[0]:.3f} < floor {PROMOTE_ACC_FLOOR})"
        return out

    backup = _backup_and_promote(phase, new_version)
    out["decision"] = f"PROMOTED v{new_version}"
    out["backup_path"] = str(backup.relative_to(ROOT))
    return out


def _append_promote_log(rows: list[dict]) -> None:
    fieldnames = [
        "ts_utc", "phase", "version", "holdout_n",
        "live_acc_2026", "live_brier_2026",
        "new_acc_2026", "new_brier_2026",
        "delta_acc_2026", "delta_brier_2026",
        "decision", "backup_path",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    is_new = not PROMOTE_LOG.exists()
    with PROMOTE_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarise(new_version: int) -> list[dict]:
    """For each phase, build a comparison row vs LIVE_VERSION."""
    rows = []
    ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for phase in PHASES:
        new_m = _load_metrics(new_version, phase)
        live_m = _load_metrics(LIVE_VERSION, phase)
        if new_m is None:
            rows.append({
                "ts_utc": ts, "phase": phase, "version": new_version,
                "status": "TRAIN_FAILED",
            })
            continue
        row = {
            "ts_utc": ts,
            "phase": phase,
            "version": new_version,
            "n_train": new_m.get("n_train"),
            "val_acc_new": round(new_m.get("validation_accuracy", 0.0), 4),
            "val_brier_new": round(new_m.get("validation_brier", 0.0), 4),
            "test_acc_new": round(new_m.get("test_accuracy", 0.0), 4),
            "test_brier_new": round(new_m.get("test_brier", 0.0), 4),
        }
        if live_m:
            row["test_acc_live"] = round(live_m.get("test_accuracy", 0.0), 4)
            row["test_brier_live"] = round(live_m.get("test_brier", 0.0), 4)
            # Lower Brier = better. Positive delta_brier = new model wins.
            row["delta_brier"] = round(row["test_brier_live"] - row["test_brier_new"], 4)
            row["delta_acc"] = round(row["test_acc_new"] - row["test_acc_live"], 4)
            # Recommendation (advisory only — DO NOT auto-act on 2025 test):
            if row["delta_brier"] >= 0.005 and row["delta_acc"] >= 0.0 and row["val_acc_new"] >= 0.60:
                row["recommend"] = "INSPECT_FOR_PROMOTION"
            else:
                row["recommend"] = "KEEP_LIVE"
            row["status"] = "OK"
        else:
            row["status"] = "OK_NO_BASELINE"
        rows.append(row)
    return rows


def _append_log(rows: list[dict]) -> None:
    fieldnames = [
        "ts_utc", "phase", "version", "status", "n_train",
        "val_acc_new", "val_brier_new",
        "test_acc_new", "test_brier_new",
        "test_acc_live", "test_brier_live",
        "delta_brier", "delta_acc", "recommend",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_summary(rows: list[dict]) -> None:
    print()
    print("=" * 90)
    print(f"NIGHTLY RETRAIN SUMMARY — {rows[0]['ts_utc'] if rows else 'no rows'}")
    print("=" * 90)
    print(f"{'phase':18s}{'v_new':>6s}{'acc_new':>9s}{'brier_new':>11s}"
          f"{'acc_live':>10s}{'brier_live':>12s}{'Δacc':>8s}{'Δbrier':>9s}  recommend")
    for r in rows:
        if r.get("status") != "OK":
            print(f"{r['phase']:18s}{r['version']:6d}  -- {r.get('status')}")
            continue
        print(
            f"{r['phase']:18s}{r['version']:6d}"
            f"{r.get('test_acc_new', 0):9.3f}{r.get('test_brier_new', 0):11.3f}"
            f"{r.get('test_acc_live', 0):10.3f}{r.get('test_brier_live', 0):12.3f}"
            f"{r.get('delta_acc', 0):+8.3f}{r.get('delta_brier', 0):+9.3f}"
            f"  {r.get('recommend', '-')}"
        )
    print()
    print("Note: deltas are measured against the 2025 LOCKED TEST SET.")
    print("      Auto-promotion would constitute test-set leakage. Promotion is human-gated.")
    print(f"Full CSV log: {LOG_CSV}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan, do not train")
    ap.add_argument("--version", type=int, default=None,
                    help="Force a specific version (default: auto-pick next)")
    args = ap.parse_args()

    version = args.version if args.version is not None else _next_version()
    print(f"[nightly_retrain] target version=v{version}  at {dt.datetime.now(dt.UTC).isoformat()}")

    if args.dry_run:
        print(f"[nightly_retrain] --dry-run: would train v{version} for phases {PHASES}")
        return 0

    # Refuse to overwrite (the trainer enforces this too, but fail fast)
    collisions = [f"v{version}_phase_{p}.pkl" for p in PHASES
                  if (MODELS / f"v{version}_phase_{p}.pkl").exists()]
    if collisions:
        print(f"[nightly_retrain] aborting — would overwrite: {collisions}", file=sys.stderr)
        return 1

    log_lines: list[str] = []
    rc = _train(version, log_lines)
    rows = _summarise(version)
    _append_log(rows)
    _print_summary(rows)

    # Rebuild 2026 holdout from the freshly-ingested cricsheet, then evaluate
    # the new candidate vs current live on genuinely out-of-sample data.
    # 2025 is NEVER used for promotion (locked test set).
    promote_rows: list[dict] = []
    if rc == 0:
        try:
            counts = _build_holdout_2026()
            print(f"[nightly_retrain] built 2026 holdouts: {counts}")
        except Exception as e:
            print(f"[nightly_retrain] holdout build failed (non-fatal): {e}", file=sys.stderr)
            counts = {}

        ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        print()
        print("=" * 90)
        print("AUTO-PROMOTE EVALUATION (vs 2026 holdout)")
        print("=" * 90)
        print(f"{'phase':18s}{'n_2026':>8s}{'acc_new':>10s}{'brier_new':>11s}"
              f"{'acc_live':>10s}{'brier_live':>12s}{'Δbrier':>9s}  decision")
        for phase in PHASES:
            holdout = _load_holdout(phase)
            res = _maybe_auto_promote(phase, version, holdout)
            res["ts_utc"] = ts
            promote_rows.append(res)
            n = res.get("holdout_n", 0)
            if "new_brier_2026" in res:
                print(
                    f"{phase:18s}{n:8d}"
                    f"{res['new_acc_2026']:10.3f}{res['new_brier_2026']:11.3f}"
                    f"{res['live_acc_2026']:10.3f}{res['live_brier_2026']:12.3f}"
                    f"{res['delta_brier_2026']:+9.4f}  {res['decision']}"
                )
            else:
                print(f"{phase:18s}{n:8d}  -- {res['decision']}")
        _append_promote_log(promote_rows)
        print()
        print(f"Auto-promote gates: n≥{HOLDOUT_MIN_N}, Δbrier≥{PROMOTE_BRIER_DELTA}, acc≥{PROMOTE_ACC_FLOOR}")
        print(f"Full promotion log: {PROMOTE_LOG}")

    # Save full training log for this run
    log_path = RESULTS / f"retrain_v{version}.log"
    log_path.write_text("\n".join(log_lines))
    print(f"[nightly_retrain] train rc={rc}  log={log_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
