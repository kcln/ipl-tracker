"""v6 Phase D4 — one-shot retraining script.

Orchestrates a full retrain of v6_* base models and v7_ensemble against the
current parquet snapshot. Produces immutable artifacts suffixed with a
caller-supplied version tag (typically YYYYMMDD) so retrains never overwrite
prior weights.

This script is intentionally a STUB / scaffold while Phase A/B/C are still
landing. The orchestration sequence and CLI shape are stable; the per-step
implementation is filled in once each base model exists.

CLI:
    ml/.venv/bin/python -m ml.src.v6.retrain --version-suffix 20260512
    ml/.venv/bin/python -m ml.src.v6.retrain --version-suffix 20260512 --dry-run
    ml/.venv/bin/python -m ml.src.v6.retrain --version-suffix 20260512 \
        --skip phase_a --skip phase_b

Output (all under ml/data/models/, never overwriting):
    v6_phase_pre_match_{suffix}.pkl + .json
    v6_phase_post_toss_{suffix}.pkl + .json
    v6_phase_post_pp1_{suffix}.pkl + .json
    v6_phase_innings_break_{suffix}.pkl + .json
    v6_player_{suffix}.pkl + .json
    v7_ensemble_{suffix}.pkl + .json
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
ML = ROOT / "ml"
MODELS = ML / "data" / "models"
HIST = ML / "data" / "historical"
V6_HIST = HIST / "v6"


# Pipeline steps. Each step is (name, description, callable, depends_on).
# When `callable` is None it's still a planning placeholder (TODO).
# `depends_on` documents which Phase parquets / models must exist first.

def _step_phase_a_features(suffix: str, dry_run: bool) -> None:
    """Build Phase A derived features (A1-A5) from the current cricsheet snapshot.

    TODO(Phase A): wire to ml.src.v6.build_a1..a5 modules once they all land.
    Each one writes a parquet under ml/data/historical/v6/.
    """
    print(f"[retrain] phase_a: build A1-A5 feature parquets under {V6_HIST}/",
          flush=True)
    if dry_run:
        return
    # TODO: from ml.src.v6 import build_a1, build_a2, build_a3, build_a4, build_a5
    # for mod in (build_a1, build_a2, build_a3, build_a4, build_a5):
    #     mod.main()
    raise NotImplementedError(
        "Phase A builders not yet wired into retrain; use --dry-run or --skip phase_a"
    )


def _step_phase_b_features(suffix: str, dry_run: bool) -> None:
    """Build Phase B external data (weather, playing XI).

    TODO(Phase B): call ml.src.v6.build_b1 (weather) and build_b2 (playing XI).
    Weather requires open-meteo HTTP — be mindful of rate limits.
    """
    print(f"[retrain] phase_b: weather + playing_xi parquets under {V6_HIST}/",
          flush=True)
    if dry_run:
        return
    # TODO: from ml.src.v6 import build_b1, build_b2
    # build_b1.main(); build_b2.main()
    raise NotImplementedError("Phase B builders not yet wired into retrain")


def _step_train_v6_phase_models(suffix: str, dry_run: bool) -> None:
    """Train v6_phase_{pre_match, post_toss, post_pp1, innings_break} with the
    expanded feature set (Phase A + Phase B). Mirror ml.src.train_phase but
    append `_{suffix}` to artifact names so we never overwrite prior weights.

    TODO(Phase C.1.1-1.4): once Phase A/B parquets are stable, fork
    ml/src/train_phase.py into ml/src/v6/train_v6_phase.py that:
      - Joins A1-A5 + B1 features onto features_pre_match.parquet
      - Trains LightGBM + isotonic calibration per phase
      - Writes ml/data/models/v6_phase_<phase>_<suffix>.pkl + .json
    """
    out = [MODELS / f"v6_phase_{p}_{suffix}.pkl"
           for p in ("pre_match", "post_toss", "post_pp1", "innings_break")]
    print(f"[retrain] phase_c1: train v6_phase_* -> "
          + ", ".join(str(p.relative_to(ROOT)) for p in out), flush=True)
    if dry_run:
        return
    raise NotImplementedError("v6 phase models trainer not yet wired into retrain")


def _step_train_v6_player(suffix: str, dry_run: bool) -> None:
    """Train v6_player with A1 phase-specific player aggregates.

    TODO(Phase C.1.5): fork ml.src.train_player into ml/src/v6/train_v6_player.py
    that consumes A1 (phase_player_stats.parquet) and writes
    v6_player_{suffix}.pkl + .json.
    """
    out = MODELS / f"v6_player_{suffix}.pkl"
    print(f"[retrain] phase_c1.5: train v6_player -> {out.relative_to(ROOT)}",
          flush=True)
    if dry_run:
        return
    raise NotImplementedError("v6 player model trainer not yet wired into retrain")


def _step_train_v7_ensemble(suffix: str, dry_run: bool) -> None:
    """Train v7_ensemble — LightGBM meta-learner stacking v6_phase_pre_match +
    v6_player (optionally v3_phase_pre_match as a stable reference).

    TODO(Phase C.2): fork ml.src.train_ensemble.py into
    ml/src/v6/train_v7_ensemble.py that:
      - Loads v6_phase_pre_match_{suffix} and v6_player_{suffix}
      - Generates out-of-fold predictions on the 2008-2023 train window
        (TimeSeriesSplit) to avoid leakage into the meta-learner
      - Trains a LightGBM classifier on those OOF predictions
      - Writes v7_ensemble_{suffix}.pkl + .json
    """
    out = MODELS / f"v7_ensemble_{suffix}.pkl"
    print(f"[retrain] phase_c2: train v7_ensemble -> {out.relative_to(ROOT)}",
          flush=True)
    if dry_run:
        return
    raise NotImplementedError("v7 ensemble trainer not yet wired into retrain")


def _step_drift_check(suffix: str, dry_run: bool) -> None:
    """After retraining, run the drift monitor against the new v7 ensemble.

    Hooks into ml.src.v6.drift_monitor which already supports --ensemble.
    """
    print(f"[retrain] post: drift check against v7_ensemble_{suffix}", flush=True)
    if dry_run:
        return
    # Lazy import: avoid loading heavy deps when --dry-run.
    from ml.src.v6 import drift_monitor
    scored = drift_monitor.score_season(2026, f"v7_ensemble_{suffix}")
    report = drift_monitor.build_report(scored, baseline=0.549)
    print(drift_monitor.format_report(report, 2026, f"v7_ensemble_{suffix}"))


STEPS = [
    ("phase_a", _step_phase_a_features,
     "build Phase A derived feature parquets (A1-A5)"),
    ("phase_b", _step_phase_b_features,
     "build Phase B external feature parquets (weather, playing XI)"),
    ("train_v6_phase", _step_train_v6_phase_models,
     "train v6_phase_* (pre_match, post_toss, post_pp1, innings_break)"),
    ("train_v6_player", _step_train_v6_player,
     "train v6_player with phase-specific player aggregates"),
    ("train_v7_ensemble", _step_train_v7_ensemble,
     "train v7_ensemble LightGBM meta-learner"),
    ("drift_check", _step_drift_check,
     "run drift_monitor against the freshly trained v7"),
]


def _validate_suffix(suffix: str) -> None:
    if not suffix:
        raise SystemExit("--version-suffix is required and non-empty")
    if not suffix.replace("_", "").replace("-", "").isalnum():
        raise SystemExit(f"--version-suffix must be alphanumeric (got: {suffix!r})")


def _existing_artifacts(suffix: str) -> list[pathlib.Path]:
    """Return any files that already use this suffix — we never overwrite."""
    return sorted(MODELS.glob(f"*_{suffix}.pkl")) + sorted(MODELS.glob(f"*_{suffix}.json"))


def main():
    ap = argparse.ArgumentParser(description="v6/v7 one-shot retraining (D4)")
    ap.add_argument("--version-suffix", required=True,
                    help="Suffix appended to every artifact (typically YYYYMMDD). "
                         "Refuses to run if files with this suffix already exist.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned pipeline without executing it.")
    ap.add_argument("--skip", action="append", default=[],
                    choices=[name for name, _, _ in STEPS],
                    help="Skip a specific step (repeatable).")
    ap.add_argument("--only", action="append", default=[],
                    choices=[name for name, _, _ in STEPS],
                    help="Run ONLY these steps (repeatable). Overrides --skip.")
    args = ap.parse_args()

    _validate_suffix(args.version_suffix)

    existing = _existing_artifacts(args.version_suffix) if not args.dry_run else []
    if existing:
        print(f"[retrain] refusing to overwrite existing artifacts with "
              f"suffix {args.version_suffix!r}:")
        for p in existing:
            print(f"[retrain]   {p.relative_to(ROOT)}")
        return 1

    print(f"[retrain] start suffix={args.version_suffix} "
          f"dry_run={args.dry_run} utc={dt.datetime.now(dt.UTC).isoformat()}",
          flush=True)

    if args.only:
        plan = [s for s in STEPS if s[0] in args.only]
    else:
        plan = [s for s in STEPS if s[0] not in args.skip]

    if not plan:
        print("[retrain] no steps selected; nothing to do")
        return 0

    print("[retrain] plan:")
    for name, _fn, desc in plan:
        print(f"[retrain]   - {name}: {desc}")
    print()

    for name, fn, _desc in plan:
        print(f"[retrain] === {name} ===", flush=True)
        try:
            fn(args.version_suffix, args.dry_run)
        except NotImplementedError as e:
            if args.dry_run:
                print(f"[retrain]   (dry-run) NotImplementedError suppressed: {e}")
                continue
            print(f"[retrain]   STOP — step not yet implemented: {e}", flush=True)
            return 2

    print(f"[retrain] done suffix={args.version_suffix}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
