"""Phase D smoke tests:
  - D1: rendered cricsheet-refresh launchd plist parses (plutil lint).
  - D2: drift monitor runs end-to-end against the current dataset and
        returns a parseable report.
  - D3: venue sub-models experiment script produces a markdown summary.
  - D4: retraining one-shot script is importable + CLI-runnable in --dry-run.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
PY = ROOT / "ml" / ".venv" / "bin" / "python"


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------

def test_d1_plist_template_exists_and_renders():
    """The plist template should exist and, after templating, parse via plutil."""
    template = ROOT / "launchd" / "com.kcln.ipl-cricsheet-refresh.plist"
    assert template.exists(), f"missing {template}"
    text = template.read_text()
    assert "__REPO_PATH__" in text
    assert "__HOME__" in text
    assert "ml.src.ingest" in text
    assert "--force" in text
    # StartCalendarInterval at 18:00 UTC == 23:30 IST. (IST = UTC+5:30, so
    # 23:30 IST = 18:00 UTC.)
    assert "<integer>18</integer>" in text  # Hour
    assert "<integer>0</integer>" in text   # Minute


def test_d1_install_script_dry_run(tmp_path):
    """install_cricsheet_refresh.sh --dry-run renders without launchctl."""
    script = ROOT / "launchd" / "install_cricsheet_refresh.sh"
    assert script.exists()
    r = subprocess.run(["bash", str(script), "--dry-run"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"dry-run failed:\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "dry-run: skipping launchctl bootstrap" in r.stdout
    rendered = pathlib.Path.home() / "Library" / "LaunchAgents" / "com.kcln.ipl-cricsheet-refresh.plist"
    assert rendered.exists(), "expected rendered plist at ~/Library/LaunchAgents"
    rendered_text = rendered.read_text()
    assert "__REPO_PATH__" not in rendered_text
    assert "__HOME__" not in rendered_text
    plutil = subprocess.run(["plutil", "-lint", str(rendered)],
                            capture_output=True, text=True)
    assert plutil.returncode == 0, f"plutil lint failed: {plutil.stdout} {plutil.stderr}"


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------

def test_d2_drift_monitor_json_output_parseable():
    """`drift_monitor --json` should produce parseable JSON with expected keys."""
    r = subprocess.run([str(PY), "-m", "ml.src.v6.drift_monitor", "--json"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode in (0, 1), (
        f"unexpected exit code {r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    )
    report = json.loads(r.stdout)
    for k in ("n_matches", "cumulative_accuracy", "baseline_accuracy",
              "drift_pp", "monthly", "verdict"):
        assert k in report, f"missing key {k} in drift monitor report"
    if report["n_matches"] > 0:
        assert isinstance(report["cumulative_accuracy"], float)
        assert 0.0 <= report["cumulative_accuracy"] <= 1.0
        assert isinstance(report["monthly"], list)
        for m in report["monthly"]:
            assert "month" in m and "n" in m and "accuracy" in m


def test_d2_drift_monitor_text_output_grep_friendly():
    """Default text output should contain `[drift_monitor]` prefix for grepping."""
    r = subprocess.run([str(PY), "-m", "ml.src.v6.drift_monitor"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode in (0, 1)
    assert "[drift_monitor]" in r.stdout


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------

def test_d3_venue_experiment_writes_summary():
    """Running the venue sub-models experiment writes a markdown summary."""
    r = subprocess.run([str(PY), "-m", "ml.src.v6.venue_sub_models"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"venue experiment failed:\n{r.stdout}\n{r.stderr}"
    summary = ROOT / "ml" / "docs" / "v6_venue_experiment.md"
    assert summary.exists(), f"missing summary at {summary}"
    text = summary.read_text()
    assert "Venue sub-model experiment" in text
    # Markdown table header
    assert "| venue |" in text or "|venue|" in text


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------

def test_d4_retrain_script_importable_and_dry_run():
    """The retrain script should be importable and runnable with --dry-run."""
    r = subprocess.run(
        [str(PY), "-m", "ml.src.v6.retrain", "--version-suffix", "20991231",
         "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, f"retrain --dry-run failed:\n{r.stdout}\n{r.stderr}"
    assert "20991231" in r.stdout
    assert "[retrain]" in r.stdout
