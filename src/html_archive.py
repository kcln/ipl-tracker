"""Append messages to docs/index.html — single page, newest day first.

We parse with BeautifulSoup so we can re-insert articles idempotently
(same generated_at → same article id → upsert).
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from .state import IST, PT  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "index.html"


SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IPL 2026 tracker</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=Raleway:wght@700;800;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
  <style>
    /* Page-specific overrides on top of brand.css */
    body { padding: 48px 24px 96px; }
    .wrap { max-width: 760px; margin: 0 auto; }
    header.page-head { margin-bottom: 56px; }
    header.page-head .eyebrow {
      font-family: var(--font-body);
      font-size: 10px; font-weight: 700; letter-spacing: 0.35em;
      text-transform: uppercase; color: var(--crimson);
      display: flex; align-items: center; gap: 14px;
      margin-bottom: 16px;
    }
    header.page-head .eyebrow::before {
      content: ''; width: 28px; height: 1px; background: var(--crimson); flex-shrink: 0;
    }
    header.page-head h1 {
      font-family: var(--font-display); font-weight: 900;
      font-size: clamp(36px, 6vw, 64px); letter-spacing: -0.03em; line-height: 0.98;
    }
    header.page-head h1 em { font-style: normal; color: var(--crimson); }
    header.page-head p {
      color: var(--text-muted); max-width: 56ch;
      font-weight: 300; line-height: 1.6; margin-top: 14px;
      font-size: 15px;
    }
    details {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: var(--shadow-sm);
    }
    details[open] { box-shadow: var(--shadow-md); }
    details > summary {
      font-family: var(--font-display); font-weight: 800;
      font-size: 18px; letter-spacing: -0.01em;
      cursor: pointer; list-style: none;
      display: flex; align-items: center; justify-content: space-between;
    }
    details > summary::after {
      content: '+'; font-weight: 300; font-size: 22px;
      color: var(--text-muted); transition: transform 0.2s;
    }
    details[open] > summary::after { content: '−'; }
    summary::-webkit-details-marker { display: none; }
    article {
      border-top: 1px solid var(--border);
      margin-top: 16px;
      padding-top: 16px;
    }
    article time {
      display: block;
      font-size: 10px; letter-spacing: 0.25em; text-transform: uppercase;
      color: var(--crimson); font-weight: 700;
      margin-bottom: 10px;
    }
    article pre {
      font-family: var(--font-body); font-weight: 400; font-size: 14px;
      line-height: 1.7; color: var(--text);
      white-space: pre-wrap; word-wrap: break-word;
      background: transparent; border: 0; padding: 0; margin: 0;
    }

    /* Live-source card at top — clickable ESPN Cricinfo logo */
    .live-source {
      display: block;
      max-width: 760px;
      margin: 0 auto 40px;
      padding: 28px 32px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: var(--shadow-sm);
      text-decoration: none;
      color: inherit;
      transition: box-shadow 0.18s, border-color 0.18s, transform 0.18s;
      position: relative;
    }
    .live-source:hover {
      box-shadow: var(--shadow-md);
      border-color: rgba(224, 0, 28, 0.25);
      transform: translateY(-1px);
    }
    .live-source .live-label {
      display: block;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.32em;
      text-transform: uppercase;
      color: var(--crimson);
      margin-bottom: 14px;
    }
    .live-source .live-label::before {
      content: '';
      display: inline-block;
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--crimson);
      margin-right: 10px;
      vertical-align: 1px;
      animation: live-pulse 1.6s ease-in-out infinite;
    }
    @keyframes live-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(224,0,28,0.55); }
      50%      { box-shadow: 0 0 0 5px rgba(224,0,28,0); }
    }
    .live-source .live-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }
    .live-source .live-logo {
      max-height: 36px;
      width: auto;
      display: block;
      /* Wikimedia PNG has white background — multiply against the cream card */
      mix-blend-mode: multiply;
    }
    .live-source .live-arrow {
      font-family: var(--font-display);
      font-size: 22px;
      font-weight: 800;
      color: var(--crimson);
      flex-shrink: 0;
      transition: transform 0.18s;
    }
    .live-source:hover .live-arrow { transform: translateX(4px); }
    .live-source .live-sub {
      display: block;
      margin-top: 14px;
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 400;
      line-height: 1.4;
    }
    @media (max-width: 520px) {
      .live-source { padding: 22px 22px; }
      .live-source .live-logo { max-height: 28px; }
    }

    /* Footer */
    footer.page-foot {
      margin-top: 80px; padding-top: 32px;
      border-top: 1px solid var(--border);
      max-width: 760px;
      margin-left: auto;
      margin-right: auto;
    }
    .foot-credits {
      display: flex;
      flex-direction: column;
      gap: 12px;
      align-items: center;
      text-align: center;
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .foot-credits .built-by {
      font-family: var(--font-display);
      font-weight: 800;
      color: var(--text);
      font-size: 14px;
      letter-spacing: -0.01em;
    }
    .foot-credits .built-by em {
      font-style: normal;
      color: var(--crimson);
    }
    .foot-credits .sources {
      font-size: 11px;
      color: var(--text-faint);
      letter-spacing: 0.04em;
    }
    .foot-credits .sources a {
      color: var(--text-muted);
      text-decoration: none;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1px;
    }
    .foot-credits .sources a:hover {
      color: var(--crimson);
      border-color: var(--crimson);
    }
    .foot-credits .repo {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--crimson);
      text-decoration: none;
      font-weight: 600;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 8px 16px;
      border: 1px solid rgba(224,0,28,0.22);
      border-radius: 100px;
      background: rgba(224,0,28,0.06);
      transition: background 0.15s, border-color 0.15s;
    }
    .foot-credits .repo:hover {
      background: rgba(224,0,28,0.12);
      border-color: rgba(224,0,28,0.4);
    }

    .empty {
      color: var(--text-muted); font-style: italic;
      padding: 32px 0; text-align: center;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <a class="live-source" href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener noreferrer">
      <span class="live-label">Live scoreboard</span>
      <span class="live-row">
        <img class="live-logo" src="assets/espncricinfo-logo.png" alt="ESPN Cricinfo" width="500" height="66">
        <span class="live-arrow" aria-hidden="true">→</span>
      </span>
      <span class="live-sub">Open the official IPL 2026 series page for ball-by-ball, scorecards, and commentary.</span>
    </a>
    <header class="page-head">
      <div class="eyebrow">IPL 2026</div>
      <h1>Daily <em>tracker</em></h1>
      <p>A machine-curated record of every match day this season — predictions in the morning, results after each match, a recap at the end of the day.</p>
    </header>
    <main id="days"></main>
    <footer class="page-foot">
      <div class="foot-credits">
        <div class="built-by">Built by <em>KCL</em></div>
        <a class="repo" href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
        <div class="sources">
          Data:
          <a href="https://www.iplt20.com/" target="_blank" rel="noopener noreferrer">iplt20.com</a>
          ·
          <a href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener noreferrer">ESPN Cricinfo</a>
        </div>
      </div>
    </footer>
  </div>
</body>
</html>
"""


def _ensure_index() -> BeautifulSoup:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        INDEX.write_text(SHELL, encoding="utf-8")
    text = INDEX.read_text(encoding="utf-8")
    return BeautifulSoup(text, "html.parser")


def _format_day_long(date_iso: str) -> str:
    return datetime.strptime(date_iso, "%Y-%m-%d").strftime("%A, %B %-d, %Y")


def _find_day_details(soup: BeautifulSoup, date_iso: str):
    return soup.find("details", attrs={"data-day": date_iso})


def _create_day_details(soup: BeautifulSoup, date_iso: str):
    main = soup.find("main", id="days")
    if main is None:
        main = soup.new_tag("main", id="days")
        soup.body.append(main)

    details = soup.new_tag("details", attrs={"data-day": date_iso, "open": ""})
    summary = soup.new_tag("summary")
    summary.string = _format_day_long(date_iso)
    details.append(summary)

    # Insert in date-desc order (newest first)
    inserted = False
    for existing in main.find_all("details", recursive=False):
        existing_day = existing.get("data-day", "")
        if date_iso > existing_day:
            existing.insert_before(details)
            inserted = True
            break
    if not inserted:
        main.append(details)

    # Collapse all but the newest day
    all_details = main.find_all("details", recursive=False)
    for d in all_details[1:]:
        d.attrs.pop("open", None)
    return details


def _label_for(msg_type: str) -> str:
    if msg_type == "morning":
        return "Morning brief"
    if msg_type == "end_of_day":
        return "Day recap"
    if msg_type.startswith("post_match"):
        n = msg_type.split("_")[-1]
        return f"Match {n} result"
    return msg_type.replace("_", " ").title()


def upsert_message(date_iso: str, msg_type: str, generated_at_iso: str, body: str) -> None:
    soup = _ensure_index()
    details = _find_day_details(soup, date_iso) or _create_day_details(soup, date_iso)

    article_id = f"msg-{date_iso}-{msg_type}"
    existing = soup.find("article", id=article_id)
    if existing:
        existing.decompose()

    article = soup.new_tag("article", id=article_id, attrs={"data-type": msg_type})

    # Render timestamp in PT for display
    try:
        dt = datetime.fromisoformat(generated_at_iso)
        ts_human = dt.strftime("%H:%M PT")
    except ValueError:
        ts_human = generated_at_iso

    time_tag = soup.new_tag("time", datetime=generated_at_iso)
    time_tag.string = f"{ts_human} · {_label_for(msg_type)}"
    article.append(time_tag)

    pre = soup.new_tag("pre")
    pre.string = body
    article.append(pre)

    # Insert articles in chronological order within the day
    inserted = False
    for sibling in details.find_all("article", recursive=False):
        sib_time = sibling.find("time")
        sib_iso = sib_time.get("datetime", "") if sib_time else ""
        if generated_at_iso < sib_iso:
            sibling.insert_before(article)
            inserted = True
            break
    if not inserted:
        details.append(article)

    INDEX.write_text(str(soup), encoding="utf-8")
