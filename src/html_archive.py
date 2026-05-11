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
    footer.page-foot {
      margin-top: 64px; padding-top: 24px;
      border-top: 1px solid var(--border);
      font-size: 12px; color: var(--text-faint);
      text-align: center;
    }
    .empty {
      color: var(--text-muted); font-style: italic;
      padding: 32px 0; text-align: center;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page-head">
      <div class="eyebrow">IPL 2026</div>
      <h1>Daily <em>tracker</em></h1>
      <p>A machine-curated record of every match day this season — predictions in the morning, results after each match, a recap at the end of the day.</p>
    </header>
    <main id="days"></main>
    <footer class="page-foot">
      Generated automatically · github.com/kcln/ipl-tracker
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
