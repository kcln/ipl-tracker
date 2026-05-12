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
  <title>IPL 2026 — Daily tracker · KC Lakshminarasimham</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#F5F1EB">
  <meta name="description" content="A machine-curated IPL 2026 tracker — predictions in the morning, results after each match, a recap at night. Texted daily.">
  <meta property="og:title" content="IPL 2026 · Daily tracker">
  <meta property="og:description" content="Predictions in the morning, results after each match, a recap at night. Texted daily.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@1,400;1,700;1,900&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
  <style>
    /* Page-specific layout — typography, colors, components all come from brand.css */
    body { padding: 0; }

    /* ── HERO (mirrors kcl-brand) ── */
    .hero {
      position: relative;
      padding: 88px 64px 96px;
      max-width: 1200px;
      margin: 0 auto;
      overflow: visible;
    }
    .hero-shape-red   { width: 200px; height: 200px; top: 60px; left: 40px;       background: var(--crimson); }
    .hero-shape-navy  { width: 130px; height: 130px; top: 50px; right: 80px;      background: var(--indigo); }
    .hero-shape-amber { width: 80px;  height: 80px;  bottom: 40px; right: 220px;  background: var(--amber); }
    .hero-shape-rose  { border-left: 56px solid transparent; border-right: 56px solid transparent; border-bottom: 92px solid var(--rose); bottom: 80px; left: 280px; }
    .hero-inner { position: relative; z-index: 2; }
    .hero-h1 {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: var(--text-3xl);
      line-height: 0.96;
      letter-spacing: -0.025em;
      max-width: 880px;
      margin: var(--space-6) 0;
    }
    .hero-h1 em {
      font-style: italic;
      color: var(--crimson);
      font-weight: 900;
      position: relative;
    }
    .hero-h1 em::after {
      content: '';
      position: absolute;
      left: 0; right: 0; bottom: 8px;
      height: 10px;
      background: var(--crimson);
      opacity: 0.18;
      z-index: -1;
    }
    .hero-sub {
      font-family: var(--font-body);
      font-size: var(--text-md);
      font-weight: 400;
      line-height: 1.55;
      color: var(--text-muted);
      max-width: 620px;
      background: var(--bg);
      padding: 14px 20px;
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      margin-bottom: var(--space-7);
    }
    .hero-actions { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }

    /* ── Generic section wrapper ── */
    .section {
      padding: 64px 64px;
      max-width: 1200px;
      margin: 0 auto;
    }

    /* ── ESPN Cricinfo source card (top of "Live now") ── */
    .live-source {
      display: block;
      padding: 22px 26px;
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      color: inherit;
      margin-bottom: 32px;
      transition: transform 0.08s, box-shadow 0.08s;
    }
    .live-source:hover { transform: translate(-2px, -2px); box-shadow: var(--shadow-hover); }
    .live-source:active { transform: translate(2px, 2px); box-shadow: 0 0 0 0 var(--text); }
    .live-source .live-label {
      display: inline-flex; align-items: center; gap: 10px;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--crimson);
      margin-bottom: 14px;
    }
    .live-source .live-label::before {
      content: '';
      width: 8px; height: 8px;
      background: var(--crimson);
      border-radius: 50%;
      animation: live-pulse 1.6s ease-in-out infinite;
    }
    @keyframes live-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(224,0,28,0.6); }
      50%      { box-shadow: 0 0 0 6px rgba(224,0,28,0); }
    }
    .live-source .live-row {
      display: flex; align-items: center; justify-content: space-between; gap: 24px;
    }
    .live-source .live-logo { max-height: 38px; width: auto; display: block; mix-blend-mode: multiply; }
    .live-source .live-arrow {
      font-family: var(--font-hero); font-style: italic;
      font-size: 32px; font-weight: 900;
      color: var(--crimson); flex-shrink: 0;
      transition: transform 0.18s;
    }
    .live-source:hover .live-arrow { transform: translateX(4px); }
    .live-source .live-sub {
      display: block;
      margin-top: 14px;
      font-size: var(--text-sm);
      color: var(--text-muted);
      line-height: 1.5;
    }

    /* ── Day archive sections (uses brand card aesthetic) ── */
    main#days { margin-bottom: 0; }
    details {
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      padding: 24px 28px;
      margin-bottom: 24px;
    }
    details:hover { transform: translate(-1px, -1px); box-shadow: var(--shadow-hover); transition: transform 0.08s, box-shadow 0.08s; }
    details > summary {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: 26px;
      letter-spacing: -0.015em;
      line-height: 1.1;
      cursor: pointer;
      list-style: none;
      display: flex; align-items: center; justify-content: space-between;
      color: var(--text);
    }
    details > summary::after {
      content: '+';
      font-family: var(--font-label);
      font-weight: 500;
      font-size: 18px;
      color: var(--text);
      width: 30px; height: 30px;
      display: inline-flex; align-items: center; justify-content: center;
      border: 2px solid var(--text);
      flex-shrink: 0;
      background: var(--bg);
    }
    details[open] > summary::after { content: '−'; background: var(--crimson); color: #fff; border-color: var(--crimson); }
    summary::-webkit-details-marker { display: none; }
    article {
      border-top: 1px solid var(--border);
      margin-top: 20px;
      padding-top: 20px;
    }
    article time {
      display: block;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--crimson);
      font-weight: 500;
      margin-bottom: 14px;
      line-height: 1.5;
    }
    article pre {
      font-family: var(--font-body);
      font-weight: 400;
      font-size: var(--text-sm);
      line-height: 1.75;
      color: var(--text);
      white-space: pre-wrap; word-wrap: break-word;
      background: transparent; border: 0; padding: 0; margin: 0;
    }

    /* ── Signup section ── */
    section.signup h2 {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: clamp(40px, 6vw, 68px);
      letter-spacing: -0.022em;
      line-height: 1.02;
      margin-bottom: 20px;
    }
    section.signup h2 em { font-style: italic; color: var(--crimson); font-weight: 900; }
    section.signup .lead {
      font-family: var(--font-body);
      font-size: var(--text-md);
      color: var(--text-muted);
      max-width: 56ch;
      line-height: 1.65;
      margin-bottom: 36px;
    }

    /* ── Platform-tab preview ── */
    .preview-box {
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      padding: 0;
      margin-bottom: 48px;
    }
    .preview-tabs { display: flex; border-bottom: 2px solid var(--text); }
    .preview-tab {
      flex: 1;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      padding: 16px 14px;
      background: var(--bg-card);
      color: var(--text-muted);
      border: 0;
      border-right: 2px solid var(--text);
      cursor: pointer;
      transition: background 0.12s, color 0.12s;
    }
    .preview-tab:last-child { border-right: 0; }
    .preview-tab[aria-selected="true"] { background: var(--text); color: var(--bg); }
    .preview-tab:hover:not([aria-selected="true"]) { background: var(--bg-hover); color: var(--text); }
    .preview-pane { padding: 32px 28px; display: none; }
    .preview-pane[data-active="true"] { display: block; }

    /* iOS bubble */
    .ios-frame {
      background: #fff; border: 1px solid rgba(0,0,0,0.08);
      max-width: 360px; margin: 0 auto;
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }
    .ios-header {
      background: linear-gradient(to bottom, #f9f9f9, #efefef);
      border-bottom: 1px solid rgba(0,0,0,0.1);
      padding: 14px 12px 12px;
      text-align: center;
      position: relative;
    }
    .ios-header .ios-back { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: #007aff; font-size: 17px; line-height: 1; }
    .ios-header .ios-name { font-size: 13px; font-weight: 600; color: #000; }
    .ios-header .ios-status { font-size: 11px; color: rgba(0,0,0,0.5); margin-top: 2px; }
    .ios-body { padding: 18px 14px 22px; background: #fff; min-height: 280px; }
    .ios-time { text-align: center; font-size: 11px; color: rgba(0,0,0,0.45); font-weight: 500; margin: 4px 0 12px; }
    .ios-time strong { font-weight: 600; color: rgba(0,0,0,0.7); }
    .ios-bubble {
      max-width: 80%; background: #007aff; color: #fff;
      padding: 8px 12px; border-radius: 18px;
      font-size: 14px; line-height: 1.4;
      white-space: pre-wrap; word-wrap: break-word;
    }

    /* Android bubble */
    .and-frame {
      background: #fff; border: 1px solid rgba(0,0,0,0.08);
      max-width: 360px; margin: 0 auto;
      font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .and-header { background: #fff; border-bottom: 1px solid rgba(0,0,0,0.08); padding: 14px 16px; display: flex; align-items: center; gap: 14px; }
    .and-header .and-back { color: #444; font-size: 20px; line-height: 1; }
    .and-header .and-avatar { width: 32px; height: 32px; background: #1a73e8; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; }
    .and-header .and-meta { display: flex; flex-direction: column; }
    .and-header .and-name { font-size: 14px; font-weight: 500; color: #202124; }
    .and-header .and-status { font-size: 11px; color: rgba(0,0,0,0.55); }
    .and-body { padding: 16px 12px 22px; background: #fff; min-height: 280px; }
    .and-day { text-align: center; font-size: 11px; color: rgba(0,0,0,0.55); font-weight: 500; margin: 4px 0 14px; }
    .and-bubble {
      max-width: 80%; background: #e2e2e2; color: #1f1f1f;
      padding: 10px 14px; border-radius: 18px 18px 18px 4px;
      font-size: 14px; line-height: 1.45;
      white-space: pre-wrap; word-wrap: break-word;
      margin-right: auto;
    }
    .and-tag { font-size: 10px; color: rgba(0,0,0,0.45); letter-spacing: 0.02em; margin-top: 6px; text-transform: uppercase; }

    /* ── Form (Bauhaus inputs) ── */
    form.signup-form { display: grid; gap: 22px; max-width: 560px; }
    form.signup-form label {
      display: block;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--text);
      margin-bottom: 10px;
    }
    form.signup-form input[type="text"],
    form.signup-form input[type="tel"] {
      width: 100%;
      font-family: var(--font-body);
      font-size: var(--text-base);
      font-weight: 500;
      padding: 14px 16px;
      background: var(--bg-card);
      color: var(--text);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow-chip);
      transition: transform 0.12s, box-shadow 0.12s;
    }
    form.signup-form input:focus {
      outline: 0;
      transform: translate(-2px, -2px);
      box-shadow: var(--shadow);
      border-color: var(--crimson);
    }
    .platform-radios { display: flex; gap: 14px; }
    .platform-radios label.radio {
      flex: 1;
      cursor: pointer;
      font-family: var(--font-body);
      font-weight: 600;
      font-size: var(--text-sm);
      letter-spacing: 0.05em;
      text-transform: uppercase;
      text-align: center;
      padding: 14px;
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow-chip);
      color: var(--text);
      transition: transform 0.12s, box-shadow 0.12s, background 0.12s;
      user-select: none;
      margin-bottom: 0;
    }
    .platform-radios label.radio:hover {
      transform: translate(-1px, -1px);
      box-shadow: var(--shadow);
    }
    .platform-radios input[type="radio"] { position: absolute; opacity: 0; width: 0; height: 0; }
    .platform-radios label.radio:has(input:checked) {
      background: var(--text);
      color: var(--bg);
      transform: translate(2px, 2px);
      box-shadow: 0 0 0 0 var(--text);
    }
    .form-fine {
      font-size: var(--text-sm);
      color: var(--text-muted);
      line-height: 1.55;
      margin-top: 8px;
      max-width: 56ch;
    }
    .form-fine code {
      font-family: var(--font-label);
      font-size: 12px;
      background: var(--bg-hover);
      padding: 2px 6px;
      border: 1px solid var(--border);
    }

    /* ── Footer (mirrors kcl-brand) ── */
    .footer {
      margin-top: var(--space-7);
      padding: 36px 64px;
      border-top: var(--border-width) solid var(--text);
      max-width: 1200px;
      margin-left: auto; margin-right: auto;
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 18px;
      font-family: var(--font-label);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .footer-name {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: 18px;
      letter-spacing: -0.01em;
      text-transform: none;
      color: var(--text);
    }
    .footer-name a {
      color: var(--text);
      border-bottom: 2px solid var(--crimson);
      padding-bottom: 1px;
      transition: color 0.12s;
    }
    .footer-name a:hover { color: var(--crimson); }
    .footer-links { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .footer-links a {
      color: var(--crimson);
      border-bottom: 1px solid var(--crimson);
      text-transform: none;
      letter-spacing: 0.02em;
      font-family: var(--font-body);
      font-size: 13px;
      font-weight: 500;
    }
    .footer-sep { color: var(--text-faint); }

    @media (max-width: 768px) {
      .brand-nav { padding: 16px 20px; }
      .brand-nav-links { display: none; }
      .hero { padding: 48px 20px 72px; }
      .hero-shape-red { width: 130px; height: 130px; top: 30px; left: -30px; }
      .hero-shape-navy { width: 80px; height: 80px; top: 20px; right: -20px; }
      .hero-shape-amber, .hero-shape-rose { display: none; }
      .section { padding: 48px 20px; }
      .footer { padding: 24px 20px; }
      details > summary { font-size: 22px; }
      .live-source .live-logo { max-height: 30px; }
    }

  </style>
</head>
<body>

  <a href="#main" class="skip-link">Skip to content</a>

  <!-- ── NAV ── -->
  <nav class="brand-nav" aria-label="Primary">
    <a href="https://github.com/kcln" class="brand-logo" target="_blank" rel="noopener" aria-label="KC Lakshminarasimham">
      <span class="brand-mark"><img src="assets/lion-transparent.png" alt="Simham lion mark" width="40" height="40"></span>
    </a>
    <ul class="brand-nav-links">
      <li><a href="#match-log" class="active">Match log</a></li>
      <li><a href="#signup">Get the texts</a></li>
      <li><a href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener">Source</a></li>
    </ul>
    <a href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" class="btn btn-primary" target="_blank" rel="noopener">Live Scores →</a>
  </nav>

  <main id="main">

    <!-- ── HERO ── -->
    <section class="hero">
      <div class="brand-shape hero-shape-red"></div>
      <div class="brand-shape brand-shape-circle hero-shape-navy"></div>
      <div class="brand-shape hero-shape-amber"></div>
      <div class="brand-shape brand-shape-triangle hero-shape-rose"></div>

      <div class="hero-inner">
        <div class="brand-eyebrow">IPL 2026 · Daily tracker</div>
        <h1 class="hero-h1">Every match.<br>Every <em>prediction.</em></h1>
        <p class="hero-sub">A machine-curated record of every match day this season — a prediction before play, a result after each match, a recap at night. Texted daily. Public archive below.</p>
        <div class="hero-actions">
          <a href="#signup" class="btn btn-primary">Get the texts</a>
          <a href="#match-log" class="btn btn-ghost">Browse the archive →</a>
        </div>
      </div>
    </section>

    <!-- ── LIVE NOW (ESPN Cricinfo) ── -->
    <section class="section">
      <div class="brand-section-label">Live now</div>
      <a class="live-source" href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener noreferrer">
        <span class="live-label">Live scoreboard</span>
        <span class="live-row">
          <img class="live-logo" src="assets/espncricinfo-logo.png" alt="ESPN Cricinfo" width="500" height="66">
          <span class="live-arrow" aria-hidden="true">→</span>
        </span>
        <span class="live-sub">Open the official IPL 2026 series page for ball-by-ball, scorecards, and commentary.</span>
      </a>
    </section>

    <!-- ── MATCH LOG ── -->
    <section class="section" id="match-log">
      <div class="brand-section-label">Match log</div>
      <div id="days"></div>
    </section>

    <!-- ── SIGNUP ── -->
    <section class="section signup" id="signup">
      <div class="brand-section-label">Get the texts</div>
      <h2>Want the daily updates on <em>your phone?</em></h2>
      <p class="lead">Drop your number below and pick your platform. I'll add you to the next morning's batch — predictions before play, results after every match, day recap at night.</p>

      <div class="preview-box" id="preview-box">
        <div class="preview-tabs" role="tablist" aria-label="Message preview platform">
          <button class="preview-tab" role="tab" aria-selected="true" data-target="ios-pane">iPhone (iMessage)</button>
          <button class="preview-tab" role="tab" aria-selected="false" data-target="and-pane">Android (SMS)</button>
        </div>

        <div class="preview-pane" id="ios-pane" data-active="true" role="tabpanel">
          <div class="ios-frame">
            <div class="ios-header">
              <span class="ios-back">‹</span>
              <div class="ios-name">IPL Tracker</div>
              <div class="ios-status">iMessage</div>
            </div>
            <div class="ios-body">
              <div class="ios-time"><strong>Today</strong> 11:42 PM</div>
              <div class="ios-bubble">IPL 2026 - Monday, May 11 - Day recap

DC beat PBKS by 3 Wickets

Predictions today: 0 of 1 correct

Updated top 4: RCB, SRH, GT, PBKS
Predicted final top 4: SRH, RCB, GT, RR

Archive: https://kcln.github.io/ipl-tracker/</div>
            </div>
          </div>
        </div>

        <div class="preview-pane" id="and-pane" role="tabpanel">
          <div class="and-frame">
            <div class="and-header">
              <span class="and-back">←</span>
              <div class="and-avatar">I</div>
              <div class="and-meta">
                <span class="and-name">IPL Tracker</span>
                <span class="and-status">SMS · just now</span>
              </div>
            </div>
            <div class="and-body">
              <div class="and-day">Today, 11:42 PM</div>
              <div class="and-bubble">IPL 2026 - Monday, May 11 - Day recap

DC beat PBKS by 3 Wickets

Predictions today: 0 of 1 correct

Updated top 4: RCB, SRH, GT, PBKS
Predicted final top 4: SRH, RCB, GT, RR

Archive: https://kcln.github.io/ipl-tracker/</div>
              <div class="and-tag">via SMS · forwarded from iMessage</div>
            </div>
          </div>
        </div>
      </div>

      <form class="signup-form" id="signup-form" novalidate>
        <div>
          <label for="su-name">Name</label>
          <input id="su-name" name="name" type="text" autocomplete="name" placeholder="e.g. Priya Kumar">
        </div>
        <div>
          <label for="su-phone">Phone (with country code)</label>
          <input id="su-phone" name="phone" type="tel" autocomplete="tel" placeholder="+14155551234" required>
        </div>
        <div>
          <label>Platform</label>
          <div class="platform-radios">
            <label class="radio"><input type="radio" name="platform" value="iPhone" checked> iPhone</label>
            <label class="radio"><input type="radio" name="platform" value="Android"> Android</label>
          </div>
        </div>
        <button class="btn btn-primary" type="submit">Sign up →</button>
        <p class="form-fine">Opens WhatsApp with your details pre-filled — you just hit send. KC reviews each request before adding you. Phone format: <code>+14155551234</code> (country code, no spaces or dashes).</p>
      </form>
    </section>

  </main>

  <!-- ── FOOTER ── -->
  <footer class="footer">
    <span class="footer-name">Built by <a href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener">KC Lakshminarasimham</a></span>
    <span class="footer-links">
      <a href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener">Source on GitHub</a>
      <span class="footer-sep">·</span>
      <a href="https://www.iplt20.com/" target="_blank" rel="noopener">iplt20.com</a>
      <span class="footer-sep">·</span>
      <a href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener">ESPN Cricinfo</a>
    </span>
  </footer>

  <script>
    // Platform tabs
    document.querySelectorAll('.preview-tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        document.querySelectorAll('.preview-tab').forEach(function (t) {
          t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
        });
        document.querySelectorAll('.preview-pane').forEach(function (p) {
          p.setAttribute('data-active', p.id === tab.dataset.target ? 'true' : 'false');
        });
      });
    });

    // Sync platform radios with the preview tabs (and vice versa)
    document.querySelectorAll('input[name="platform"]').forEach(function (radio) {
      radio.addEventListener('change', function () {
        var target = radio.value === 'iPhone' ? 'ios-pane' : 'and-pane';
        document.querySelector('.preview-tab[data-target="' + target + '"]').click();
      });
    });
    document.querySelectorAll('.preview-tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        var val = tab.dataset.target === 'ios-pane' ? 'iPhone' : 'Android';
        var radio = document.querySelector('input[name="platform"][value="' + val + '"]');
        if (radio && !radio.checked) radio.checked = true;
      });
    });

    // Form → WhatsApp click-to-chat (no third-party service, no API key, $0)
    // Opens WhatsApp on phone or WhatsApp Web with the message pre-typed —
    // visitor just hits Send and KC gets a real WhatsApp notification.
    document.getElementById('signup-form').addEventListener('submit', function (ev) {
      ev.preventDefault();
      var name = (document.getElementById('su-name').value || '').trim();
      var phone = (document.getElementById('su-phone').value || '').trim();
      var platform = (document.querySelector('input[name="platform"]:checked') || {}).value || 'iPhone';

      if (!phone) {
        alert('Please enter a phone number.');
        return;
      }
      if (!/^\\+\\d{8,15}$/.test(phone)) {
        if (!confirm('That doesn\\'t look like an E.164 phone number (e.g. +14155551234). Send anyway?')) return;
      }

      var message = [
        'Hi KC — please add me to the IPL 2026 tracker.',
        '',
        'Name: ' + (name || '(not given)'),
        'Phone: ' + phone,
        'Platform: ' + platform
      ].join('\\n');

      // KC's WhatsApp number (digits only, with country code, no +)
      var waNumber = '18049287108';
      var href = 'https://wa.me/' + waNumber + '?text=' + encodeURIComponent(message);
      window.open(href, '_blank', 'noopener');
    });
  </script>
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
    main = soup.find(id="days")
    if main is None:
        main = soup.new_tag("div", id="days")
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

    # Render timestamp in all four zones to match the message body convention
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(generated_at_iso)
        ist = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        et = dt.astimezone(ZoneInfo("America/New_York"))
        ct = dt.astimezone(ZoneInfo("America/Chicago"))
        pt = dt.astimezone(ZoneInfo("America/Los_Angeles"))

        def _stamp(d, label):
            return f"{d.strftime('%-I:%M %p').lower().replace(' ', '')} {label}"

        ts_human = (
            f"{_stamp(ist, 'IST')} · {_stamp(et, 'ET')} · "
            f"{_stamp(ct, 'CT')} · {_stamp(pt, 'PT')}"
        )
    except (ValueError, ImportError):
        ts_human = generated_at_iso

    time_tag = soup.new_tag("time", datetime=generated_at_iso)
    time_tag.string = f"{_label_for(msg_type)} · {ts_human}"
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
