"""Append messages to docs/index.html — single page, newest day first.

We parse with BeautifulSoup so we can re-insert articles idempotently
(same generated_at → same article id → upsert).
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from .state import IST, PT  # noqa: F401

# Only ACTUAL match results get bolded in the web view:
#   * the "X beat Y by Z" outcome line itself
#   * the "Updated top 4: ..." standings (state after a match)
# Predictions, previews, and current snapshots stay regular weight.
_RESULT_LINE_RE  = re.compile(r'^([A-Z][A-Z0-9]+) beat ([A-Z][A-Z0-9]+) by (.+)$')
_UPDATED_TOP4_RE = re.compile(r'^(Updated top 4:\s*)(.+)$', re.IGNORECASE)

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
    article pre strong {
      font-weight: 700;
      color: var(--text);
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

    /* Honeypot — off-screen so humans never see/tab to it */
    .honey {
      position: absolute !important;
      left: -10000px !important;
      width: 1px; height: 1px;
      overflow: hidden;
    }

    /* Inline success / error / cap state */
    .signup-state {
      margin-top: 8px;
      padding: 22px 26px;
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      font-family: var(--font-body);
      font-size: var(--text-base);
      line-height: 1.55;
      color: var(--text);
    }
    .signup-state.success { border-color: var(--teal); box-shadow: 4px 4px 0 0 var(--teal); }
    .signup-state.error   { border-color: var(--crimson); box-shadow: 4px 4px 0 0 var(--crimson); }
    .signup-state .state-label {
      display: block;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 700;   /* bolder per request */
      letter-spacing: 0.22em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .signup-state.success .state-label { color: var(--teal); }
    .signup-state.error   .state-label { color: var(--crimson); }
    .signup-state .state-headline {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 900;   /* bolder per request — full Playfair black */
      font-size: 32px;
      letter-spacing: -0.018em;
      line-height: 1.06;
      color: var(--text);
      margin-bottom: 8px;
    }
    .signup-state.success .state-headline { color: var(--teal); }
    .signup-state.error   .state-headline { color: var(--crimson); }
    .signup-state.success .state-headline em { color: var(--teal); font-style: italic; font-weight: 900; }
    .signup-state.error   .state-headline em { color: var(--crimson); font-style: italic; font-weight: 900; }
    .signup-state .state-body {
      color: var(--text-muted);
      font-size: var(--text-sm);
      line-height: 1.6;
    }
    .signup-state button.try-again {
      margin-top: 14px;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text);
      background: var(--bg);
      border: 2px solid var(--text);
      padding: 8px 14px;
      cursor: pointer;
      transition: background 0.12s;
    }
    .signup-state button.try-again:hover { background: var(--bg-hover); }

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
        <!-- Honeypot — bots fill, humans don't see this -->
        <div class="honey" aria-hidden="true">
          <label for="su-website">Website (leave blank)</label>
          <input id="su-website" name="website" type="text" tabindex="-1" autocomplete="off">
        </div>
        <button class="btn btn-primary" id="signup-submit" type="submit">Sign up →</button>
        <p class="form-fine" id="signup-fine">Each request is reviewed before texts start. Phone format: <code>+14155551234</code> (country code, no spaces or dashes).</p>
      </form>
      <!-- Success / error / cap states (toggled by JS) -->
      <div class="signup-state" id="signup-state" hidden role="status" aria-live="polite"></div>
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

    // Form → Google Apps Script ($0, KC-owned Sheet log, 100/day cap, IP rate-limit, honeypot)
    var SIGNUP_URL = 'https://script.google.com/macros/s/AKfycbx8wwSgBEPz-SMMTtsi2sEt9xzAUWgDBAtN7Wdg94wJb8VLT-Q5dctZDO0rl_1s4yV6/exec';

    var formEl   = document.getElementById('signup-form');
    var stateEl  = document.getElementById('signup-state');
    var submitEl = document.getElementById('signup-submit');

    function showState(kind, label, headline, body, withRetry) {
      formEl.hidden = (kind === 'success' || kind === 'capped');
      stateEl.hidden = false;
      stateEl.className = 'signup-state ' + kind;
      stateEl.innerHTML =
        '<span class="state-label">' + label + '</span>' +
        '<div class="state-headline">' + headline + '</div>' +
        '<div class="state-body">' + body + '</div>' +
        (withRetry ? '<button type="button" class="try-again">Try again</button>' : '');
      var retry = stateEl.querySelector('.try-again');
      if (retry) retry.addEventListener('click', function () {
        stateEl.hidden = true;
        formEl.hidden = false;
        submitEl.disabled = false;
        submitEl.textContent = 'Sign up →';
      });
    }

    // Page-load preflight — disable form if daily cap is full
    fetch(SIGNUP_URL, { method: 'GET' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data && data.full) {
          showState('error',
            'Full for today',
            'Sign-ups are <em>full</em> for today.',
            'Come back after midnight Pacific time — capacity resets daily. Your spot will be there.',
            false);
        }
      })
      .catch(function () { /* silently ignore — form still works */ });

    // Visitor IP — best-effort via ipify (no PII, just rate-limit material)
    var visitorIP = '';
    fetch('https://api.ipify.org?format=json')
      .then(function (r) { return r.json(); })
      .then(function (d) { visitorIP = (d && d.ip) || ''; })
      .catch(function () { /* ignore — Apps Script just skips the per-IP check */ });

    formEl.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var name     = (document.getElementById('su-name').value || '').trim();
      var phone    = (document.getElementById('su-phone').value || '').trim();
      var platform = (document.querySelector('input[name="platform"]:checked') || {}).value || 'iPhone';
      var website  = (document.getElementById('su-website').value || '').trim();   // honeypot

      if (!phone) {
        showState('error', 'Missing phone', 'We need your <em>phone number</em>.',
          'Add it above (with country code, e.g. <code>+14155551234</code>) and try again.', true);
        return;
      }
      if (!/^\\+\\d{8,15}$/.test(phone)) {
        if (!confirm("That doesn\\'t look like an E.164 phone number (e.g. +14155551234). Send anyway?")) return;
      }

      submitEl.disabled = true;
      submitEl.textContent = 'Sending…';

      var payload = JSON.stringify({
        name: name, phone: phone, platform: platform,
        website: website, ip: visitorIP, ua: navigator.userAgent
      });

      // text/plain to avoid the CORS preflight Apps Script doesn't handle
      fetch(SIGNUP_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain;charset=utf-8' },
        body: payload,
        redirect: 'follow'
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data && data.ok) {
            showState('success',
              "You\\'re in",
              "You\\'re <em>on the list.</em>",
              "Match-day texts will start arriving on " + (phone) + " before the next match. Spread the word if you like it.",
              false);
          } else if (data && data.error === 'daily_cap') {
            showState('error', 'Full for today', 'Sign-ups are <em>full</em> for today.',
              data.message || 'Try again after midnight Pacific time.', false);
          } else if (data && data.error === 'rate_limit') {
            showState('error', 'Slow down', 'Too many <em>tries</em>.',
              data.message || 'Wait an hour and try again.', true);
          } else {
            showState('error', 'Something broke', 'We could not <em>save</em> that.',
              (data && data.message) || 'Try again in a minute. If it keeps failing, email kcl.narasimham@gmail.com.', true);
          }
        })
        .catch(function () {
          showState('error', 'Network error', 'Could not <em>reach</em> the server.',
            'Check your connection and try again.', true);
        });
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


def _populate_pre(soup: BeautifulSoup, pre_tag, body_text: str) -> None:
    """Render the message body into a <pre> tag with these adjustments for
    the web archive view (the iMessage text stays plain):

      * Drop the trailing "Archive: <url>" line — redundant when you're
        already looking at the archive.
      * Bold ONLY actual results: the "X beat Y by Z" outcome and the
        "Updated top 4: ..." standings. Predictions, Current top 4,
        Predicted final top 4, match previews — all stay regular weight.
    """
    lines = [ln for ln in body_text.split('\n') if not ln.lstrip().startswith('Archive:')]
    while lines and not lines[-1].strip():
        lines.pop()

    for i, line in enumerate(lines):
        if i > 0:
            pre_tag.append('\n')

        # "DC beat PBKS by 3 Wickets" — actual match result
        m = _RESULT_LINE_RE.match(line)
        if m:
            strong_winner = soup.new_tag('strong')
            strong_winner.string = m.group(1)
            pre_tag.append(strong_winner)
            pre_tag.append(' beat ')
            strong_loser = soup.new_tag('strong')
            strong_loser.string = m.group(2)
            pre_tag.append(strong_loser)
            pre_tag.append(' by ' + m.group(3))
            continue

        # "Updated top 4: RCB, SRH, GT, PBKS" — standings after a result
        m = _UPDATED_TOP4_RE.match(line)
        if m:
            pre_tag.append(m.group(1))
            strong = soup.new_tag('strong')
            strong.string = m.group(2)
            pre_tag.append(strong)
            continue

        # Everything else (Match preview, Prediction, Reason, Current top 4,
        # Predicted final top 4, recap counts, etc.) — plain text
        pre_tag.append(line)


def upsert_message(date_iso: str, msg_type: str, generated_at_iso: str, body: str) -> None:
    soup = _ensure_index()
    details = _find_day_details(soup, date_iso) or _create_day_details(soup, date_iso)

    article_id = f"msg-{date_iso}-{msg_type}"
    existing = soup.find("article", id=article_id)
    if existing:
        existing.decompose()

    article = soup.new_tag("article", id=article_id, attrs={"data-type": msg_type})

    time_tag = soup.new_tag("time", datetime=generated_at_iso)
    time_tag.string = _label_for(msg_type)
    article.append(time_tag)

    pre = soup.new_tag("pre")
    _populate_pre(soup, pre, body)
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
