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

# Bold rules differ by message type — see _populate_pre for the matrix.
_RESULT_LINE_RE      = re.compile(r'^([A-Z][A-Z0-9]+) beat ([A-Z][A-Z0-9]+) by (.+)$')
_PREDICTION_RE       = re.compile(r'^(Prediction:\s*)([A-Z][A-Z0-9]+)( wins.*)$')
_UPDATED_TOP4_RE     = re.compile(r'^(Updated top 4:\s*)(.+)$',          re.IGNORECASE)
_PREDICTED_TOP4_RE   = re.compile(r'^(Predicted final top 4:\s*)(.+)$', re.IGNORECASE)
# Phase-message specific
_UPDATED_PRED_RE     = re.compile(r'^(Updated prediction:\s*)([A-Z][A-Z0-9]+)( wins.*)$')
_PP_SCORE_RE         = re.compile(r'^([A-Z][A-Z0-9]+): (\d+/\d+) after ([\d.]+) overs')
_INNINGS_FINAL_RE    = re.compile(r'^([A-Z][A-Z0-9]+) finished (\d+/\d+) in ([\d.]+) overs\.?$')
_INNINGS_TARGET_RE   = re.compile(r'^([A-Z][A-Z0-9]+) need (\d+) to win\.?$')

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "index.html"


SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IPL 2026 — Daily tracker · KC Lakshminarasimham</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#F8F5F1">
  <meta name="description" content="A machine-curated IPL 2026 tracker — predictions in the morning, results after each match, a recap at night. Texted daily.">
  <meta property="og:title" content="IPL 2026 · Daily tracker">
  <meta property="og:description" content="Predictions in the morning, results after each match, a recap at night. Texted daily.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Work+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:           #F8F5F1;
      --card:         #FFFFFF;
      --card-2:       #FAF8F5;
      --brown:        #3F2A26;
      --brown-2:      #5A3E37;
      --brown-soft:   rgba(63,42,38,0.08);
      --brown-hair:   rgba(63,42,38,0.14);
      --ink:          #1F1612;
      --ink-2:        #3D2E27;
      --ink-soft:     rgba(31,22,18,0.62);
      --ink-faint:    rgba(31,22,18,0.40);
      --hair:         rgba(31,22,18,0.10);
      --hair-soft:    rgba(31,22,18,0.06);
      --p-50:  #FAF5FF;  --p-100: #F3E8FF;  --p-200: #E9D5FF;
      --p-300: #D8B4FE;  --p-400: #C084FC;  --p-500: #A855F7;
      --p-600: #9333EA;  --p-700: #7E22CE;  --p-800: #6B21A8;  --p-900: #581C87;
      --radius-lg: 24px;  --radius-md: 16px;  --radius-sm: 10px;
      --shadow-sm: 0 1px 2px rgba(31,22,18,0.05);
      --shadow-md: 0 1px 2px rgba(31,22,18,0.04), 0 8px 24px -8px rgba(31,22,18,0.10);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { -webkit-font-smoothing: antialiased; }
    body { background: var(--bg); color: var(--ink); font-family: 'Work Sans', system-ui, sans-serif; font-weight: 400; min-height: 100vh; }
    a { color: inherit; text-decoration: none; }

    .wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; }

    nav.bar { display: flex; align-items: center; justify-content: space-between; padding: 28px 0; }
    nav .mark { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 17px; letter-spacing: -0.01em; display: inline-flex; align-items: center; gap: 12px; }
    nav .mark::before { content: ''; width: 24px; height: 24px; background: linear-gradient(135deg, var(--p-400), var(--p-800)); border-radius: 8px; }
    nav .live { font-family: 'Work Sans', sans-serif; font-weight: 600; font-size: 13px; color: var(--ink); padding: 9px 16px; background: var(--card); border-radius: 100px; box-shadow: var(--shadow-sm); transition: all 0.15s; display: inline-flex; align-items: center; gap: 8px; }
    nav .live .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--p-600); animation: pulse 1.6s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(147,51,234,0.6); } 50% { box-shadow: 0 0 0 6px rgba(147,51,234,0); } }
    nav .live:hover { background: var(--brown); color: var(--card); }
    nav .live:hover .dot { background: var(--p-400); }

    .hero-grid { display: grid; grid-template-columns: 2fr 1fr; grid-template-rows: auto auto; gap: 16px; margin-top: 8px; }
    .h-card { background: var(--card); border-radius: var(--radius-lg); padding: 32px; box-shadow: var(--shadow-md); border: 1px solid var(--hair-soft); }
    .h-card.dark { background: var(--brown); color: var(--card); }
    .h-card .kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 18px; }
    .h-card.dark .kicker { color: rgba(255,255,255,0.72); }
    .h-card h1 { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: clamp(40px, 5.5vw, 64px); line-height: 1.0; letter-spacing: -0.025em; }
    .h-card h1 .grad { background: linear-gradient(135deg, var(--p-400), var(--p-800)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .h-card .lead { margin-top: 18px; color: var(--ink-soft); font-size: 16px; line-height: 1.6; max-width: 52ch; }

    .score { display: grid; grid-template-rows: auto 1fr auto; gap: 8px; height: 100%; }
    .score .team-line { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 28px; letter-spacing: -0.02em; line-height: 1.05; margin-top: 8px; }
    .score .team-line .vs { color: rgba(255,255,255,0.55); margin: 0 6px; font-weight: 400; }
    .score .result { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.20em; text-transform: uppercase; color: rgba(255,255,255,0.84); margin-top: auto; }
    .score .result .win { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 14px; letter-spacing: 0; text-transform: none; display: block; margin-top: 4px; color: var(--card); }

    .stat .v { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 56px; letter-spacing: -0.03em; line-height: 1; margin: 10px 0 6px; color: var(--p-700); }
    .stat .desc { font-size: 13px; color: var(--ink-soft); line-height: 1.5; }

    .section-head { margin: 56px 0 18px; display: flex; align-items: center; justify-content: space-between; }
    .section-head h2 { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 22px; letter-spacing: -0.018em; }
    .section-head .count { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-faint); }

    main#days { display: block; }

    details.day { background: var(--card); border-radius: var(--radius-lg); border: 1px solid var(--hair-soft); box-shadow: var(--shadow-sm); margin-bottom: 14px; overflow: hidden; transition: box-shadow 0.2s; }
    details.day:hover { box-shadow: var(--shadow-md); }
    details.day > summary { list-style: none; cursor: pointer; padding: 22px 28px; display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: center; }
    details.day > summary::-webkit-details-marker { display: none; }
    .day-head { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 19px; letter-spacing: -0.015em; line-height: 1.2; }
    .day-sub { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.20em; text-transform: uppercase; color: var(--ink-faint); margin-top: 6px; }
    details.day > summary .toggle { width: 32px; height: 32px; background: var(--bg); border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; color: var(--ink-soft); font-family: 'JetBrains Mono', monospace; font-size: 16px; transition: transform 0.25s, background 0.15s, color 0.15s; }
    details.day[open] > summary .toggle { transform: rotate(45deg); background: var(--p-700); color: var(--card); }

    .day-body { padding: 0 28px 24px; }
    article { padding: 18px 0; border-top: 1px solid var(--hair-soft); display: grid; grid-template-columns: 160px 1fr; gap: 22px; align-items: start; }
    article:first-of-type { border-top: 1px solid var(--hair); }
    article .meta { display: flex; flex-direction: column; gap: 8px; }
    article .meta .tag { display: inline-block; align-self: flex-start; font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; padding: 4px 10px; border-radius: 100px; }
    article .meta .tag.morning { color: var(--p-700); background: var(--p-100); }
    article .meta .tag.phase   { color: var(--p-800); background: var(--p-200); }
    article .meta .tag.result  { color: var(--card);  background: var(--p-700); }
    article .meta .tag.recap   { color: var(--card);  background: var(--brown); }
    article .meta .when { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--ink-faint); line-height: 1.7; }
    article .meta .when span { display: block; }
    article .body { font-family: 'Work Sans', sans-serif; font-size: 14.5px; line-height: 1.7; color: var(--ink-2); white-space: pre-wrap; }
    article .body strong { font-weight: 700; color: var(--p-800); }

    .signup-grid { margin-top: 56px; display: grid; grid-template-columns: 5fr 7fr; gap: 16px; }
    .signup-pitch { background: var(--brown); color: var(--card); border-radius: var(--radius-lg); padding: 36px 32px; position: relative; overflow: hidden; }
    .signup-pitch::before { content: ''; position: absolute; bottom: -50%; right: -30%; width: 160%; height: 200%; background: radial-gradient(circle, rgba(168,85,247,0.40) 0%, transparent 60%); pointer-events: none; }
    .signup-pitch-inner { position: relative; }
    .signup-pitch .kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--p-300); margin-bottom: 18px; }
    .signup-pitch h2 { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: clamp(28px, 4vw, 42px); line-height: 1.05; letter-spacing: -0.022em; }
    .signup-pitch p { margin-top: 16px; color: rgba(255,255,255,0.72); font-size: 15px; line-height: 1.6; }
    .signup-pitch .stats { margin-top: 28px; display: flex; gap: 28px; flex-wrap: wrap; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.10); }
    .signup-pitch .stats div { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(255,255,255,0.55); }
    .signup-pitch .stats strong { display: block; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 22px; letter-spacing: -0.012em; color: var(--card); margin-bottom: 4px; }

    .signup-form-box { background: var(--card); border-radius: var(--radius-lg); padding: 36px 32px; box-shadow: var(--shadow-md); border: 1px solid var(--hair-soft); }
    form { display: grid; gap: 18px; }
    form label { font-family: 'Work Sans', sans-serif; font-weight: 600; font-size: 12px; letter-spacing: 0.04em; color: var(--ink); margin-bottom: 8px; display: block; }
    input[type="text"], input[type="tel"] { width: 100%; font-family: 'Work Sans', sans-serif; font-size: 16px; font-weight: 500; background: var(--bg); color: var(--ink); border: 1px solid transparent; border-radius: var(--radius-sm); padding: 14px 16px; outline: none; transition: all 0.15s; }
    input::placeholder { color: var(--ink-faint); }
    input:focus { background: var(--card); border-color: var(--p-600); box-shadow: 0 0 0 4px var(--p-100); }

    .platform { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .platform label.radio { cursor: pointer; text-align: center; padding: 14px; border-radius: var(--radius-sm); background: var(--bg); color: var(--ink-soft); margin-bottom: 0; font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 14px; letter-spacing: 0; text-transform: none; border: 1px solid transparent; transition: all 0.12s; }
    .platform label.radio:has(input:checked) { background: var(--brown); color: var(--card); }
    .platform input[type="radio"] { display: none; }

    /* Honeypot — off-screen */
    .honey { position: absolute !important; left: -10000px !important; width: 1px; height: 1px; overflow: hidden; }

    button.send { justify-self: stretch; font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 15px; background: var(--brown); color: var(--card); padding: 16px 24px; border: 0; border-radius: var(--radius-sm); cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 10px; transition: all 0.15s; }
    button.send::after { content: '→'; font-size: 18px; transition: transform 0.2s; }
    button.send:hover { background: var(--p-700); }
    button.send:hover::after { transform: translateX(4px); }
    button.send:disabled { opacity: 0.6; cursor: wait; }

    .fine { margin-top: 14px; font-size: 12px; color: var(--ink-faint); line-height: 1.55; }
    .fine code { background: var(--bg); padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

    /* Inline success / error state */
    .signup-state { padding: 28px 32px; background: var(--card); border-radius: var(--radius-lg); border: 1px solid var(--hair-soft); box-shadow: var(--shadow-md); }
    .signup-state .state-label { display: block; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 12px; }
    .signup-state .state-headline { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 26px; letter-spacing: -0.018em; line-height: 1.08; margin-bottom: 8px; }
    .signup-state .state-body { color: var(--ink-soft); font-size: 14px; line-height: 1.6; }
    .signup-state.success .state-label, .signup-state.success .state-headline { color: var(--p-700); }
    .signup-state.success .state-headline em { color: var(--p-800); font-style: italic; font-weight: 900; }
    .signup-state.error   .state-label, .signup-state.error   .state-headline { color: var(--brown); }
    .signup-state.error   .state-headline em { color: var(--brown-2); font-style: italic; font-weight: 900; }
    .signup-state button.try-again { margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink); background: var(--bg); border: 1px solid var(--hair); border-radius: var(--radius-sm); padding: 8px 14px; cursor: pointer; transition: background 0.12s; }
    .signup-state button.try-again:hover { background: var(--p-100); }

    footer.foot { margin: 64px 0 56px; padding: 24px 28px; background: var(--card); border-radius: var(--radius-lg); border: 1px solid var(--hair-soft); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; }
    footer .made { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 14px; letter-spacing: -0.005em; }
    footer .made a { background: linear-gradient(120deg, var(--p-100), var(--p-200)); color: var(--p-800); padding: 3px 10px; border-radius: 100px; font-weight: 600; }
    footer .links { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-faint); }

    @media (max-width: 800px) {
      .hero-grid { grid-template-columns: 1fr; }
      .signup-grid { grid-template-columns: 1fr; }
      article { grid-template-columns: 1fr; gap: 10px; }
    }
    @media (max-width: 540px) {
      .wrap { padding: 0 18px; }
      .h-card { padding: 24px; }
      .h-card h1 { font-size: 40px; }
      details.day > summary { padding: 18px 20px; }
      .day-body { padding: 0 20px 20px; }
      .signup-pitch, .signup-form-box { padding: 28px 24px; }
    }
  </style>
</head>
<body>

  <div class="wrap">

    <nav class="bar">
      <span class="mark">Lakshminarasimham</span>
      <a class="live" href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener"><span class="dot"></span> Live scores →</a>
    </nav>

    <section class="hero-grid">
      <div class="h-card" style="grid-row: span 2;">
        <div class="kicker">IPL 2026 · Daily tracker</div>
        <h1>Every match.<br>Every <span class="grad">prediction.</span></h1>
        <p class="lead">A machine-curated record of every match day this season — a prediction before play, a result after each match, a recap at night. Texted daily to your phone.</p>
      </div>
      <div class="h-card dark">
        <div class="score">
          <div class="kicker">Most recent</div>
          <div class="team-line" id="hero-match">__HERO_MATCH__</div>
          <div class="result">
            <span id="hero-meta">__HERO_META__</span>
            <span class="win" id="hero-win">__HERO_WIN__</span>
          </div>
        </div>
      </div>
      <div class="h-card stat">
        <div class="kicker">Leader</div>
        <div class="v" id="hero-leader">__HERO_LEADER__</div>
        <div class="desc" id="hero-leader-desc">__HERO_LEADER_DESC__</div>
      </div>
    </section>

    <div class="section-head">
      <h2>Match log</h2>
      <span class="count" id="match-count">__MATCH_COUNT__</span>
    </div>

    <main id="days"></main>

    <section class="signup-grid" id="signup">
      <div class="signup-pitch">
        <div class="signup-pitch-inner">
          <div class="kicker">Get the texts</div>
          <h2>Match-day updates, on your phone.</h2>
          <p>Predictions before play. Results as they happen. A clean recap at night. Free, no ads, opt out anytime.</p>
          <div class="stats">
            <div><strong>~3</strong>texts / match day</div>
            <div><strong>1-tap</strong>opt out</div>
          </div>
        </div>
      </div>

      <div class="signup-form-box">
        <form id="signup-form" novalidate>
          <div>
            <label for="su-name">Name</label>
            <input id="su-name" name="name" type="text" autocomplete="name" placeholder="Priya Kumar">
          </div>
          <div>
            <label for="su-phone">Phone (with country code)</label>
            <input id="su-phone" name="phone" type="tel" autocomplete="tel" placeholder="+14155551234" required>
          </div>
          <div>
            <label>Platform</label>
            <div class="platform">
              <label class="radio"><input type="radio" name="platform" value="iPhone" checked> iPhone</label>
              <label class="radio"><input type="radio" name="platform" value="Android"> Android</label>
            </div>
          </div>
          <div class="honey" aria-hidden="true">
            <label for="su-website">Website</label>
            <input id="su-website" name="website" type="text" tabindex="-1" autocomplete="off">
          </div>
          <button class="send" id="signup-submit" type="submit">Sign up</button>
          <p class="fine" id="signup-fine">Each request is reviewed before texts start. Reply <code>STOP</code> anytime to leave, <code>START</code> to rejoin. Format: <code>+14155551234</code></p>
        </form>
        <div class="signup-state" id="signup-state" hidden role="status" aria-live="polite"></div>
      </div>
    </section>

    <footer class="foot">
      <span class="made">Built by <a href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener">KC Lakshminarasimham</a></span>
      <span class="links">iplt20.com · ESPN Cricinfo</span>
    </footer>

  </div>

  <script>
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
        submitEl.textContent = 'Sign up';
      });
    }

    fetch(SIGNUP_URL, { method: 'GET' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data && data.full) {
          showState('error', 'Full for today',
            'Sign-ups are <em>full</em> for today.',
            'Come back after midnight Pacific time — capacity resets daily.',
            false);
        }
      })
      .catch(function () {});

    var visitorIP = '';
    fetch('https://api.ipify.org?format=json')
      .then(function (r) { return r.json(); })
      .then(function (d) { visitorIP = (d && d.ip) || ''; })
      .catch(function () {});

    formEl.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var name     = (document.getElementById('su-name').value || '').trim();
      var phone    = (document.getElementById('su-phone').value || '').trim();
      var platform = (document.querySelector('input[name="platform"]:checked') || {}).value || 'iPhone';
      var website  = (document.getElementById('su-website').value || '').trim();

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

      fetch(SIGNUP_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain;charset=utf-8' },
        body: payload,
        redirect: 'follow'
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data && data.ok) {
            showState('success', "You're in",
              "You're <em>on the list.</em>",
              "Match-day texts will start arriving on " + phone + " before the next match. Spread the word if you like it.",
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
    if msg_type == "season_recap":
        return "Season recap"
    if msg_type.startswith("post_match"):
        n = msg_type.split("_")[-1]
        return f"Match {n} result"
    if msg_type.startswith("toss_"):
        return "Toss"
    if msg_type.startswith("innings_break_"):
        return "Innings break"
    if msg_type.startswith("powerplay_1_"):
        return "Powerplay 1"
    if msg_type.startswith("powerplay_2_"):
        return "Powerplay 2"
    return msg_type.replace("_", " ").title()


def _populate_pre(soup: BeautifulSoup, pre_tag, body_text: str, msg_type: str = '') -> None:
    """Render the message body into a <pre> tag with these adjustments for
    the web archive view (the iMessage text stays plain):

      * Drop the trailing "Archive: <url>" line — redundant when you're
        already looking at the archive.
      * Apply bold rules per message type:
          - morning      : Prediction winner + "Predicted final top 4"
          - post_match_* : only the winning team in "X beat Y by Z"
          - end_of_day   : winning team(s) in result lines + "Updated top 4"
    """
    lines = [ln for ln in body_text.split('\n') if not ln.lstrip().startswith('Archive:')]
    while lines and not lines[-1].strip():
        lines.pop()

    is_morning    = (msg_type == 'morning')
    is_post_match = msg_type.startswith('post_match')
    is_recap      = (msg_type == 'end_of_day')
    is_toss       = msg_type.startswith('toss_')
    is_pp         = msg_type.startswith('powerplay_')
    is_break      = msg_type.startswith('innings_break_')
    is_phase      = is_toss or is_pp or is_break

    def _wrap_strong(text: str):
        s = soup.new_tag('strong')
        s.string = text
        return s

    for i, line in enumerate(lines):
        if i > 0:
            pre_tag.append('\n')

        # ── Morning brief: bold predicted winner ──────────────────
        if is_morning:
            m = _PREDICTION_RE.match(line)
            if m:
                pre_tag.append(m.group(1))
                pre_tag.append(_wrap_strong(m.group(2)))
                pre_tag.append(m.group(3))
                continue

            # Bold "Predicted final top 4: ..." list
            m = _PREDICTED_TOP4_RE.match(line)
            if m:
                pre_tag.append(m.group(1))
                pre_tag.append(_wrap_strong(m.group(2)))
                continue

        # ── Result line "X beat Y by Z" — bold only the winner (X) ──
        if is_post_match or is_recap:
            m = _RESULT_LINE_RE.match(line)
            if m:
                pre_tag.append(_wrap_strong(m.group(1)))
                pre_tag.append(' beat ' + m.group(2) + ' by ' + m.group(3))
                continue

        # ── Bold "Updated top 4: ..." list in both match-result and recap ──
        if is_post_match or is_recap:
            m = _UPDATED_TOP4_RE.match(line)
            if m:
                pre_tag.append(m.group(1))
                pre_tag.append(_wrap_strong(m.group(2)))
                continue

        # ── Phase messages: bold predicted winner in "Updated prediction" ──
        if is_phase:
            m = _UPDATED_PRED_RE.match(line)
            if m:
                pre_tag.append(m.group(1))
                pre_tag.append(_wrap_strong(m.group(2)))
                pre_tag.append(m.group(3))
                continue

        # ── Powerplay: bold the "X/Y" score in "Team: X/Y after Z overs" ──
        if is_pp:
            m = _PP_SCORE_RE.match(line)
            if m:
                team, score, overs = m.group(1), m.group(2), m.group(3)
                # Keep the rest of the line intact (could include " chasing N")
                tail = line[m.end():]
                pre_tag.append(f"{team}: ")
                pre_tag.append(_wrap_strong(score))
                pre_tag.append(f" after {overs} overs")
                pre_tag.append(tail)
                continue

        # ── Innings break: bold the finished score "X/Y" ──
        if is_break:
            m = _INNINGS_FINAL_RE.match(line)
            if m:
                team, score, overs = m.group(1), m.group(2), m.group(3)
                pre_tag.append(f"{team} finished ")
                pre_tag.append(_wrap_strong(score))
                pre_tag.append(f" in {overs} overs.")
                continue
            # Innings break: bold the target "N" in "Team need N to win"
            m = _INNINGS_TARGET_RE.match(line)
            if m:
                team, target = m.group(1), m.group(2)
                pre_tag.append(f"{team} need ")
                pre_tag.append(_wrap_strong(target))
                pre_tag.append(" to win.")
                continue

        # Default: plain text
        pre_tag.append(line)


def upsert_message(date_iso: str, msg_type: str, generated_at_iso: str, body: str) -> None:
    soup = _ensure_index()
    details = _find_day_details(soup, date_iso) or _create_day_details(soup, date_iso)

    article_id = f"msg-{date_iso}-{msg_type}"
    existing = soup.find("article", id=article_id)
    if existing:
        existing.decompose()

    article = soup.new_tag("article", id=article_id, attrs={"data-type": msg_type})
    article["data-generated"] = generated_at_iso

    # Meta column — colored tag + multi-zone time stack
    meta = soup.new_tag("div", attrs={"class": "meta"})

    if msg_type == "morning":
        tag_cls = "morning"
    elif msg_type == "end_of_day":
        tag_cls = "recap"
    elif msg_type.startswith("post_match"):
        tag_cls = "result"
    elif msg_type.startswith("toss_") or msg_type.startswith("powerplay_") or msg_type.startswith("innings_break_"):
        tag_cls = "phase"
    else:
        tag_cls = "morning"
    tag = soup.new_tag("span", attrs={"class": f"tag {tag_cls}"})
    tag.string = _label_for(msg_type)
    meta.append(tag)

    when = soup.new_tag("span", attrs={"class": "when"})
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(generated_at_iso)
        zones = [
            ("IST", dt.astimezone(ZoneInfo("Asia/Kolkata"))),
            ("ET",  dt.astimezone(ZoneInfo("America/New_York"))),
            ("CT",  dt.astimezone(ZoneInfo("America/Chicago"))),
            ("PT",  dt.astimezone(ZoneInfo("America/Los_Angeles"))),
        ]
        for label, d in zones:
            line = soup.new_tag("span")
            line.string = f"{d.strftime('%-I:%M %p').lower().replace(' ', '')} {label}"
            when.append(line)
    except (ValueError, ImportError):
        when.string = generated_at_iso
    meta.append(when)
    article.append(meta)

    # Body — div with white-space: pre-wrap (replaces old <pre>)
    body_div = soup.new_tag("div", attrs={"class": "body"})
    _populate_pre(soup, body_div, body, msg_type=msg_type)
    article.append(body_div)

    # Insert articles in chronological order within the day
    inserted = False
    for sibling in details.find_all("article", recursive=False):
        sib_iso = sibling.get("data-generated", "")
        if not sib_iso:
            sib_time = sibling.find("time")
            sib_iso = sib_time.get("datetime", "") if sib_time else ""
        if generated_at_iso < sib_iso:
            sibling.insert_before(article)
            inserted = True
            break
    if not inserted:
        details.append(article)

    INDEX.write_text(str(soup), encoding="utf-8")
