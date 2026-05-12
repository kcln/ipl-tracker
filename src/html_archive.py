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
  <title>IPL 2026 — Daily tracker</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@1,400;1,700;1,900&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
  <style>
    /* Page layout on top of brand.css (Editorial × Bauhaus) */
    body { padding: 56px 24px 80px; }
    .wrap { max-width: 760px; margin: 0 auto; }

    /* ─── ESPN Cricinfo live-source card ─── */
    .live-source {
      display: block;
      margin: 0 0 56px;
      padding: 24px 28px;
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      color: inherit;
      transition: transform 0.12s ease-out, box-shadow 0.12s ease-out;
    }
    .live-source:hover {
      transform: translate(-2px, -2px);
      box-shadow: var(--shadow-hover);
    }
    .live-source:active {
      transform: translate(2px, 2px);
      box-shadow: 0 0 0 0 var(--text);
    }
    .live-source .live-label {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--crimson);
      margin-bottom: 16px;
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
      display: flex; align-items: center; justify-content: space-between;
      gap: 24px;
    }
    .live-source .live-logo {
      max-height: 38px; width: auto; display: block;
      mix-blend-mode: multiply;
    }
    .live-source .live-arrow {
      font-family: var(--font-hero); font-style: italic;
      font-size: 32px; font-weight: 900;
      color: var(--crimson);
      flex-shrink: 0;
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

    /* ─── Hero ─── */
    header.page-head { margin-bottom: 64px; }
    header.page-head .eyebrow {
      display: inline-flex; align-items: center; gap: 12px;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--crimson);
      background: var(--bg);
      padding: 6px 14px;
      border: var(--border-width) solid var(--text);
      margin-bottom: 28px;
    }
    header.page-head h1 {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 900;
      font-size: clamp(56px, 9vw, 112px);
      letter-spacing: -0.025em;
      line-height: 0.96;
      color: var(--text);
    }
    header.page-head h1 em {
      font-style: italic;
      color: var(--crimson);
      font-weight: 900;
    }
    header.page-head p {
      font-family: var(--font-body);
      color: var(--text-muted);
      max-width: 56ch;
      font-weight: 400;
      line-height: 1.65;
      margin-top: 24px;
      font-size: var(--text-base);
    }

    /* ─── Day sections ─── */
    main#days { margin-bottom: 72px; }
    details {
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      border-radius: 0;
      padding: 22px 26px;
      margin-bottom: 20px;
      box-shadow: var(--shadow);
    }
    details > summary {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: 24px;
      letter-spacing: -0.01em;
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
      font-size: 20px;
      color: var(--crimson);
      width: 28px; height: 28px;
      display: inline-flex; align-items: center; justify-content: center;
      border: 2px solid var(--text);
      flex-shrink: 0;
    }
    details[open] > summary::after { content: '−'; }
    summary::-webkit-details-marker { display: none; }
    article {
      border-top: 1px solid var(--border);
      margin-top: 18px;
      padding-top: 18px;
    }
    article time {
      display: block;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--crimson);
      font-weight: 500;
      margin-bottom: 12px;
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

    /* ─── Signup section ─── */
    section.signup {
      margin: 72px 0 56px;
      padding: 0;
    }
    section.signup .section-label {
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.28em;
      text-transform: uppercase;
      color: var(--text);
      margin-bottom: 22px;
      display: flex; align-items: center; gap: 14px;
    }
    section.signup .section-label::before {
      content: '';
      width: 28px; height: 2px; background: var(--crimson);
    }
    section.signup h2 {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      font-size: clamp(36px, 5.5vw, 56px);
      letter-spacing: -0.02em;
      line-height: 1.04;
      margin-bottom: 18px;
    }
    section.signup h2 em { font-style: italic; color: var(--crimson); font-weight: 900; }
    section.signup .lead {
      color: var(--text-muted);
      max-width: 56ch;
      line-height: 1.65;
      margin-bottom: 36px;
    }

    /* ─── Platform-tab preview ─── */
    .preview-box {
      background: var(--bg-card);
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      padding: 0;
      margin-bottom: 40px;
    }
    .preview-tabs {
      display: flex;
      border-bottom: 2px solid var(--text);
    }
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
    .preview-tab[aria-selected="true"] {
      background: var(--text);
      color: var(--bg);
    }
    .preview-tab:hover:not([aria-selected="true"]) { background: var(--bg-hover); color: var(--text); }
    .preview-pane { padding: 32px 28px; display: none; }
    .preview-pane[data-active="true"] { display: block; }

    /* iOS bubble */
    .ios-frame {
      background: #fff;
      border: 1px solid rgba(0,0,0,0.08);
      max-width: 360px;
      margin: 0 auto;
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }
    .ios-header {
      background: linear-gradient(to bottom, #f9f9f9, #efefef);
      border-bottom: 1px solid rgba(0,0,0,0.1);
      padding: 14px 12px 12px;
      text-align: center;
      position: relative;
    }
    .ios-header .ios-back {
      position: absolute;
      left: 12px; top: 50%;
      transform: translateY(-50%);
      color: #007aff;
      font-size: 17px;
      line-height: 1;
    }
    .ios-header .ios-name {
      font-size: 13px;
      font-weight: 600;
      color: #000;
    }
    .ios-header .ios-status {
      font-size: 11px;
      color: rgba(0,0,0,0.5);
      margin-top: 2px;
    }
    .ios-body { padding: 18px 14px 22px; background: #fff; min-height: 280px; }
    .ios-time {
      text-align: center;
      font-size: 11px;
      color: rgba(0,0,0,0.45);
      font-weight: 500;
      margin: 4px 0 12px;
    }
    .ios-time strong { font-weight: 600; color: rgba(0,0,0,0.7); }
    .ios-bubble {
      max-width: 80%;
      background: #007aff;
      color: #fff;
      padding: 8px 12px;
      border-radius: 18px;
      font-size: 14px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-wrap: break-word;
      box-shadow: 0 1px 0 rgba(0,0,0,0.05);
    }
    .ios-bubble.left { margin-right: auto; background: #e9e9eb; color: #000; }

    /* Android (Google Messages) bubble */
    .and-frame {
      background: #fff;
      border: 1px solid rgba(0,0,0,0.08);
      max-width: 360px;
      margin: 0 auto;
      font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .and-header {
      background: #fff;
      border-bottom: 1px solid rgba(0,0,0,0.08);
      padding: 14px 16px;
      display: flex; align-items: center; gap: 14px;
    }
    .and-header .and-back {
      color: #444;
      font-size: 20px;
      line-height: 1;
    }
    .and-header .and-avatar {
      width: 32px; height: 32px;
      background: #1a73e8;
      color: #fff;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; font-weight: 600;
    }
    .and-header .and-meta { display: flex; flex-direction: column; }
    .and-header .and-name {
      font-size: 14px; font-weight: 500; color: #202124;
    }
    .and-header .and-status {
      font-size: 11px; color: rgba(0,0,0,0.55);
    }
    .and-body { padding: 16px 12px 22px; background: #fff; min-height: 280px; }
    .and-day {
      text-align: center;
      font-size: 11px;
      color: rgba(0,0,0,0.55);
      font-weight: 500;
      margin: 4px 0 14px;
    }
    .and-bubble {
      max-width: 80%;
      background: #1a73e8;
      color: #fff;
      padding: 10px 14px;
      border-radius: 18px 18px 18px 4px;
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-wrap: break-word;
      margin-right: auto;
    }
    .and-bubble.green {
      background: #e2e2e2; color: #1f1f1f;
    }
    .and-tag {
      font-size: 10px;
      color: rgba(0,0,0,0.45);
      letter-spacing: 0.02em;
      margin-top: 6px;
      text-transform: uppercase;
    }

    /* ─── Form ─── */
    form.signup-form { display: grid; gap: 18px; }
    form.signup-form label {
      display: block;
      font-family: var(--font-label);
      font-size: var(--text-xs);
      font-weight: 500;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--text);
      margin-bottom: 8px;
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
      border-radius: 0;
      box-shadow: var(--shadow-chip);
      transition: transform 0.12s, box-shadow 0.12s;
    }
    form.signup-form input:focus {
      outline: 0;
      transform: translate(-2px, -2px);
      box-shadow: var(--shadow);
      border-color: var(--crimson);
    }
    .platform-radios {
      display: flex;
      gap: 14px;
    }
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
    }
    .platform-radios label.radio:hover {
      transform: translate(-1px, -1px);
      box-shadow: var(--shadow);
    }
    .platform-radios input[type="radio"] {
      position: absolute;
      opacity: 0; width: 0; height: 0;
    }
    .platform-radios input[type="radio"]:checked + span {
      display: block;
    }
    .platform-radios label.radio:has(input:checked) {
      background: var(--text);
      color: var(--bg);
      transform: translate(2px, 2px);
      box-shadow: 0 0 0 0 var(--text);
    }
    button.send-btn {
      font-family: var(--font-body);
      font-weight: 700;
      font-size: var(--text-base);
      letter-spacing: 0.05em;
      text-transform: uppercase;
      background: var(--crimson);
      color: #fff;
      border: var(--border-width) solid var(--text);
      box-shadow: var(--shadow);
      padding: 16px 28px;
      cursor: pointer;
      transition: transform 0.12s, box-shadow 0.12s;
      justify-self: start;
    }
    button.send-btn:hover {
      transform: translate(-2px, -2px);
      box-shadow: var(--shadow-hover);
    }
    button.send-btn:active {
      transform: translate(2px, 2px);
      box-shadow: 0 0 0 0 var(--text);
    }
    .form-fine {
      font-size: var(--text-sm);
      color: var(--text-muted);
      line-height: 1.55;
      margin-top: 8px;
    }
    .form-fine code {
      font-family: var(--font-label);
      font-size: 12px;
      background: var(--bg-hover);
      padding: 2px 6px;
      border: 1px solid var(--border);
    }

    /* ─── Footer ─── */
    footer.page-foot {
      margin-top: 80px;
      padding-top: 36px;
      border-top: 2px solid var(--text);
    }
    .foot-credits {
      display: flex;
      flex-direction: column;
      gap: 18px;
      align-items: flex-start;
      font-size: var(--text-sm);
      color: var(--text-muted);
      line-height: 1.6;
    }
    .foot-credits .built-by {
      font-family: var(--font-hero);
      font-style: italic;
      font-weight: 700;
      color: var(--text);
      font-size: var(--text-md);
      letter-spacing: -0.01em;
    }
    .foot-credits .built-by a {
      color: var(--text);
      border-bottom: 2px solid var(--crimson);
      padding-bottom: 1px;
      transition: color 0.12s;
    }
    .foot-credits .built-by a:hover { color: var(--crimson); }
    .foot-credits .sources {
      font-family: var(--font-label);
      font-size: var(--text-xs);
      color: var(--text-faint);
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    .foot-credits .sources a {
      color: var(--text-muted);
      border-bottom: 1px solid var(--text-faint);
      padding-bottom: 1px;
    }
    .foot-credits .sources a:hover { color: var(--crimson); border-color: var(--crimson); }

    @media (max-width: 540px) {
      body { padding: 32px 18px 64px; }
      .live-source { padding: 18px 18px; }
      .live-source .live-logo { max-height: 28px; }
      .preview-pane { padding: 22px 14px; }
      header.page-head h1 { font-size: 56px; }
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

    <section class="signup">
      <div class="section-label">Get the texts</div>
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
              <div class="and-bubble green">IPL 2026 - Monday, May 11 - Day recap

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
        <button class="send-btn" type="submit">Send my number →</button>
        <p class="form-fine">Submits via your email app — KC reviews each request before adding you. Phone format: <code>+14155551234</code> (country code, no spaces or dashes).</p>
      </form>
    </section>

    <footer class="page-foot">
      <div class="foot-credits">
        <div class="built-by">Built by <a href="https://github.com/kcln/ipl-tracker" target="_blank" rel="noopener noreferrer">KC Lakshminarasimham</a></div>
        <div class="sources">
          Data ·
          <a href="https://www.iplt20.com/" target="_blank" rel="noopener noreferrer">iplt20.com</a>
          ·
          <a href="https://www.espncricinfo.com/series/indian-premier-league-2026-1510719" target="_blank" rel="noopener noreferrer">ESPN Cricinfo</a>
        </div>
      </div>
    </footer>

  </div>

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

    // Form → mailto
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

      var subject = 'IPL tracker — add me: ' + (name || phone);
      var body = [
        'Hi KC,',
        '',
        'Please add me to the IPL 2026 tracker iMessage list.',
        '',
        'Name:     ' + (name || '(not given)'),
        'Phone:    ' + phone,
        'Platform: ' + platform,
        '',
        'Thanks!'
      ].join('\\n');

      var to = 'kcl.narasimham@gmail.com';
      var href = 'mailto:' + to +
                 '?subject=' + encodeURIComponent(subject) +
                 '&body=' + encodeURIComponent(body);
      window.location.href = href;
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
