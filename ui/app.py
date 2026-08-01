"""
Taxor Bill Extraction — Minimal Web UI

Single-file Flask app. Run with:
    python ui/app.py
"""

import os
import sys
import time
import base64
import tempfile
import threading
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.models.gemini_extractor import GeminiExtractor
from src.models.openrouter_extractor import OpenRouterExtractor
from src.eval.cost_tracker import CostTracker

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

FIELDS = ["vendor", "bill_number", "date", "amount", "currency", "tax_details"]

# ── HTML template ────────────────────────────────────────────────────────────
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxor — Handwritten Bill Extractor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #22263a;
    --border:    #2e3347;
    --accent:    #6c63ff;
    --accent2:   #a78bfa;
    --text:      #e2e4ef;
    --muted:     #6b7280;
    --green:     #22c55e;
    --green-bg:  rgba(34,197,94,.12);
    --yellow:    #f59e0b;
    --yellow-bg: rgba(245,158,11,.12);
    --red:       #f87171;
    --gemini:    #4285f4;
    --gemma:     #7c3aed;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  .container { max-width: 900px; margin: 0 auto; }

  /* ── Header ── */
  header { text-align: center; margin-bottom: 2.5rem; }
  header h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -.5px;
  }
  header p { color: var(--muted); margin-top: .4rem; font-size: .9rem; }
  .free-tier-note {
    display: inline-block; margin-top: .7rem;
    background: rgba(107,99,255,.15); border: 1px solid rgba(107,99,255,.3);
    color: var(--accent2); padding: .3rem .9rem; border-radius: 999px;
    font-size: .78rem; font-weight: 500;
  }

  /* ── Upload card ── */
  .upload-card {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color .2s, background .2s;
    cursor: pointer;
    position: relative;
  }
  .upload-card.drag-over {
    border-color: var(--accent);
    background: rgba(107,99,255,.06);
  }
  .upload-card input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
  }
  .upload-icon { font-size: 2.8rem; margin-bottom: .6rem; }
  .upload-card h2 { font-size: 1.05rem; font-weight: 600; color: var(--text); }
  .upload-card p  { color: var(--muted); font-size: .82rem; margin-top: .3rem; }

  .btn {
    display: inline-flex; align-items: center; gap: .5rem;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    color: #fff; border: none; border-radius: 10px;
    padding: .65rem 1.6rem; font-size: .9rem; font-weight: 600;
    cursor: pointer; margin-top: 1.2rem;
    transition: opacity .2s, transform .1s;
    box-shadow: 0 4px 18px rgba(107,99,255,.35);
  }
  .btn:hover { opacity: .9; }
  .btn:active { transform: scale(.97); }
  .btn:disabled { opacity: .45; cursor: not-allowed; transform: none; }

  /* ── Preview ── */
  #preview-wrap { margin-top: 1.5rem; display: none; }
  #preview-wrap img {
    max-height: 260px; max-width: 100%;
    border-radius: 10px; border: 1px solid var(--border);
    object-fit: contain; background: var(--surface2);
  }

  /* ── Status / Spinner ── */
  #status {
    text-align: center; padding: 1.2rem;
    color: var(--muted); font-size: .9rem; display: none;
  }
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Results ── */
  #results { margin-top: 2rem; display: none; }
  #results h3 {
    font-size: 1rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: .9rem;
  }

  .model-badges {
    display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;
  }
  .badge {
    display: flex; align-items: center; gap: .5rem;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: .45rem .9rem; font-size: .8rem;
  }
  .badge-dot { width: 8px; height: 8px; border-radius: 50%; }
  .badge-label { color: var(--muted); }
  .badge-value { color: var(--text); font-weight: 500; }

  /* ── Table ── */
  .result-table-wrap { overflow-x: auto; border-radius: 12px; }
  table {
    width: 100%; border-collapse: collapse;
    background: var(--surface); font-size: .875rem;
  }
  thead th {
    padding: .8rem 1rem; text-align: left;
    background: var(--surface2); color: var(--muted);
    font-size: .75rem; text-transform: uppercase; letter-spacing: .07em;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  thead th:first-child { border-radius: 12px 0 0 0; }
  thead th:last-child  { border-radius: 0 12px 0 0; }
  tbody td {
    padding: .7rem 1rem; border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  tbody tr:last-child td { border-bottom: none; }
  .field-name { color: var(--muted); font-weight: 500; white-space: nowrap; }
  .val { font-weight: 500; }
  .val.null { color: var(--muted); font-style: italic; font-weight: 400; }

  /* agreement coloring */
  .agree   td:nth-child(2), .agree   td:nth-child(3) { background: var(--green-bg); }
  .disagree td:nth-child(2), .disagree td:nth-child(3) { background: var(--yellow-bg); }
  .agree   .agree-indicator { color: var(--green); }
  .disagree .agree-indicator { color: var(--yellow); }
  .agree-indicator { margin-left: .3rem; font-size: .8rem; }

  /* ── Legend ── */
  .legend {
    display: flex; gap: 1rem; flex-wrap: wrap;
    margin-top: .8rem; font-size: .78rem; color: var(--muted);
  }
  .legend-item { display: flex; align-items: center; gap: .35rem; }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }

  /* ── Error banner ── */
  #error-banner {
    background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.35);
    color: var(--red); border-radius: 10px; padding: .9rem 1.2rem;
    margin-top: 1.2rem; font-size: .875rem; display: none;
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>⚡ Taxor Bill Extractor</h1>
    <p>Upload a handwritten bill image — both models extract in parallel.</p>
    <span class="free-tier-note">⏳ Free-tier models — responses may take 10+ seconds</span>
  </header>

  <div class="upload-card" id="drop-zone">
    <input type="file" id="file-input" accept="image/*">
    <div class="upload-icon">🧾</div>
    <h2>Drop a bill image here</h2>
    <p>or click to browse &nbsp;·&nbsp; JPG, PNG, WEBP (max 16 MB)</p>
  </div>

  <div id="preview-wrap">
    <img id="preview-img" src="" alt="Bill preview">
  </div>

  <div style="text-align:center">
    <button class="btn" id="run-btn" disabled onclick="runExtraction()">
      <span>✦</span> Extract from Both Models
    </button>
  </div>

  <div id="error-banner"></div>
  <div id="status"></div>

  <div id="results">
    <h3>Extraction Results</h3>
    <div class="model-badges" id="meta-badges"></div>
    <div class="result-table-wrap">
      <table id="result-table">
        <thead>
          <tr>
            <th>Field</th>
            <th><span style="color:#4285f4">●</span> Gemini 3.5 Flash-Lite</th>
            <th><span style="color:#7c3aed">●</span> Gemma 4 26B (OpenRouter)</th>
          </tr>
        </thead>
        <tbody id="result-body"></tbody>
      </table>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:rgba(34,197,94,.4)"></span> Both models agree</span>
      <span class="legend-item"><span class="legend-dot" style="background:rgba(245,158,11,.4)"></span> Models disagree</span>
    </div>
  </div>

</div>

<script>
const fileInput = document.getElementById('file-input');
const runBtn    = document.getElementById('run-btn');
const preview   = document.getElementById('preview-img');
const previewW  = document.getElementById('preview-wrap');
const status    = document.getElementById('status');
const results   = document.getElementById('results');
const tbody     = document.getElementById('result-body');
const metaBadges = document.getElementById('meta-badges');
const errBanner = document.getElementById('error-banner');
const dropZone  = document.getElementById('drop-zone');

const FIELDS = ['vendor','bill_number','date','amount','currency','tax_details'];
let selectedFile = null;

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewW.style.display = 'block';
  runBtn.disabled = false;
  results.style.display = 'none';
  errBanner.style.display = 'none';
}

function displayValue(v) {
  if (v === null || v === undefined || v === '') return '<span class="val null">—</span>';
  return `<span class="val">${String(v).replace(/</g,'&lt;')}</span>`;
}

async function runExtraction() {
  if (!selectedFile) return;
  runBtn.disabled = true;
  results.style.display = 'none';
  errBanner.style.display = 'none';
  status.style.display = 'block';
  status.innerHTML = '<span class="spinner"></span> Sending to both models… this may take 10–30 seconds on free tier.';

  const fd = new FormData();
  fd.append('image', selectedFile);

  try {
    const resp = await fetch('/extract', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || 'Server error');
      return;
    }

    renderResults(data);
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    status.style.display = 'none';
    runBtn.disabled = false;
  }
}

function renderResults(data) {
  const g = data.gemini;
  const o = data.openrouter;

  // Meta badges
  metaBadges.innerHTML = `
    <div class="badge">
      <span class="badge-dot" style="background:#4285f4"></span>
      <span class="badge-label">Gemini latency</span>
      <span class="badge-value">${(g.latency_seconds||0).toFixed(2)}s</span>
    </div>
    <div class="badge">
      <span class="badge-dot" style="background:#7c3aed"></span>
      <span class="badge-label">Gemma latency</span>
      <span class="badge-value">${(o.latency_seconds||0).toFixed(2)}s</span>
    </div>
    <div class="badge">
      <span class="badge-dot" style="background:#4285f4"></span>
      <span class="badge-label">Gemini cost</span>
      <span class="badge-value">~$${(g.estimated_cost||0).toFixed(5)}</span>
    </div>
    <div class="badge">
      <span class="badge-dot" style="background:#7c3aed"></span>
      <span class="badge-label">Gemma cost</span>
      <span class="badge-value">$${(o.estimated_cost||0).toFixed(2)} (Free)</span>
    </div>
    ${g.error ? `<div class="badge" style="border-color:rgba(248,113,113,.4)"><span style="color:#f87171">⚠ Gemini: ${g.error.slice(0,60)}</span></div>` : ''}
    ${o.error ? `<div class="badge" style="border-color:rgba(248,113,113,.4)"><span style="color:#f87171">⚠ Gemma: ${o.error.slice(0,60)}</span></div>` : ''}
  `;

  // Table rows
  tbody.innerHTML = FIELDS.map(field => {
    const gv = g[field] !== undefined ? String(g[field]) : null;
    const ov = o[field] !== undefined ? String(o[field]) : null;
    const agree = (gv || '') === (ov || '');
    return `
      <tr class="${agree ? 'agree' : 'disagree'}">
        <td class="field-name">${field.replace('_',' ')}</td>
        <td>${displayValue(g[field])}<span class="agree-indicator">${agree ? '✓' : '↔'}</span></td>
        <td>${displayValue(o[field])}</td>
      </tr>`;
  }).join('');

  results.style.display = 'block';
}

function showError(msg) {
  errBanner.textContent = '⚠ ' + msg;
  errBanner.style.display = 'block';
}
</script>
</body>
</html>
"""


# ── API endpoint ─────────────────────────────────────────────────────────────
def run_extractor(extractor, image_path, result_dict, key):
    try:
        res = extractor.extract(image_path)
        model_name = res.actual_model or extractor.model_name
        cost = CostTracker.calculate_cost(model_name, res.input_tokens, res.output_tokens)
        result_dict[key] = {
            "vendor": res.vendor,
            "bill_number": res.bill_number,
            "date": res.date,
            "amount": res.amount,
            "currency": res.currency,
            "tax_details": res.tax_details,
            "latency_seconds": res.latency_seconds,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "estimated_cost": cost,
            "actual_model": model_name,
            "error": None if res.is_success else (res.error or "Extraction failed"),
        }
    except Exception as e:
        result_dict[key] = {f: None for f in ["vendor","bill_number","date","amount","currency","tax_details"]}
        result_dict[key].update({"error": str(e), "latency_seconds": 0, "estimated_cost": 0, "actual_model": None})


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    results = {}
    t1 = threading.Thread(target=run_extractor, args=(GeminiExtractor(), tmp_path, results, "gemini"))
    t2 = threading.Thread(target=run_extractor, args=(OpenRouterExtractor(), tmp_path, results, "openrouter"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return jsonify(results)


if __name__ == "__main__":
    print("=" * 60)
    print(" Taxor Bill Extractor UI")
    print(" Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)
