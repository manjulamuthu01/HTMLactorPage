import streamlit as st
import json, base64, io, time, re, os, sys, textwrap, tempfile, subprocess, atexit
import requests
from html.parser import HTMLParser
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# PADDLE WORKER — embedded & auto-launched
# ═══════════════════════════════════════════════════════════════════
# The worker script is written to a temp file and started as a
# separate OS process. PaddlePaddle lives entirely in that process —
# the Streamlit process never imports paddleocr, so the
# "PDX already initialized" crash can never happen.

_PADDLE_WORKER_SCRIPT = textwrap.dedent("""
    import os, io, sys
    os.environ.setdefault("FLAGS_use_mkldnn",  "0")
    os.environ.setdefault("FLAGS_use_onednn",  "0")
    os.environ.setdefault("GLOG_v",            "0")
    os.environ.setdefault("GLOG_logtostderr",  "0")

    import numpy as np
    from PIL import Image
    from flask import Flask, request, jsonify

    def _get_paddle_version():
        try:
            import paddleocr as _poc
            ver = getattr(_poc, "__version__", "2.0.0")
            parts = ver.split(".")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            return 2, 7

    maj, _ = _get_paddle_version()
    from paddleocr import PaddleOCR
    if maj >= 3:
        try:    ocr = PaddleOCR(lang="en")
        except: ocr = PaddleOCR()
    else:
        try:    ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False, show_log=False)
        except:
            try: ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except: ocr = PaddleOCR(lang="en")

    def _pdf_to_img_np(file_bytes):
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=200)
        return np.array(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))

    def _run_ocr(img_np):
        items = []
        if hasattr(ocr, "predict"):
            try:
                for res in (ocr.predict(img_np) or []):
                    rt = (res.get("rec_texts") if hasattr(res,"get") else None) or getattr(res,"rec_texts",None)
                    rp = ((res.get("rec_polys") or res.get("dt_polys")) if hasattr(res,"get") else None) or getattr(res,"rec_polys",None) or getattr(res,"dt_polys",None)
                    if rt:
                        for i,t in enumerate(rt):
                            if not t or not str(t).strip(): continue
                            if rp and i<len(rp):
                                poly=rp[i]; ys=[p[1] for p in poly]; xs=[p[0] for p in poly]
                                items.append({"text":str(t).strip(),"y":sum(ys)/len(ys),"x":sum(xs)/len(xs)})
                            else:
                                items.append({"text":str(t).strip(),"y":float(len(items)),"x":0.0})
                if items: return items
            except: pass
        if hasattr(ocr, "ocr"):
            try:    r2 = ocr.ocr(img_np, cls=True)
            except: r2 = ocr.ocr(img_np)
            if r2 and r2[0]:
                for line in r2[0]:
                    try:    poly,text = line[0],line[1][0]
                    except: continue
                    if text and str(text).strip():
                        if poly:
                            ys=[p[1] for p in poly]; xs=[p[0] for p in poly]
                            items.append({"text":str(text).strip(),"y":sum(ys)/len(ys),"x":sum(xs)/len(xs)})
                        else:
                            items.append({"text":str(text).strip(),"y":float(len(items)),"x":0.0})
        return items

    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status":"ok"})

    @app.route("/ocr", methods=["POST"])
    def ocr_endpoint():
        try:
            f = request.files.get("image")
            if not f: return jsonify({"status":"error","message":"no image"}),400
            fb = f.read(); ext = (f.filename or "").rsplit(".",1)[-1].lower()
            img_np = _pdf_to_img_np(fb) if ext=="pdf" else np.array(Image.open(io.BytesIO(fb)).convert("RGB"))
            return jsonify({"status":"ok","items":_run_ocr(img_np)})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)}),500

    if __name__ == "__main__":
        app.run(host="127.0.0.1", port=5050, debug=False, threaded=False)
""")

@st.cache_resource(show_spinner=False)
def _start_paddle_worker():
    """Write the worker script to a temp file and launch it as a subprocess.
    @st.cache_resource ensures this runs exactly ONCE per Streamlit server
    lifetime — never on reruns — so the worker is started only once."""
    # Check if already running from a previous server start
    try:
        r = requests.get("http://localhost:5050/health", timeout=2)
        if r.status_code == 200:
            return "already_running"
    except Exception:
        pass

    # Write script to a temp file that persists for the process lifetime
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_paddle_worker.py",
                                      delete=False, encoding="utf-8")
    tmp.write(_PADDLE_WORKER_SCRIPT)
    tmp.flush()
    tmp.close()

    proc = subprocess.Popen(
        [sys.executable, tmp.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 90s for the worker to be ready
    for _ in range(90):
        time.sleep(1)
        try:
            r = requests.get("http://localhost:5050/health", timeout=2)
            if r.status_code == 200:
                atexit.register(proc.terminate)  # clean up when Streamlit stops
                return "started"
        except Exception:
            pass
        if proc.poll() is not None:
            return "failed"
    return "timeout"

# Start the worker immediately when this script loads
_paddle_worker_status = _start_paddle_worker()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OCR IDP Application",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --bg:#fafbfc; --surface:#ffffff; --border:#e2e5eb; --border-strong:#cdd2dc;
    --accent:#2952e3; --accent-dark:#1c3cb8; --accent-soft:#eef1fd;
    --green:#15803d; --green-soft:#f0fdf4; --green-border:#bbf0cf;
    --amber:#b45309; --amber-soft:#fffaeb; --amber-border:#fbe3a8;
    --red:#b91c1c; --red-soft:#fef2f2; --red-border:#fad0cc;
    --text:#16181d; --muted:#6b7280; --muted-2:#9aa1ad;
}
html,body,[class*="css"],.stApp { background-color:var(--bg)!important; color:var(--text)!important; font-family:'Inter',sans-serif!important; }
#MainMenu,footer { visibility:hidden; }

/* ── Header ───────────────────────────────────────────── */
.brand { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.6rem 2rem; margin-bottom:1.5rem; border-left:3px solid var(--accent); }
.brand h1 { font-size:1.65rem; font-weight:700; letter-spacing:-0.5px; color:var(--text); margin:0 0 4px 0; line-height:1.2; }
.brand p { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; margin:0; }

/* ── Section labels ───────────────────────────────────── */
h4 { font-weight:700!important; font-size:0.95rem!important; color:var(--text)!important; letter-spacing:-0.2px!important; margin-bottom:0.6rem!important; }
.ctrl-label { font-size:0.78rem; font-weight:600; color:var(--muted); margin-bottom:0.4rem; }

/* ── Inputs ───────────────────────────────────────────── */
[data-testid="stSelectbox"]>div>div { background:var(--surface)!important; border:1px solid var(--border-strong)!important; border-radius:8px!important; color:var(--text)!important; font-size:0.88rem!important; }
[data-testid="stTextInput"] input { background:var(--surface)!important; border:1px solid var(--border-strong)!important; border-radius:8px!important; color:var(--text)!important; font-family:'JetBrains Mono',monospace!important; font-size:0.82rem!important; }
[data-testid="stTextInput"] input:focus { border-color:var(--accent)!important; box-shadow:0 0 0 3px var(--accent-soft)!important; }
[data-testid="stFileUploadDropzone"] { background:var(--surface)!important; border:1.5px dashed var(--border-strong)!important; border-radius:10px!important; transition:border-color 0.2s,background 0.2s!important; }
[data-testid="stFileUploadDropzone"]:hover { border-color:var(--accent)!important; background:var(--accent-soft)!important; }

/* ── Buttons ──────────────────────────────────────────── */
.stButton>button { background:var(--accent)!important; color:#fff!important; border:none!important; border-radius:8px!important; font-weight:600!important; font-size:0.88rem!important; letter-spacing:0.2px!important; padding:0.6rem 1.5rem!important; transition:background 0.15s!important; width:100%!important; }
.stButton>button:hover { background:var(--accent-dark)!important; }
.stButton>button:disabled { opacity:0.4!important; }
[data-testid="stDownloadButton"]>button { background:var(--surface)!important; color:var(--text)!important; border:1px solid var(--border-strong)!important; border-radius:8px!important; font-weight:600!important; font-size:0.85rem!important; width:100%!important; }
[data-testid="stDownloadButton"]>button:hover { border-color:var(--accent)!important; color:var(--accent)!important; }

/* ── Status boxes ─────────────────────────────────────── */
.info-box, .warn-box, .err-box, .ok-box { border-radius:8px; padding:0.7rem 1rem; font-size:0.82rem; margin:0.5rem 0; line-height:1.5; border:1px solid; }
.info-box { background:var(--accent-soft); border-color:#c7d2f7; color:var(--accent-dark); }
.warn-box { background:var(--amber-soft); border-color:var(--amber-border); color:var(--amber); }
.err-box  { background:var(--red-soft); border-color:var(--red-border); color:var(--red); white-space:pre-wrap; font-family:'JetBrains Mono',monospace; font-size:0.78rem; }
.ok-box   { background:var(--green-soft); border-color:var(--green-border); color:var(--green); }

/* ── Metrics ──────────────────────────────────────────── */
.mrow { display:flex; gap:0.75rem; margin:1rem 0; }
.mbox { flex:1; background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.9rem 0.5rem; text-align:center; }
.mval { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--text); }
.mlbl { font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.6px; margin-top:4px; font-weight:500; }

/* ── Pipeline steps ───────────────────────────────────── */
.pstep { display:flex; align-items:center; gap:10px; padding:7px 12px; border-radius:6px; font-size:0.78rem; font-weight:500; margin-bottom:4px; border:1px solid transparent; color:var(--muted-2); transition:all 0.2s; }
.pstep .pnum { font-family:'JetBrains Mono',monospace; font-size:0.68rem; width:16px; }
.pstep.active { background:var(--accent-soft); color:var(--accent-dark); font-weight:600; }
.pstep.done { color:var(--green); }
.pstep.done .pnum { color:var(--green); }

/* ── Pills / badges ───────────────────────────────────── */
.pill { display:inline-flex; align-items:center; gap:6px; font-size:0.76rem; font-weight:600; padding:4px 12px; border-radius:6px; margin-bottom:0.8rem; border:1px solid; }
.pill.ok  { background:var(--green-soft); border-color:var(--green-border); color:var(--green); }
.pill.run { background:var(--amber-soft); border-color:var(--amber-border); color:var(--amber); }
.pill.err { background:var(--red-soft); border-color:var(--red-border); color:var(--red); }
.mbadge { display:inline-flex; align-items:center; gap:8px; background:var(--surface); border:1px solid var(--border-strong); color:var(--text); font-size:0.78rem; font-weight:600; padding:5px 14px; border-radius:7px; margin-bottom:0.8rem; }

/* ── Result card / tabs ───────────────────────────────── */
.rcard { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.4rem; min-height:280px; }
.empty-state { text-align:center; padding:3rem 1rem; color:var(--muted-2); font-size:0.88rem; }
[data-baseweb="tab-list"] { background:transparent!important; border-bottom:1px solid var(--border)!important; gap:1.2rem!important; }
[data-baseweb="tab"] { color:var(--muted)!important; font-weight:600!important; font-size:0.85rem!important; padding:0.6rem 0.1rem!important; }
[aria-selected="true"] { color:var(--accent)!important; border-bottom:2px solid var(--accent)!important; }
[data-baseweb="tab-panel"] { background:transparent!important; padding:1rem 0!important; }
textarea, code, pre { font-family:'JetBrains Mono',monospace!important; }
textarea { background:var(--surface)!important; color:var(--text)!important; border:1px solid var(--border-strong)!important; border-radius:8px!important; font-size:0.82rem!important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--muted-2); }
hr { border-color:var(--border)!important; margin:1.2rem 0!important; }

/* ── Login page ───────────────────────────────────────── */
.login-wrap { max-width:420px; margin:9vh auto 0 auto; }
.login-logo { text-align:center; font-size:2.6rem; font-weight:900; color:var(--accent); letter-spacing:-1px; margin-bottom:0.3rem; }
.login-sub  { text-align:center; color:var(--muted); font-size:0.95rem; margin-bottom:2rem; }
.login-card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:2rem 2.2rem; }

/* ── App shell: sidebar + main ───────────────────────────── */
[data-testid="stSidebar"] { background:#f6f7f9!important; border-right:1px solid var(--border)!important; }
[data-testid="stSidebar"] > div { padding-top:1.2rem; }
.sb-title { font-size:1.15rem; font-weight:800; color:var(--text); letter-spacing:-0.3px; margin-bottom:2px; padding:0 0.2rem; }
.sb-sub   { font-size:0.78rem; color:var(--muted); margin-bottom:1.1rem; padding:0 0.2rem; }
.sb-section-label { font-size:0.72rem; font-weight:700; color:var(--muted-2); text-transform:uppercase; letter-spacing:0.6px; margin:1rem 0 0.5rem 0.2rem; }
[data-testid="stSidebar"] .stButton>button {
    background:var(--surface)!important; color:var(--text)!important; border:1px solid var(--border-strong)!important;
    font-weight:600!important; text-align:left!important; justify-content:flex-start!important; padding:0.55rem 0.9rem!important;
}
[data-testid="stSidebar"] .stButton>button:hover { border-color:var(--accent)!important; color:var(--accent)!important; background:var(--accent-soft)!important; }
[data-testid="stSidebar"] .nav-active button {
    background:var(--accent-soft)!important; border-color:var(--accent)!important; color:var(--accent-dark)!important;
}
.hist-row { display:flex; gap:6px; align-items:stretch; margin-bottom:6px; }
.hist-row .stButton>button { font-size:0.78rem!important; padding:0.5rem 0.7rem!important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hist-del button { background:var(--surface)!important; border:1px solid var(--border-strong)!important; color:var(--muted)!important; padding:0.3rem 0.5rem!important; min-width:0!important; }
.hist-del button:hover { border-color:var(--red)!important; color:var(--red)!important; background:var(--red-soft)!important; }
.sb-user { font-size:0.78rem; color:var(--muted); padding:0.6rem 0.2rem; border-top:1px solid var(--border); margin-top:1rem; }
.sb-user b { color:var(--text); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SUPPORTED = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "webp", "bmp"]

PIPELINE_STEPS = [
    "Document Ingestion",
    "Image Preprocessing",
    "Layout Analysis",
    "Vision-Language Understanding",
    "OCR Recognition",
    "Structure Reconstruction",
    "Data Extraction",
    "Structured JSON Output",
]

CHANDRA_API_URL  = "https://www.datalab.to/api/v1/marker"

# ── Added: engine choices, Ollama + PaddleOCR settings ──────────────
MODELS = {
    "Chandra OCR — Datalab Marker API": "chandra-ocr",
    "QWEN 2.5 VL — Local Ollama (no key needed)": "qwen2.5-vl",
    "PaddleOCR — Local (no key needed)": "paddleocr",
}
OLLAMA_BASE_URL  = "http://localhost:11434"
QWEN_MODELS_PREF = ["qwen2.5vl:3b", "qwen2.5vl:7b", "qwen2.5vl:72b"]
# Preferred models for the LLM-based JSON *structuring* step (text-only
# call, no image). Pure text models are faster/lighter than the VL model
# for this step, but we fall back to whatever VL model is loaded if no
# dedicated text model has been pulled.
STRUCTURING_MODELS_PREF = ["qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b", "phi3:mini"]

# ── Added: sign-in gate ──────────────────────────────────────────────
# Single hardcoded username/password, suitable for internal/team use.
# Override via environment variables in any real deployment rather than
# editing this file, so credentials aren't sitting in source control.
import os
APP_USERNAME = os.environ.get("OCR_APP_USERNAME", "admin")
APP_PASSWORD = os.environ.get("OCR_APP_PASSWORD", "admin123")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def render_login():
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-logo">◆ OCR IDP Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Sign in to continue</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown('<div class="ctrl-label">Username</div>', unsafe_allow_html=True)
    username = st.text_input("Username", label_visibility="collapsed", key="login_username")
    st.markdown('<div class="ctrl-label" style="margin-top:0.9rem">Password</div>', unsafe_allow_html=True)
    password = st.text_input("Password", type="password", label_visibility="collapsed", key="login_password")

    st.markdown('<div style="margin-top:1.1rem">', unsafe_allow_html=True)
    submitted = st.button("Sign in", use_container_width=True, key="login_submit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.markdown('<div class="err-box">Invalid username or password.</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION ENGINE  (rule-free — no per-document-type regex parsers)
# ═══════════════════════════════════════════════════════════════════
#
# Design: every OCR engine (Chandra / QWEN / PaddleOCR) only has to
# produce plain text (+ optional HTML tables). A single LLM-based
# "structuring" step then reads that raw text and returns clean,
# nested JSON for ANY document type — passport, Aadhaar, invoice,
# marksheet, bank form, or something we've never seen before.
#
# There is intentionally no keyword/regex document-type detector and
# no per-field regex extraction anywhere in this file. The model is
# asked to both classify the document type AND extract structured
# fields in one call, because that is the part that doesn't scale as
# a hand-written rule set.

class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._buf, self._in_cell = [], [], [], False
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self._row = []
        elif tag in ("td","th"): self._buf=[]; self._in_cell=True
    def handle_endtag(self, tag):
        if tag in ("td","th"):
            self._row.append("".join(self._buf).strip()); self._in_cell=False
        elif tag == "tr" and self._row:
            self.rows.append(self._row); self._row=[]
    def handle_data(self, data):
        if self._in_cell: self._buf.append(data)

def _parse_html_table(html_str):
    p = _HTMLTableParser(); p.feed(html_str or ""); return p.rows

def _strip_html(html_str: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</tr>|</td>|</th>", "\n", html_str or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())

def _dedup_lines(text: str) -> str:
    """Marker/Chandra sometimes doubles entire blocks of text. This is a
    pure text-cleanup pass (not a parsing rule) — it removes exact-duplicate
    consecutive-ish lines so the LLM structuring step isn't confused by
    repeated content."""
    seen, out = set(), []
    for line in (text or "").splitlines():
        s = line.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return "\n".join(out)

def _walk_json_tree_for_text(node, raw_blocks, table_htmls):
    """Generic walker: pulls plain text and table HTML out of a Marker-style
    JSON block tree. Does NOT try to interpret field meaning — that's the
    LLM's job downstream."""
    if not isinstance(node, dict):
        return
    btype = node.get("block_type", "")
    html  = node.get("html", "") or node.get("text", "")
    if btype == "Table" and html:
        table_htmls.append(html)
    elif html:
        plain = _strip_html(html).strip()
        if plain:
            raw_blocks.append(plain)
    for key in ("children", "pages", "blocks", "lines"):
        for child in node.get(key, []):
            _walk_json_tree_for_text(child, raw_blocks, table_htmls)
    for span in node.get("spans", []):
        text = (span.get("text") or "").strip()
        if text:
            raw_blocks.append(text)

def collect_text_and_tables(content: dict, raw_text: str = ""):
    """Pulls all plain text + table HTML out of a Marker-style JSON tree,
    merges with any separately-supplied raw_text, and returns
    (combined_text, list_of_table_html_strings). Pure extraction, no
    interpretation — every OCR engine funnels into this one function."""
    raw_blocks, table_htmls = [], []
    if content and isinstance(content, dict):
        _walk_json_tree_for_text(content, raw_blocks, table_htmls)
    tree_text = "\n".join(raw_blocks)
    combined = "\n".join(filter(None, [tree_text, raw_text]))
    combined = _dedup_lines(combined)
    return combined, table_htmls


# ─────────────────────────────────────────────────────────────────────
# LLM-BASED STRUCTURING  (replaces every regex/keyword rule)
# ─────────────────────────────────────────────────────────────────────

_STRUCTURING_SYSTEM_PROMPT = """Extract structured data from OCR text into JSON.

Rules:
- Output ONLY a JSON object. No markdown, no explanation, no code fences.
- First key must be "document_type" (e.g. "Invoice", "PAN Card", "Sales Report", "Passport").
- Extract every visible field. Group logically (e.g. "header", "line_items", "totals").
- For tables/lists: use a JSON array of objects.
- Omit fields that are missing or unreadable. Do not guess.
- Keep values exactly as they appear in the text.

Output format:
{"document_type": "...", "field1": "value1", "section": {"key": "val"}, "items": [...]}"""

def _build_structuring_user_prompt(raw_text: str, table_htmls: list) -> str:
    # Cap text at 3000 chars so small local models can output valid JSON
    # within their token limit. Large docs (sales reports, multi-page PDFs)
    # would otherwise cause truncated/invalid JSON output.
    text = (raw_text or "").strip()
    if len(text) > 3000:
        text = text[:3000] + "\n...[truncated]"
    parts = [f"OCR TEXT:\n{text}"]
    if table_htmls:
        for i, html in enumerate(table_htmls[:3], 1):  # max 3 tables
            rows = _parse_html_table(html)
            if rows:
                rendered = "\n".join(" | ".join(r) for r in rows[:20])  # max 20 rows
                parts.append(f"\nTABLE {i}:\n{rendered}")
    return "\n\n".join(parts)

def _repair_json(raw: str):
    """Best-effort parse of LLM JSON output. Handles:
    - markdown fences (```json ... ```)
    - leading/trailing prose
    - truncated JSON (model hit token limit mid-object)
    - single-quoted keys (some small models output these)
    """
    if not raw:
        return None
    cleaned = raw.strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract first {...} block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = cleaned[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            pass
        # Truncated JSON — try progressively shorter substrings by
        # finding the last complete key:value pair before truncation
        for trim_end in range(len(chunk)-1, max(start, len(chunk)-200), -1):
            candidate = chunk[:trim_end].rstrip().rstrip(",") + "}"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    parsed["_truncated"] = True
                    return parsed
            except Exception:
                continue

    # Last resort: try replacing single quotes with double quotes
    try:
        fixed = re.sub(r"(?<![\\])\'", '"', cleaned)
        return json.loads(fixed)
    except Exception:
        pass

    return None

def _ollama_generate(model_name: str, prompt: str, num_ctx: int, num_predict: int) -> str:
    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": True,   # streaming avoids read-timeout on slow/large models
        "options": {
            "temperature": 0,
            "num_predict":  num_predict,
            "num_ctx":      num_ctx,
        },
    }
    # Use streaming + per-chunk timeout instead of a single 300s wall.
    # Non-streamed calls block until the model finishes generating the
    # ENTIRE response before sending a single byte, so large documents
    # easily exceed any fixed timeout. Streaming resets the read timeout
    # on every token chunk, so the socket stays alive as long as the
    # model keeps producing output.
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=(30, 300),  # (connect_timeout_s, per-chunk_read_timeout_s) — 300s for slow machines
    )
    r.raise_for_status()
    parts = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            break
    result = "".join(parts).strip()
    if not result:
        raise RuntimeError(
            "Ollama returned an empty response. "
            "This usually means it ran out of memory mid-generation. "
            "Try closing other apps to free RAM, then run OCR again."
        )
    return result

def structure_with_ollama(raw_text: str, table_htmls: list, model_name: str,
                           num_ctx: int = 2048, num_predict: int = 1024) -> dict:
    """Universal structuring step for engines that don't already return
    LLM-structured JSON (QWEN raw-text mode, PaddleOCR, Chandra). Uses a
    local Ollama text model — no hardcoded fields, no document-type rules.
    Retries once with a corrective prompt if the model's first response
    isn't valid JSON, since that's the most common transient failure mode
    for small local models rather than a sign the document is unreadable."""
    user_prompt = _build_structuring_user_prompt(raw_text, table_htmls)
    full_prompt = _STRUCTURING_SYSTEM_PROMPT + "\n\n" + user_prompt

    raw_json = _ollama_generate(model_name, full_prompt, num_ctx, num_predict)
    parsed = _repair_json(raw_json)

    if not isinstance(parsed, dict):
        retry_prompt = (
            full_prompt +
            "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Respond with ONLY a single valid JSON object — no markdown fences, "
            "no commentary, no leading or trailing text of any kind."
        )
        raw_json_retry = _ollama_generate(model_name, retry_prompt, num_ctx, max(num_predict, 2048))
        parsed = _repair_json(raw_json_retry)
        if isinstance(parsed, dict):
            return parsed
        return {
            "document_type": "Unknown",
            "_structuring_error": "Model output could not be parsed as JSON after retry.",
            "_raw_model_output": (raw_json_retry or raw_json)[:2000],
        }

    return parsed

def structure_with_claude_fallback(raw_text: str, table_htmls: list) -> dict:
    """Fallback structuring when Ollama is unavailable or times out.
    Parses the raw OCR text with simple heuristics to extract key:value
    pairs so the user gets *something* useful even without a local LLM."""
    import re
    fields = {}
    # Extract key:value lines (covers most ID cards, invoices, forms)
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "Label : Value" or "Label - Value"
        m = re.match(r"^([A-Za-z][A-Za-z0-9 /()]{1,40})\s*[:\-]\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            if key and val and len(val) < 200:
                fields[key] = val
    return {
        "document_type": "Unknown",
        **(fields if fields else {"raw_text_excerpt": raw_text[:1500]}),
        "_note": "Ollama structuring was unavailable or timed out — showing heuristic extraction. "
                 "For full structured JSON run 'ollama serve' and pull a model (e.g. 'ollama pull qwen2.5:7b').",
    }


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def get_mime(ext):
    return {"pdf":"application/pdf","png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg",
            "tiff":"image/tiff","tif":"image/tiff","webp":"image/webp","bmp":"image/bmp"}.get(ext.lower(),"application/octet-stream")

def word_count(text): return len(text.split()) if text else 0

def render_pipeline(current, ph):
    html = ""
    for i, name in enumerate(PIPELINE_STEPS):
        cls = "done" if i < current else ("active" if i == current else "")
        tick = "✓" if i < current else ("▸" if i == current else "·")
        html += f'<div class="pstep {cls}"><span class="pnum">{tick}</span>{name}</div>'
    ph.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKEND — Chandra OCR
# ─────────────────────────────────────────────
def run_chandra_ocr(file_bytes, filename, api_key, use_llm=True, structuring_model=None):
    ext = filename.rsplit(".",1)[-1].lower(); mime = get_mime(ext)
    files = {
        "file":           (filename, io.BytesIO(file_bytes), mime),
        "langs":          (None,"English"),
        "force_ocr":      (None,"true"),      # always force OCR for reliable extraction
        "paginate":       (None,"false"),
        "output_format":  (None,"json"),
        "extract_images": (None,"false"),
        "use_llm":        (None,str(use_llm).lower()),
    }
    headers = {"X-Api-Key": api_key}
    r = requests.post(CHANDRA_API_URL, files=files, headers=headers, timeout=120)
    r.raise_for_status()
    result = r.json()

    if "request_check_url" in result:
        for _ in range(90):
            time.sleep(3)
            p = requests.get(result["request_check_url"], headers=headers, timeout=30).json()
            if p.get("status") == "complete": result = p; break
            if p.get("status") == "error": raise RuntimeError(p.get("error","Unknown error"))

    raw_text = (result.get("markdown") or result.get("text") or "").strip()
    raw_json = result.get("json", {})
    content_obj = raw_json if isinstance(raw_json, dict) else {}

    combined_text, table_htmls = collect_text_and_tables(content_obj, raw_text=raw_text)

    if not table_htmls:
        table_htmls = [t.get("html", "") for t in result.get("tables", []) if isinstance(t, dict) and t.get("html")]

    # Structure the OCR output into clean nested JSON. Chandra's own
    # use_llm=True already runs a layout-aware LLM server-side, but its
    # output isn't in our target envelope shape, so we still run it
    # through the same universal structuring step everyone else uses —
    # this guarantees a consistent JSON contract regardless of engine.
    if structuring_model:
        try:
            extracted = structure_with_ollama(combined_text, table_htmls, structuring_model)
        except Exception as e:
            extracted = {"document_type": "Unknown", "_structuring_error": str(e)}
    else:
        extracted = structure_with_claude_fallback(combined_text, table_htmls)

    tables_out = [{"rows": _parse_html_table(h)} for h in table_htmls if _parse_html_table(h)]

    return {
        "model":     "chandra-ocr",
        "filename":  filename,
        "status":    "success",
        "metadata":  {
            "pages":    result.get("page_count", 1),
            "language": result.get("languages", ["en"]),
            "engine":   "Chandra OCR / Datalab Marker",
        },
        "raw_text":        combined_text,
        "tables":          tables_out,
        "forms":           result.get("forms", []),
        "extracted_fields": extracted,
    }


# ─────────────────────────────────────────────
# OLLAMA HELPERS (shared by Qwen vision calls and the structuring step)
# ─────────────────────────────────────────────
def ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def ollama_pulled_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass
    return []

def find_active_qwen():
    have = ollama_pulled_models()
    have_lower = [m.lower() for m in have]
    for pref in QWEN_MODELS_PREF:
        for orig, low in zip(have, have_lower):
            if pref.lower() in low:
                return orig
    for orig, low in zip(have, have_lower):
        if "qwen" in low and "vl" in low:
            return orig
    for orig, low in zip(have, have_lower):
        if "qwen" in low:
            return orig
    return None

def find_structuring_model(vision_model_fallback: str = None) -> str:
    """Picks the best available local model for the text-only JSON
    structuring step. Prefers a dedicated text model (lighter/faster);
    falls back to any vision-capable model already pulled (Ollama VL
    models can also do text-only prompts fine), so Chandra and PaddleOCR
    get the same fallback chain QWEN's own call site already had instead
    of silently returning an unstructured stub when no text model exists."""
    have = ollama_pulled_models()
    have_lower = [m.lower() for m in have]
    for pref in STRUCTURING_MODELS_PREF:
        for orig, low in zip(have, have_lower):
            if pref.lower() in low:
                return orig
    if vision_model_fallback:
        return vision_model_fallback
    return find_active_qwen()


# ─────────────────────────────────────────────
# BACKEND — QWEN 2.5 VL (LOCAL OLLAMA)
# ─────────────────────────────────────────────
def get_free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return -1.0  # unknown — don't block on it

def _pdf_to_image_bytes(file_bytes: bytes) -> bytes:
    """Convert first page of a PDF to PNG bytes. Fully isolated so any
    fitz import/DLL error on Windows never affects non-PDF paths."""
    try:
        import fitz as _fitz
        doc = _fitz.open(stream=file_bytes, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=150)
        return pix.tobytes("png")
    except ImportError:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to process PDF files with Qwen OCR. "
            "Install it with: pip install pymupdf. "
            "On Windows DLL error: pip install --force-reinstall pymupdf. "
            "Then restart the app."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF page with PyMuPDF: {e}")

def _image_to_b64(file_bytes: bytes, ext: str):
    """Convert image (or PDF) bytes to base64 JPEG for Ollama vision calls.
    PDF handling is fully isolated - a Windows fitz DLL error will never
    affect PNG/JPG/WEBP processing."""
    # PDF: render first page to PNG, then fall through to normal image path
    if ext == "pdf":
        file_bytes = _pdf_to_image_bytes(file_bytes)

    # All image formats (PNG, JPG, WEBP, BMP, TIFF + the PNG from a PDF above)
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"Could not open the uploaded file as an image (ext={ext}): {e}. "
            "Supported formats: PNG, JPG, WEBP, BMP, TIFF."
        )
    w, h = img.size
    max_px = 1120
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _ollama_vision_call(model_name: str, prompt: str, b64_data: str,
                        num_predict: int, num_ctx: int) -> str:
    import os
    payload = {
        "model":  model_name,
        "prompt": prompt,
        "images": [b64_data],
        "stream": False,
        "options": {
            "temperature":  0,
            "num_predict":  num_predict,
            "num_ctx":      num_ctx,
            "num_thread":   max(2, (os.cpu_count() or 2)),
            "f16_kv":       True,
            "low_vram":     True,
            "mmap":         True,
        },
    }
    payload["stream"] = True
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=(30, 120),
    )
    r.raise_for_status()
    parts = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            break
    return "".join(parts).strip()

# RAM tiers: if there's plenty of free RAM use a larger context window for
# better accuracy; if RAM is tight, automatically scale down instead of
# hard-failing, and only refuse to run at all below an absolute floor where
# Ollama itself would almost certainly be killed by the OS.
# Reduced context/token sizes vs original to prevent Ollama from running
# out of memory mid-generation (which causes the "connection forcibly closed"
# / wsarecv error on Windows). A smaller context is safer on most laptops.
_RAM_TIERS = [
    (6.0, dict(num_predict=1024, num_ctx=2048)),
    (3.0, dict(num_predict=512,  num_ctx=1536)),
    (1.5, dict(num_predict=256,  num_ctx=1024)),
]
_RAM_HARD_FLOOR_GB = 0.8

def run_qwen_local(file_bytes: bytes, filename: str, model_name: str,
                    structuring_model: str = None) -> dict:
    free_gb = get_free_ram_gb()
    if 0 <= free_gb < _RAM_HARD_FLOOR_GB:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB RAM free. Close other apps and try again, "
            f"or switch to Chandra OCR / PaddleOCR which need less memory."
        )
    # Pick the largest context tier the current free RAM can support.
    call_opts = _RAM_TIERS[-1][1]
    for threshold, opts in _RAM_TIERS:
        if free_gb < 0 or free_gb >= threshold:
            call_opts = opts
            break

    ext = filename.rsplit(".", 1)[-1].lower()
    b64_data, _ = _image_to_b64(file_bytes, ext)

    prompt = (
        "Read this document image carefully. "
        "Write out every single piece of text you can see, exactly as printed. "
        "Include all names, numbers, dates, labels, addresses, and values. "
        "Output just the text, line by line. Nothing else."
    )
    raw_text = _ollama_vision_call(model_name, prompt, b64_data, **call_opts)

    structuring_model = structuring_model or find_structuring_model(vision_model_fallback=model_name)
    try:
        extracted_fields = structure_with_ollama(raw_text, [], structuring_model)
    except Exception as e:
        extracted_fields = {"document_type": "Unknown", "_structuring_error": str(e)}

    return {
        "model":            "qwen2.5-vl",
        "model_variant":    model_name,
        "filename":         filename,
        "status":           "success",
        "metadata":         {
            "engine": "Ollama local",
            "ram_tier_used": call_opts,
            "free_ram_gb_at_run": round(free_gb, 2) if free_gb >= 0 else "unknown",
        },
        "raw_text":         raw_text,
        "tables":           [],
        "forms":            [],
        "key_value_pairs":  {},
        "extracted_fields": extracted_fields,
    }


# ─────────────────────────────────────────────
# BACKEND — PaddleOCR (LOCAL)
#
# WHY HTTP SERVER APPROACH:
# PaddlePaddle's C++ engine loads into memory the moment paddleocr is
# imported — in ANY process that shares the same OS session. Once loaded,
# it permanently refuses to initialize again ("PDX already initialized").
# multiprocessing.Process, @st.cache_resource, session_state — none of
# these work because they all still import paddleocr inside the same OS
# process tree that Streamlit uses.
#
# SOLUTION: paddle_worker.py is a tiny Flask HTTP server that you start
# ONCE in a separate terminal. It owns PaddlePaddle entirely. The main
# app calls it via HTTP — zero imports of paddleocr in the Streamlit
# process. The worker never restarts, so PDX never reinitializes.
# ─────────────────────────────────────────────

PADDLE_WORKER_URL = "http://localhost:5050"

def _get_paddle_version() -> tuple:
    try:
        import paddleocr as _poc
        ver = getattr(_poc, "__version__", "2.0.0")
        parts = ver.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return 2, 7

def paddle_worker_running() -> bool:
    """Check if paddle_worker.py HTTP server is up."""
    try:
        r = requests.get(f"{PADDLE_WORKER_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def _run_paddle_via_http(img_bytes: bytes, filename: str) -> list:
    """POST image bytes to the paddle worker server, get (text,y,x) list back."""
    files = {"image": (filename, img_bytes, "application/octet-stream")}
    r = requests.post(f"{PADDLE_WORKER_URL}/ocr", files=files, timeout=120)
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "Unknown paddle worker error"))
    return [(item["text"], item["y"], item["x"]) for item in data.get("items", [])]
def run_paddle_ocr(file_bytes: bytes, filename: str, structuring_model: str = None) -> dict:
    ext = filename.rsplit(".", 1)[-1].lower()

    if not paddle_worker_running():
        raise RuntimeError(
            "PaddleOCR worker is not running.\n"
            "Open a NEW terminal, activate your venv, and run:\n"
            "    python paddle_worker.py\n"
            "Keep that terminal open, then click Run OCR again."
        )

    # Send raw bytes to the worker — it handles PDF/image conversion internally
    items = _run_paddle_via_http(file_bytes, filename)

    # Reconstruct reading order top-to-bottom, left-to-right within row bands
    items_sorted = sorted(items, key=lambda it: (round(it[1] / 12), it[2]))
    raw_text = "\n".join(t for t, _, _ in items_sorted)

    structuring_model = structuring_model or find_structuring_model()
    try:
        extracted_fields = structure_with_ollama(raw_text, [], structuring_model) if structuring_model else \
            structure_with_claude_fallback(raw_text, [])
    except Exception as e:
        extracted_fields = {"document_type": "Unknown", "_structuring_error": str(e)}

    return {
        "model":            "paddleocr",
        "filename":         filename,
        "status":           "success",
        "metadata":         {"engine": "PaddleOCR local (subprocess worker)"},
        "raw_text":         raw_text,
        "tables":           [],
        "forms":            [],
        "key_value_pairs":  {},
        "extracted_fields": extracted_fields,
    }


# ═══════════════════════════════════════════════════════════════════
# AUTH GATE — nothing below runs until signed in
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    render_login()
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []   # list of {"id","filename","model_label","result","elapsed","ts"}
if "active_page" not in st.session_state:
    st.session_state.active_page = "ocr"
if "result"  not in st.session_state: st.session_state.result  = None
if "elapsed" not in st.session_state: st.session_state.elapsed = None
if "err"     not in st.session_state: st.session_state.err     = None

with st.sidebar:
    st.markdown('<div class="sb-title">◆ OCR IDP Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Document Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Workspace</div>', unsafe_allow_html=True)
    nav_class = "nav-active" if st.session_state.active_page == "ocr" else ""
    st.markdown(f'<div class="{nav_class}">', unsafe_allow_html=True)
    if st.button("New Extraction", use_container_width=True, key="nav_new"):
        st.session_state.active_page = "ocr"
        st.session_state.result = None
        st.session_state.err = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div style="font-size:0.78rem;color:#9aa1ad;padding:0 0.2rem;">No extractions yet</div>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history):
            c1, c2 = st.columns([5, 1])
            st.markdown('<div class="hist-row">', unsafe_allow_html=True)
            with c1:
                label = item["filename"]
                label = (label[:22] + "…") if len(label) > 22 else label
                if st.button(label, key=f"hist_open_{item['id']}", use_container_width=True):
                    st.session_state.result = item["result"]
                    st.session_state.elapsed = item["elapsed"]
                    st.session_state.err = None
                    st.session_state.active_page = "ocr"
                    st.rerun()
            with c2:
                st.markdown('<div class="hist-del">', unsafe_allow_html=True)
                if st.button("✕", key=f"hist_del_{item['id']}"):
                    st.session_state.history = [h for h in st.session_state.history if h["id"] != item["id"]]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sb-user">Signed in as <b>{st.session_state.username}</b></div>',
        unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True, key="sign_out"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

st.markdown("""
<div class="brand">
    <h1>OCR IDP Application</h1>
    <p>AI-Powered Document Intelligence Platform · Structured JSON Output</p>
</div>
""", unsafe_allow_html=True)

st.markdown("#### Configuration")

# ── Added: engine selector ──────────────────────────────────────
st.markdown('<div class="ctrl-label">OCR Engine</div>', unsafe_allow_html=True)
model_label    = st.selectbox("OCR Engine", list(MODELS.keys()), label_visibility="collapsed")
selected_model = MODELS[model_label]

_ollama_ok   = ollama_running() if selected_model == "qwen2.5-vl" else False
_active_qwen = find_active_qwen() if _ollama_ok else None

if selected_model == "chandra-ocr":
    st.markdown('<div class="ctrl-label">Datalab API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("Datalab API Key", type="password",
                            placeholder="dl-xxxxxxxxxxxx", label_visibility="collapsed")
    use_llm = True
    qwen_variant = None

elif selected_model == "qwen2.5-vl":
    api_key = ""
    use_llm = True
    if not _ollama_ok:
        st.markdown('<div class="err-box">Ollama not running.<br>Run: <b>ollama serve</b></div>', unsafe_allow_html=True)
        qwen_variant = None
    elif not _active_qwen:
        st.markdown('<div class="warn-box">No Qwen2.5-VL model pulled.<br>Run: <b>ollama pull qwen2.5vl:7b</b><br>or: <b>ollama pull qwen2.5vl:3b</b></div>', unsafe_allow_html=True)
        qwen_variant = None
        with st.expander("Debug: models Ollama reports"):
            _debug_models = ollama_pulled_models()
            if _debug_models:
                st.write(_debug_models)
            else:
                st.write("Ollama returned an empty model list, or the request failed silently.")
    else:
        qwen_variant = _active_qwen
        _free_gb = get_free_ram_gb()
        if _free_gb < 0:
            st.markdown(f'<div class="ok-box">Local Ollama ready · Model: <b>{_active_qwen}</b> · No API key required</div>', unsafe_allow_html=True)
        elif _free_gb < _RAM_HARD_FLOOR_GB:
            st.markdown(f'<div class="err-box">Only {_free_gb:.1f} GB RAM free.<br>Close other apps before running — Ollama is likely to fail to load the model at this level.</div>', unsafe_allow_html=True)
        elif _free_gb < 3.0:
            st.markdown(f'<div class="warn-box">Ollama ready · Model: <b>{_active_qwen}</b><br>{_free_gb:.1f} GB RAM free — running at a reduced context size for stability. Close other apps for best accuracy.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box">Local Ollama ready · Model: <b>{_active_qwen}</b> · {_free_gb:.1f} GB RAM free · No API key required</div>', unsafe_allow_html=True)

    # Added: manual override — lets you type the exact `ollama list` name
    # if auto-detection above didn't find it for any reason.
    _manual_qwen = st.text_input(
        "Or type exact Ollama model name (optional override)",
        value="", placeholder="e.g. qwen2.5vl:3b",
        key="manual_qwen_override"
    )
    if _manual_qwen.strip():
        qwen_variant = _manual_qwen.strip()

else:  # paddleocr
    api_key = ""
    use_llm = True
    qwen_variant = None
    if _paddle_worker_status in ("started", "already_running"):
        st.markdown('<div class="ok-box">PaddleOCR ready — worker started automatically</div>', unsafe_allow_html=True)
    elif _paddle_worker_status == "failed":
        st.markdown('<div class="err-box">PaddleOCR worker failed to start. Check that paddleocr and flask are installed:<br><b>pip install paddleocr paddlepaddle flask</b></div>', unsafe_allow_html=True)
    elif _paddle_worker_status == "timeout":
        st.markdown('<div class="warn-box">PaddleOCR worker is still loading (downloading model). Wait 30s and try again.</div>', unsafe_allow_html=True)
    else:
        if paddle_worker_running():
            st.markdown('<div class="ok-box">PaddleOCR worker running</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="err-box">PaddleOCR worker not responding. Restart the app.</div>', unsafe_allow_html=True)

# Both Chandra and PaddleOCR hand their raw OCR text to a local Ollama
# model for the JSON-structuring step. Surface that dependency clearly
# instead of silently returning unstructured output if Ollama is down
# OR if Ollama is up but has no usable model pulled at all.
if selected_model in ("chandra-ocr", "paddleocr"):
    if not ollama_running():
        st.markdown(
            '<div class="warn-box">Ollama isn\'t running — structured JSON extraction needs it. '
            'Run <b>ollama serve</b> and pull a text model (e.g. <b>ollama pull qwen2.5:7b</b>) for best results. '
            'OCR will still run, but the output will fall back to raw text only.</div>',
            unsafe_allow_html=True)
    elif not find_structuring_model():
        st.markdown(
            '<div class="err-box">Ollama is running but no usable model is pulled. '
            'Run <b>ollama pull qwen2.5:7b</b> (or any text/vision model) before running OCR, '
            'otherwise the output will fall back to raw text only with no structured fields.</div>',
            unsafe_allow_html=True)

st.markdown("---")
st.markdown("#### Upload Document")
up_col, run_col = st.columns([2, 1], gap="large")

with up_col:
    uploaded = st.file_uploader(
        "Drop your document — PDF, PNG, JPG, JPEG, TIFF, WEBP, BMP",
        type=SUPPORTED, label_visibility="collapsed",
    )
    if uploaded:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        fb  = uploaded.getvalue()
        st.markdown(f"""<div style='font-size:0.78rem;color:#6b7280;margin-top:4px'>
        <b style='color:#16181d'>{uploaded.name}</b>
        &nbsp;·&nbsp; {len(fb)/1024:.1f} KB &nbsp;·&nbsp; .{ext.upper()}
        </div>""", unsafe_allow_html=True)
        if ext != "pdf":
            try:
                st.image(Image.open(io.BytesIO(fb)), use_container_width=True, caption=uploaded.name)
            except Exception:
                pass
        else:
            st.markdown("""<div style='text-align:center;padding:1.5rem;background:#fafbfc;
                border:1px solid #e2e5eb;border-radius:10px;margin-top:0.5rem'>
                <div style='font-family:JetBrains Mono,monospace;color:#2952e3;font-size:0.82rem;font-weight:600'>
                    PDF ready for processing</div></div>""", unsafe_allow_html=True)

with run_col:
    st.markdown("#### Pipeline")
    pipeline_ph = st.empty()
    render_pipeline(-1, pipeline_ph)
    st.markdown("<br>", unsafe_allow_html=True)

    if selected_model == "chandra-ocr":
        run_disabled = (uploaded is None) or (not api_key)
    elif selected_model == "qwen2.5-vl":
        run_disabled = (uploaded is None) or (not _ollama_ok) or (not qwen_variant)
    else:  # paddleocr
        run_disabled = (uploaded is None) or (not paddle_worker_running())

    run_btn = st.button("Run OCR", disabled=run_disabled)

    if uploaded is None:
        st.markdown('<div class="warn-box">Upload a file to enable</div>', unsafe_allow_html=True)
    elif selected_model == "chandra-ocr" and not api_key:
        st.markdown('<div class="warn-box">Enter Datalab API key above</div>', unsafe_allow_html=True)
    elif selected_model == "qwen2.5-vl" and not _ollama_ok:
        st.markdown('<div class="err-box">Start Ollama first</div>', unsafe_allow_html=True)
    elif selected_model == "qwen2.5-vl" and not qwen_variant:
        st.markdown('<div class="warn-box">Pull a Qwen2.5-VL model first, or type its exact name above</div>', unsafe_allow_html=True)
    elif selected_model == "qwen2.5-vl" and uploaded and uploaded.name.rsplit(".",1)[-1].lower() == "pdf":
        _fitz_ok = False
        try:
            import fitz as _fitz_check  # noqa: F401
            _fitz_ok = True
        except Exception:
            pass
        if not _fitz_ok:
            st.markdown(
                '<div class="warn-box">PDF support for Qwen requires PyMuPDF.<br>'
                'Install: <b>pip install pymupdf</b><br>'
                'On Windows DLL error: <b>pip install --force-reinstall pymupdf</b></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn and uploaded:
    st.session_state.result = None
    st.session_state.err    = None
    fb    = uploaded.getvalue()
    fname = uploaded.name
    t0    = time.time()
    status_ph = st.empty()

    try:
        for step_i in range(len(PIPELINE_STEPS)):
            render_pipeline(step_i, pipeline_ph)
            status_ph.markdown(
                f'<div class="pill run">{PIPELINE_STEPS[step_i]}…</div>',
                unsafe_allow_html=True)
            time.sleep(0.18)
            if step_i == 4:
                if selected_model == "chandra-ocr":
                    _structuring_model = find_structuring_model() if ollama_running() else None
                    result = run_chandra_ocr(fb, fname, api_key, use_llm=use_llm,
                                             structuring_model=_structuring_model)
                elif selected_model == "qwen2.5-vl":
                    variant = qwen_variant
                    result  = run_qwen_local(fb, fname, variant)
                else:  # paddleocr
                    _structuring_model = find_structuring_model() if ollama_running() else None
                    result = run_paddle_ocr(fb, fname, structuring_model=_structuring_model)

        render_pipeline(len(PIPELINE_STEPS), pipeline_ph)
        st.session_state.result  = result
        st.session_state.elapsed = round(time.time() - t0, 2)
        status_ph.markdown(
            f'<div class="pill ok">Done in {st.session_state.elapsed}s</div>',
            unsafe_allow_html=True)

        # Added: record this run in sidebar history so it can be reopened later.
        import uuid as _uuid
        st.session_state.history.append({
            "id": _uuid.uuid4().hex[:8],
            "filename": fname,
            "model_label": model_label,
            "result": result,
            "elapsed": st.session_state.elapsed,
            "ts": time.time(),
        })

    except requests.exceptions.HTTPError as e:
        st.session_state.err = f"API Error {e.response.status_code}:\n{e.response.text[:500]}"
    except Exception as e:
        st.session_state.err = str(e)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
st.markdown("#### Extraction Results")

if st.session_state.err:
    st.markdown(f'<div class="err-box">Error:\n{st.session_state.err}</div>', unsafe_allow_html=True)

elif st.session_state.result:
    res      = st.session_state.result
    raw_text = res.get("raw_text", "")
    tables   = res.get("tables",  [])
    forms    = res.get("forms",   [])
    kv               = res.get("key_value_pairs", {})
    extracted_fields  = res.get("extracted_fields", {})

    if res["model"] == "chandra-ocr":
        m_name = "CHANDRA OCR"
    elif res["model"] == "qwen2.5-vl":
        m_name = f"QWEN 2.5 VL · {res.get('model_variant','local')}"
    else:
        m_name = "PADDLEOCR"

    _doc_type = extracted_fields.get("document_type", "") if isinstance(extracted_fields, dict) else ""
    _badge_text = f"{m_name} &nbsp;·&nbsp; {res['filename']}"
    if _doc_type:
        _badge_text += f" &nbsp;·&nbsp; {_doc_type}"
    st.markdown(f'<div class="mbadge">{_badge_text}</div>', unsafe_allow_html=True)

    if isinstance(extracted_fields, dict) and extracted_fields.get("_structuring_error"):
        st.markdown(
            f'<div class="warn-box">JSON structuring step had an issue: '
            f'{extracted_fields["_structuring_error"]}<br>Raw OCR text is still available below.</div>',
            unsafe_allow_html=True)
    elif isinstance(extracted_fields, dict) and extracted_fields.get("_note"):
        st.markdown(
            f'<div class="err-box">{extracted_fields["_note"]}</div>',
            unsafe_allow_html=True)

    def _count_fields(obj):
        if isinstance(obj, dict):
            return sum(_count_fields(v) for v in obj.values())
        elif isinstance(obj, list):
            return len(obj)
        elif obj and obj != "":
            return 1
        return 0

    if isinstance(extracted_fields, dict):
        field_count = _count_fields(extracted_fields)
    else:
        field_count = len(forms) or len(kv)

    st.markdown(f"""
    <div class="mrow">
        <div class="mbox"><div class="mval">{word_count(raw_text)}</div><div class="mlbl">Words</div></div>
        <div class="mbox"><div class="mval">{len(tables)}</div><div class="mlbl">Tables</div></div>
        <div class="mbox"><div class="mval">{field_count}</div><div class="mlbl">Fields</div></div>
        <div class="mbox"><div class="mval">{st.session_state.elapsed}s</div><div class="mlbl">Time</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Extracted Data", "Full JSON", "Download"])

    with tab1:
        ef = extracted_fields

        def _html_table(rows_of_dicts):
            """Render a list of dicts as a plain HTML table — no JS/Arrow needed."""
            if not rows_of_dicts:
                return ""
            headers = list(rows_of_dicts[0].keys())
            hdr = "".join(f"<th style='text-align:left;padding:6px 12px;border-bottom:2px solid #e2e5eb;font-size:0.78rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px'>{h}</th>" for h in headers)
            body = ""
            for i, row in enumerate(rows_of_dicts):
                bg = "#fafbfc" if i % 2 == 0 else "#ffffff"
                cells = "".join(f"<td style='padding:6px 12px;border-bottom:1px solid #f0f1f3;font-size:0.85rem;color:#16181d'>{str(row.get(h,''))}</td>" for h in headers)
                body += f"<tr style='background:{bg}'>{cells}</tr>"
            return f"<div style='overflow-x:auto;border:1px solid #e2e5eb;border-radius:8px;margin:0.5rem 0'><table style='width:100%;border-collapse:collapse'><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table></div>"

        def _kv_table(pairs):
            """Render key-value pairs as a two-column HTML table."""
            rows = [{"Field": k, "Value": v} for k, v in pairs]
            return _html_table(rows)

        def _render_ef(ef, depth=0):
            """Recursively render extracted_fields dict as plain HTML tables."""
            if not isinstance(ef, dict) or not ef:
                return
            flat_rows = []
            for k, v in ef.items():
                if depth == 0 and k in ("document_type", "_structuring_error", "_raw_model_output", "_note"):
                    continue
                label = k.replace("_", " ").title()
                if isinstance(v, str) and v:
                    flat_rows.append({"Field": label, "Value": v})
                elif isinstance(v, (int, float)):
                    flat_rows.append({"Field": label, "Value": str(v)})
                elif isinstance(v, dict) and v:
                    if depth == 0:
                        st.markdown(f"**{label}**")
                    _render_ef(v, depth + 1)
                elif isinstance(v, list) and v:
                    if isinstance(v[0], dict):
                        st.markdown(f"**{label}**")
                        try:
                            cols = list(v[0].keys())
                            rows = [{c.replace("_"," ").title(): str(r.get(c,"")) for c in cols} for r in v]
                            st.markdown(_html_table(rows), unsafe_allow_html=True)
                        except Exception:
                            st.text(str(v)[:500])
                    elif isinstance(v[0], str):
                        flat_rows.append({"Field": label, "Value": ", ".join(v)})
            if flat_rows:
                st.markdown(_html_table(flat_rows), unsafe_allow_html=True)

        if isinstance(ef, dict) and ef:
            _render_ef(ef)
        elif forms:
            st.markdown(_html_table([{"Field": f.get("label",""), "Value": f.get("value","")} for f in forms]), unsafe_allow_html=True)
        elif kv:
            st.markdown(_kv_table(list(kv.items())), unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">No structured fields found — check the Full JSON tab for raw_text.</div>', unsafe_allow_html=True)

        # Show extracted tables if any
        if tables:
            st.markdown("**Extracted Tables**")
            for tbl in tables:
                rows = tbl.get("rows", tbl) if isinstance(tbl, dict) else tbl
                lbl  = tbl.get("label", "") if isinstance(tbl, dict) else ""
                if lbl:
                    st.markdown(f"*{lbl}*")
                if rows and isinstance(rows, list):
                    try:
                        if isinstance(rows[0], dict):
                            st.markdown(_html_table(rows), unsafe_allow_html=True)
                        elif isinstance(rows[0], list) and len(rows) > 1:
                            cols = [str(c) for c in rows[0]]
                            data = [{cols[i]: str(cell) for i, cell in enumerate(r)} for r in rows[1:]]
                            st.markdown(_html_table(data), unsafe_allow_html=True)
                    except Exception:
                        st.text(str(rows)[:500])

        if raw_text:
            with st.expander("Raw OCR text"):
                st.text(raw_text[:8000] + ("…" if len(raw_text) > 8000 else ""))

    with tab2:
        st.code(json.dumps(res, indent=2, ensure_ascii=False), language="json")

    with tab3:
        jbytes = json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8")
        bname  = res["filename"].rsplit(".",1)[0] + "_ocr.json"
        st.download_button("Download JSON", data=jbytes,
                           file_name=bname, mime="application/json", use_container_width=True)
        st.markdown(f"""<div style='font-size:0.74rem;color:#6b7280;margin-top:6px'>
        {bname} &nbsp;·&nbsp; {len(jbytes)/1024:.1f} KB</div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="rcard empty-state">
        Upload a document above and click <b>Run OCR</b> to see results here.
    </div>""", unsafe_allow_html=True)