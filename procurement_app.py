"""
BIM Procurement Assistant — Streamlit App
TFM Tool: BIM Schedule → Procurement Pipeline + AI Strategy
Author: Lucía Varona Vidaurrazaga | IIT Spring 2026
Run with: streamlit run procurement_app.py
"""

import streamlit as st
import pandas as pd
import io
import re
import zipfile
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIM Procurement Assistant",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .ai-box {
        background: #fffdf0; border-left: 5px solid #f1c40f;
        padding: 20px; border-radius: 8px;
        white-space: pre-wrap; font-family: monospace; font-size: 0.9rem;
    }
    .info-box {
        background: #e8f4fd; border-left: 5px solid #3498db;
        padding: 12px 16px; border-radius: 6px;
        font-size: 0.88rem; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _s(v) -> str:
    """Safe string: NaN → empty, always stripped."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _is_subtotal(val: str) -> bool:
    """Detect BIM subtotal rows: 'FLOOR: 12', ': 1', 'Grand total: 1618'."""
    return bool(re.search(r':\s*\d+\s*$', val.strip(), re.IGNORECASE))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — READ RAW CSV (handles any BIM export)
# ══════════════════════════════════════════════════════════════════════════════

def _read_raw(uploaded_file) -> pd.DataFrame:
    """
    Reads any BIM CSV regardless of:
      • Column count (5 cols Revit EE vs 36 cols Gurtz Panelboard)
      • UTF-8 BOM  •  CRLF / LF  •  First row shorter than rest
    Returns a DataFrame with integer column indices.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    # Decode
    if raw[:3] == b'\xef\xbb\xbf':
        text = raw.decode('utf-8-sig')
    else:
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                text = raw.decode(enc); break
            except UnicodeDecodeError:
                continue

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [l for l in text.splitlines() if l.strip()]

    # Pre-detect max columns so no row is truncated
    max_cols = max(l.count(',') + 1 for l in lines) if lines else 5

    buf = io.StringIO(text)
    df = pd.read_csv(
        buf,
        header=None,
        names=list(range(max_cols)),
        dtype=str,
        on_bad_lines='skip',
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DETECT SCHEMA  (full-pass, no early-exit on generic rows)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_schema(df: pd.DataFrame):
    """
    Scans ALL of the first 20 rows before deciding.
    Priority: revit_ee  >  panelboard  >  generic.
    Returns (header_row_index, schema_str).
    """
    scan = min(20, len(df))
    generic_idx = None  # first generic-looking row, used only as last resort

    for i in range(scan):
        row = [_s(v).lower() for v in df.iloc[i]]

        # ── Revit Electrical Equipment Schedule ──────────────────────────
        if 'family' in row and 'count' in row:
            return i, 'revit_ee'

        # ── Panelboard / multi-column BIM schedule ────────────────────────
        hits = sum([
            'level'      in row,
            'panel name' in row,
            'type'       in row,
            'location'   in row,
            'supply from' in row,
        ])
        if hits >= 3:
            return i, 'panelboard'

        # ── Generic fallback candidate (save, don't return yet) ───────────
        non_empty = [v for v in row if v and v != 'nan']
        if generic_idx is None and len(non_empty) >= 4 and i > 0:
            if not re.match(r'^\d', non_empty[0]):
                generic_idx = i

    return (generic_idx, 'generic') if generic_idx is not None else (None, 'unknown')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a — PARSE: REVIT ELECTRICAL EQUIPMENT SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

REVIT_COLS = ["Family", "Type", "Panel Name", "Level", "Count"]

def _parse_revit_ee(df: pd.DataFrame, hdr: int):
    header = [_s(v) for v in df.iloc[hdr]]

    # Map expected columns to their positional index
    col_map = {}
    for target in REVIT_COLS:
        for j, h in enumerate(header):
            if h.lower() == target.lower():
                col_map[target] = j
                break

    df_data = df.iloc[hdr + 1:].reset_index(drop=True)

    work = pd.DataFrame()
    for name in REVIT_COLS:
        if name in col_map:
            work[name] = df_data[col_map[name]].apply(_s)
        else:
            work[name] = ""

    # Count must be numeric and positive
    work["Count"] = pd.to_numeric(work["Count"], errors='coerce')
    work = work[work["Count"].notna() & (work["Count"] > 0)].copy()
    work["Count"] = work["Count"].astype(int)

    # Remove Revit subtotal rows: Type + Panel Name + Level all empty
    subtotal = (work["Type"] == "") & (work["Panel Name"] == "") & (work["Level"] == "")
    work = work[~subtotal & (work["Family"] != "")].reset_index(drop=True)

    return _make_outputs(work, family_col="Family")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b — PARSE: PANELBOARD / MULTI-COLUMN BIM SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_panelboard(df: pd.DataFrame, hdr: int):
    header = [_s(v) for v in df.iloc[hdr]]

    def find(names):
        for name in names:
            for j, h in enumerate(header):
                if h.strip().lower() == name.lower():
                    return j
        return None

    idx = {
        'level':    find(["Level"]),
        'location': find(["Location"]),
        'closet':   find(["Closet Type"]),
        'panel':    find(["Panel Name"]),
        'supply':   find(["Supply From"]),
        'type':     find(["Type"]),
        'tub':      find(["Tub Number"]),
        'mcb':      find(["MCB Rating"]),
        'mains':    find(["Mains Type"]),
        'voltage':  find(["Voltage"]),
    }

    def g(row, key):
        i = idx.get(key)
        if i is None or i >= len(row):
            return ""
        return _s(row.iloc[i])

    rows = []
    for _, row in df.iloc[hdr + 1:].iterrows():
        level = g(row, 'level')
        panel = g(row, 'panel')
        ptype = g(row, 'type')

        # Skip blank rows
        if not level and not panel and not ptype:
            continue
        # Skip subtotal rows like "PLAZA LEVEL: 144" or ": 1"
        if _is_subtotal(level):
            continue
        # Skip section-header rows (level present but no panel/type)
        if level and not panel and not ptype:
            continue
        # Need at least type OR panel
        if not panel and not ptype:
            continue
        # Skip grand total row
        if re.search(r'grand total', level, re.IGNORECASE):
            continue

        rows.append({
            "Level":       re.sub(r':\s*\d+\s*$', '', level).strip(),
            "Location":    g(row, 'location'),
            "Closet Type": g(row, 'closet'),
            "Panel Name":  panel,
            "Supply From": g(row, 'supply'),
            "Type":        ptype,
            "Tub Number":  g(row, 'tub'),
            "MCB Rating":  g(row, 'mcb'),
            "Mains Type":  g(row, 'mains'),
            "Voltage":     g(row, 'voltage'),
            "Count":       1,
        })

    if not rows:
        return pd.DataFrame(), pd.DataFrame(columns=["Family", "Count"]), pd.DataFrame(), 0

    work = pd.DataFrame(rows)
    return _make_outputs(work, family_col="Type")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3c — PARSE: GENERIC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _parse_generic(df: pd.DataFrame, hdr: int):
    header = [_s(v) for v in df.iloc[hdr]]
    df_data = df.iloc[hdr + 1:].copy()
    df_data.columns = (header + [""] * max(0, len(df_data.columns) - len(header)))[:len(df_data.columns)]
    df_data = df_data.dropna(how='all').reset_index(drop=True)

    # Rename columns to standard names where possible
    rmap = {}
    for col in df_data.columns:
        cl = str(col).lower()
        if 'family' in cl and 'Family' not in rmap.values():          rmap[col] = 'Family'
        elif ('type' in cl or 'model' in cl) and 'Type' not in rmap.values(): rmap[col] = 'Type'
        elif 'panel' in cl and 'Panel Name' not in rmap.values():     rmap[col] = 'Panel Name'
        elif 'level' in cl and 'Level' not in rmap.values():          rmap[col] = 'Level'
        elif ('count' in cl or 'qty' in cl) and 'Count' not in rmap.values(): rmap[col] = 'Count'
    df_data = df_data.rename(columns=rmap)

    # Ensure required columns exist
    for col in ['Family', 'Type', 'Panel Name', 'Level', 'Count']:
        if col not in df_data.columns:
            df_data[col] = ""

    # If no Count column found, each row = 1 unit
    if df_data["Count"].eq("").all():
        df_data["Count"] = 1
    else:
        df_data["Count"] = pd.to_numeric(df_data["Count"], errors='coerce').fillna(1).astype(int)

    # Use first non-empty text column as Family if still empty
    if df_data["Family"].eq("").all():
        for col in df_data.columns:
            if col not in ['Type', 'Panel Name', 'Level', 'Count', 'Issues']:
                non_empty = df_data[col].apply(_s)
                if non_empty.ne("").any():
                    df_data["Family"] = non_empty
                    break

    # Remove rows where all identity fields are empty
    mask = (
        df_data["Family"].apply(_s).ne("") |
        df_data["Panel Name"].apply(_s).ne("") |
        df_data["Type"].apply(_s).ne("")
    )
    work = df_data[mask].reset_index(drop=True)

    if work.empty:
        return work, pd.DataFrame(columns=["Family","Count"]), pd.DataFrame(), 0

    family_col = "Family" if work["Family"].apply(_s).ne("").any() else "Type"
    return _make_outputs(work, family_col=family_col)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD OUTPUTS (deterministic — no AI arithmetic here)
# ══════════════════════════════════════════════════════════════════════════════

def _make_outputs(work: pd.DataFrame, family_col: str):
    """
    Builds df_clean, df_family, df_issues, total_units.
    All arithmetic done here in Python — never by the LLM.
    """
    if work.empty:
        return work, pd.DataFrame(columns=["Family","Count"]), pd.DataFrame(), 0

    # Ensure Count is numeric
    work = work.copy()
    work["Count"] = pd.to_numeric(work["Count"], errors='coerce').fillna(0).astype(int)
    work = work[work["Count"] > 0].reset_index(drop=True)

    if work.empty:
        return work, pd.DataFrame(columns=["Family","Count"]), pd.DataFrame(), 0

    # Data quality issues
    def issues(row):
        errs = []
        if _s(row.get("Type", ""))       == "":  errs.append("Type undefined")
        if _s(row.get("Panel Name", "")) == "":  errs.append("Panel not assigned")
        if _s(row.get("Level", ""))      == "":  errs.append("Level missing")
        return "; ".join(errs)

    work["Issues"] = work.apply(issues, axis=1)
    df_issues = work[work["Issues"] != ""].copy()

    # Family summary — Python sum, integer, verified
    if family_col not in work.columns:
        family_col = work.columns[0]

    df_family = (
        work.groupby(family_col, dropna=False)["Count"]
        .sum()
        .reset_index()
        .rename(columns={family_col: "Family"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    df_family["Count"] = df_family["Count"].astype(int)

    # Total — single source of truth
    total_units = int(work["Count"].sum())

    return work, df_family, df_issues, total_units


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def clean_bim_csv(uploaded_file):
    df_raw = _read_raw(uploaded_file)
    hdr, schema = _detect_schema(df_raw)

    if hdr is None:
        st.error(
            "❌ No valid header found in the first 20 rows. "
            "Please upload a CSV exported from Autodesk Revit or another BIM tool."
        )
        st.stop()

    if schema == 'revit_ee':
        label  = "Revit Electrical Equipment Schedule"
        result = _parse_revit_ee(df_raw, hdr)
    elif schema == 'panelboard':
        label  = "Panelboard / BIM Multi-Column Schedule"
        result = _parse_panelboard(df_raw, hdr)
    else:
        label  = "Generic BIM Schedule (auto-mapped)"
        result = _parse_generic(df_raw, hdr)

    df_clean, df_family, df_issues, total_units = result
    return df_clean, df_family, df_issues, total_units, label


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER  (v3 prompt — zero hallucinations design)
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(total_units, n_rows, n_issues, df_family, schema_label):
    lines = []
    check = 0
    for _, r in df_family.iterrows():
        q = int(r["Count"])
        check += q
        lines.append(f"  • {r['Family']}: {q} units")
    family_data = "\n".join(lines)

    return f"""You are an office-based Senior Procurement Engineer on an MEP/Electrical project.
You received a validated BOM from a {schema_label} processed by a deterministic Python pipeline.

═══ VALIDATED DATA — READ-ONLY, DO NOT RECALCULATE ═══
• Total equipment units  : {total_units}
• BOM line items         : {n_rows}
• Data quality issues    : {n_issues}
• Pipeline cross-check   : {check}  ← verified by Python (must equal Total above)

Equipment inventory (exact quantities — copy as-is, never modify or add up):
{family_data}

═══ TASKS ═══

**1. RFQ LOTS**
Group families into logical procurement batches by supplier specialisation.
RULES (violation makes output unusable):
  (a) MUTUALLY EXCLUSIVE — each family in ONE lot only, never duplicated.
  (b) EXACT QUANTITIES — copy from inventory above. No totals, no subtotals, no additions.

**2. 4-WEEK OFFICE ACTION PLAN**
  Week 1: RFQ prep + supplier shortlisting{f' + escalate {n_issues} Data Quality issues to BIM Manager' if n_issues > 0 else ''}
  Week 2: RFQ issue + supplier follow-up
  Week 3: Quotation analysis + negotiation
  Week 4: PO placement + floor-level delivery scheduling
  No site visits. Data quality issues → BIM Manager corrects in Revit.

**3. TOP 3 RISKS + MITIGATIONS**
Focus on: lead times, supplier capacity, BIM data gaps blocking orders, delivery sequencing.

═══ ABSOLUTE RULES ═══
• No recalculation of any number
• No tables — bullet points only
• No site visits
• Professional tone
"""


def call_ai(provider, api_key, model, prompt):
    if provider == "Ollama (local)":
        try:
            import ollama
            r = ollama.chat(model=model or "llama3",
                            messages=[{"role":"user","content":prompt}])
            return r["message"]["content"]
        except Exception as e:
            return f"[Ollama error] {e}"

    if provider == "Groq (free)":
        try:
            from openai import OpenAI
            c = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            r = c.chat.completions.create(
                model=model or "llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                max_tokens=1800, temperature=0.2)
            return r.choices[0].message.content
        except Exception as e:
            return f"[Groq error] {e}"

    if provider == "OpenAI":
        try:
            from openai import OpenAI
            c = OpenAI(api_key=api_key)
            r = c.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=1800, temperature=0.2)
            return r.choices[0].message.content
        except Exception as e:
            return f"[OpenAI error] {e}"

    if provider == "Claude (Anthropic)":
        try:
            import anthropic
            c = anthropic.Anthropic(api_key=api_key)
            m = c.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=1800,
                messages=[{"role":"user","content":prompt}])
            return m.content[0].text
        except Exception as e:
            return f"[Anthropic error] {e}"

    return "No AI provider configured."


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_excel(df_clean, df_family, df_issues):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_clean.drop(columns=["Issues"], errors="ignore").to_excel(w, sheet_name="Clean_BOM", index=False)
        df_family.to_excel(w, sheet_name="Summary_by_Family", index=False)
        df_issues.to_excel(w, sheet_name="Data_Quality_Log", index=False)
    return buf.getvalue()


def build_word(df_clean, df_family, df_issues, total_units, ai_strategy, label):
    from docx import Document
    doc = Document()
    doc.add_heading("Procurement Engineering Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph(f"Source format: {label}")
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        f"Total equipment units: {total_units}\n"
        f"BOM line items: {len(df_clean)}\n"
        f"Data quality issues: {len(df_issues)}\n"
        f"Equipment families/types: {len(df_family)}"
    )
    doc.add_heading("2. AI-Generated Procurement Strategy", level=1)
    doc.add_paragraph(ai_strategy)
    doc.add_heading("3. Summary by Equipment Family / Type", level=1)
    for _, r in df_family.iterrows():
        doc.add_paragraph(f"{r['Family']}: {int(r['Count'])} units", style="List Bullet")
    doc.add_heading("4. Detailed BOM", level=1)
    for _, r in df_clean.iterrows():
        parts = " | ".join(
            f"{c}: {r[c]}" for c in df_clean.columns
            if c != "Issues" and _s(r.get(c,""))
        )
        doc.add_paragraph(parts, style="List Bullet")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_html(df_clean, df_family, df_issues, total_units, ai_strategy, label):
    fam_li = "".join(
        f"<li><strong>{_s(r['Family'])}</strong>: {int(r['Count'])} units</li>"
        for _, r in df_family.iterrows()
    )
    items_html = "".join(
        "<div class='item-row'>" + " | ".join(
            f"<strong>{c}</strong>: {_s(r.get(c,''))}"
            for c in df_clean.columns
            if c != "Issues" and _s(r.get(c,""))
        ) + "</div>"
        for _, r in df_clean.iterrows()
    )
    iss_li = "".join(
        f"<li>{_s(r.get('Panel Name', r.get('Family','')))} — {_s(r.get('Issues',''))}</li>"
        for _, r in df_issues.iterrows()
    ) if not df_issues.empty else "<li>✅ No issues found.</li>"

    ai_esc = ai_strategy.replace("<","&lt;").replace(">","&gt;")
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Procurement Dashboard</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;margin:40px;background:#f8f9fa;line-height:1.6}}
  h1{{color:#2c3e50}}.card{{background:white;padding:25px;border-radius:12px;
  box-shadow:0 4px 10px rgba(0,0,0,0.05);margin-bottom:25px}}
  .stat-grid{{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap}}
  .stat{{flex:1;min-width:130px;padding:20px;color:white;border-radius:8px;text-align:center}}
  .ai-box{{border-left:5px solid #f1c40f;padding-left:15px;background:#fffdf0;white-space:pre-wrap}}
  .item-row{{padding:8px 4px;border-bottom:1px solid #eee;font-size:.85rem}}
  .item-row:hover{{background:#f1f3f5}}
</style></head><body>
<h1>⚡ BIM Procurement Dashboard</h1>
<p>Generated: {ts} &nbsp; <em>{label}</em></p>
<div class="stat-grid">
  <div class="stat" style="background:#3498db"><h2>{total_units}</h2><p>Total Units</p></div>
  <div class="stat" style="background:#2ecc71"><h2>{len(df_clean)}</h2><p>BOM Lines</p></div>
  <div class="stat" style="background:#e74c3c"><h2>{len(df_issues)}</h2><p>Issues</p></div>
  <div class="stat" style="background:#9b59b6"><h2>{len(df_family)}</h2><p>Families</p></div>
</div>
<div class="card" style="border-top:5px solid #f1c40f">
  <h2>🤖 AI Procurement Strategy</h2><div class="ai-box">{ai_esc}</div></div>
<div class="card"><h2>📦 Summary by Family/Type</h2><ul>{fam_li}</ul></div>
<div class="card"><h2>⚠️ Data Quality Issues ({len(df_issues)})</h2><ul>{iss_li}</ul></div>
<div class="card"><h2>📋 Full BOM ({len(df_clean)} lines)</h2>
  <div style="max-height:500px;overflow-y:auto">{items_html}</div></div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚡ BIM Procurement Assistant")
st.caption("TFM Tool — BIM Schedule → Procurement Pipeline + AI Strategy · Lucía Varona Vidaurrazaga · IIT 2026")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🤖 AI Provider")

    provider = st.selectbox("Select AI",
        ["Groq (free)", "Ollama (local)", "OpenAI", "Claude (Anthropic)"])
    api_key = model = ""

    if provider == "Groq (free)":
        api_key = st.text_input("Groq API Key", type="password",
                                help="Free at console.groq.com → API Keys")
        model = st.selectbox("Model", [
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
        st.success("✅ Groq is free — no credit card needed!")
    elif provider == "Ollama (local)":
        model = st.text_input("Ollama model", value="llama3")
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    elif provider == "Claude (Anthropic)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"])

    st.divider()
    st.markdown("""
**Compatible formats:**
- Revit Electrical Equipment Schedule
- Panelboard / multi-column BIM schedule
- Any BIM CSV with header row

**How to use:**
1. Upload your Revit / BIM CSV
2. Enter your API key
3. Click **Run Pipeline**
4. Download all outputs
""")

# ── UPLOAD + RUN ──────────────────────────────────────────────────────────────
col_up, col_btn = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "📂 Upload BIM Equipment / Panelboard Schedule (CSV)",
        type=["csv"],
        help="Export directly from Revit or any BIM tool — no manual editing needed."
    )
with col_btn:
    st.write(""); st.write("")
    run = st.button("🚀 Run Pipeline", type="primary",
                    use_container_width=True, disabled=(uploaded is None))

# ── PIPELINE ──────────────────────────────────────────────────────────────────
if run and uploaded:

    with st.spinner("Step 1/3 — Parsing & cleaning BIM data..."):
        df_clean, df_family, df_issues, total_units, label = clean_bim_csv(uploaded)

    st.markdown(
        f"<div class='info-box'>📄 <strong>Detected format:</strong> {label} &nbsp;·&nbsp; "
        f"{len(df_clean)} BOM lines · {total_units} total units · {len(df_issues)} issues</div>",
        unsafe_allow_html=True
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Total Units",  total_units)
    c2.metric("📋 BOM Lines",   len(df_clean))
    c3.metric("⚠️ Issues",      len(df_issues))
    c4.metric("📦 Families",    len(df_family))

    # Preview tabs
    st.subheader("📊 Data Preview")
    t1, t2, t3 = st.tabs(["✅ Clean BOM", "📦 By Family / Type", "⚠️ Quality Issues"])

    with t1:
        st.dataframe(df_clean.drop(columns=["Issues"], errors="ignore"),
                     use_container_width=True, height=320)
    with t2:
        if not df_family.empty:
            st.bar_chart(df_family.set_index("Family")["Count"])
        st.dataframe(df_family, use_container_width=True)
    with t3:
        if not df_issues.empty:
            st.warning(f"{len(df_issues)} rows with missing procurement-critical attributes.")
            show_cols = [c for c in ["Family","Type","Level","Panel Name","Count","Issues"]
                         if c in df_issues.columns]
            st.dataframe(df_issues[show_cols], use_container_width=True)
        else:
            st.success("✅ BOM is complete — no data quality issues!")

    # AI Strategy
    st.divider()
    st.subheader("🤖 AI Procurement Strategy")
    with st.spinner(f"Step 2/3 — Calling {provider}..."):
        prompt = build_prompt(total_units, len(df_clean), len(df_issues), df_family, label)
        ai_out = call_ai(provider, api_key, model, prompt)
    st.markdown(f"<div class='ai-box'>{ai_out}</div>", unsafe_allow_html=True)

    # Exports
    st.divider()
    st.subheader("📥 Download Outputs")
    with st.spinner("Step 3/3 — Building export files..."):
        xls  = build_excel(df_clean, df_family, df_issues)
        docx = build_word(df_clean, df_family, df_issues, total_units, ai_out, label)
        html = build_html(df_clean, df_family, df_issues, total_units, ai_out, label)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("procurement_outputs/tfm_outputs_procurement.xlsx", xls)
            zf.writestr("procurement_outputs/Procurement_Technical_Annex.docx", docx)
            zf.writestr("procurement_outputs/Procurement_Report.html", html.encode("utf-8"))
        zipped = zbuf.getvalue()

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("📊 Excel", xls,  "tfm_outputs_procurement.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    d2.download_button("📝 Word",  docx, "Procurement_Technical_Annex.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True)
    d3.download_button("🌐 HTML",  html.encode("utf-8"), "Procurement_Report.html",
        "text/html", use_container_width=True)
    d4.download_button("📦 ZIP",   zipped, "procurement_outputs.zip",
        "application/zip", use_container_width=True)

    st.success("✅ Pipeline complete! All files ready to download.")

elif not uploaded:
    st.info("👆 Upload a Revit or BIM CSV file to get started.")
