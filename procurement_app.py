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

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-number { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .metric-label  { color: #666; font-size: 0.9rem; margin: 0; }
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
# CSV PARSING — UNIVERSAL, HANDLES ANY REVIT/BIM EXPORT FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def _read_raw(uploaded_file) -> pd.DataFrame:
    """
    Reads any BIM CSV export regardless of:
      - Number of columns (5 cols like Revit EE Schedule, 36 cols like Gurtz Panelboard)
      - BOM markers (UTF-8 BOM)
      - CRLF or LF line endings
      - First row having fewer columns than the rest
    Returns a raw DataFrame with integer column indices.
    """
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()

    # Detect encoding (handle BOM)
    if raw_bytes[:3] == b'\xef\xbb\xbf':
        content = raw_bytes.decode('utf-8-sig')
    else:
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                content = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.splitlines()

    # Pre-detect maximum number of columns so pandas doesn't truncate rows
    max_cols = max((line.count(',') + 1) for line in lines if line.strip()) if lines else 5

    buf = io.StringIO(content)
    df = pd.read_csv(
        buf,
        header=None,
        names=list(range(max_cols)),
        dtype=str,
        on_bad_lines='skip',
    )
    return df


def _is_subtotal_row(row_val: str) -> bool:
    """Detect Revit/BIM subtotal rows like 'PLAZA LEVEL: 144' or 'Grand total: 1618' or ': 1'."""
    if not isinstance(row_val, str):
        return False
    val = row_val.strip()
    # Matches patterns like "FLOOR: 12", ": 1", "Grand total: 1618"
    return bool(re.search(r':\s*\d+\s*$', val, re.IGNORECASE))


def _clean_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMA DETECTION: figure out which kind of CSV this is
# ──────────────────────────────────────────────────────────────────────────────

# Revit Electrical Equipment Schedule columns (your TFM dataset)
REVIT_EE_COLS = ["Family", "Type", "Panel Name", "Level", "Count"]

# Panelboard / general BIM schedule columns (Gurtz format)
PANELBOARD_COLS = ["Level", "Location", "Closet Type", "Panel Name",
                   "Supply From", "Type", "Tub Number", "Cover Number"]


def _detect_header_row(df: pd.DataFrame):
    """
    Scans first 20 rows looking for the main column-header row.
    Returns (header_row_idx, schema_type) where schema_type is
    'revit_ee' or 'panelboard' or 'generic'.
    """
    for i in range(min(20, len(df))):
        row = [_clean_str(v).lower() for v in df.iloc[i]]
        row_joined = " ".join(row)

        # Revit EE Schedule: must contain 'family' and 'count'
        if 'family' in row and 'count' in row:
            return i, 'revit_ee'

        # Panelboard schedule: 'level', 'panel name', 'type' present
        if 'level' in row and 'panel name' in row and 'type' in row:
            return i, 'panelboard'

        # Generic: has several non-empty cells that look like headers
        non_empty = [v for v in row if v and v not in ('nan',)]
        if len(non_empty) >= 4 and i > 0:
            # Heuristic: header rows don't have numbers as first cell
            if not re.match(r'^\d', non_empty[0]):
                return i, 'generic'

    return None, 'unknown'


# ──────────────────────────────────────────────────────────────────────────────
# REVIT ELECTRICAL EQUIPMENT SCHEDULE PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def _parse_revit_ee(df: pd.DataFrame, header_idx: int):
    """
    Handles the original Revit Electrical Equipment Schedule format:
    columns: Family | Type | Panel Name | Level | Count
    Returns (df_clean, df_family, df_issues, total_units, schema_label)
    """
    df_data = df.iloc[header_idx + 1:].copy()
    df_data.columns = list(df_data.columns)  # keep int indices

    # Map to known column positions from header row
    header_row = [_clean_str(v) for v in df.iloc[header_idx]]
    col_map = {}
    for target in REVIT_EE_COLS:
        for j, h in enumerate(header_row):
            if h.lower() == target.lower():
                col_map[target] = j
                break

    # Build working df with named columns
    used_cols = {t: col_map[t] for t in REVIT_EE_COLS if t in col_map}
    df_work = pd.DataFrame()
    for name, idx in used_cols.items():
        df_work[name] = df_data[idx].apply(_clean_str)

    # Fill missing columns with empty string
    for col in REVIT_EE_COLS:
        if col not in df_work.columns:
            df_work[col] = ""

    # Numeric conversion for Count
    df_work["Count"] = pd.to_numeric(df_work["Count"], errors='coerce')

    # Remove rows where Count couldn't be parsed (subtotals, blanks)
    df_work = df_work[df_work["Count"].notna()].copy()
    df_work["Count"] = df_work["Count"].astype(int)
    df_work = df_work[df_work["Count"] > 0].copy()

    # Remove Revit subtotal rows (Type, Panel Name, Level all empty → subtotal)
    subtotal_mask = (
        (df_work["Type"] == "") &
        (df_work["Panel Name"] == "") &
        (df_work["Level"] == "")
    )
    df_work = df_work[~subtotal_mask].copy()

    # Remove empty Family rows
    df_work = df_work[df_work["Family"] != ""].copy()
    df_work = df_work.reset_index(drop=True)

    return _build_outputs(df_work, schema='revit_ee')


# ──────────────────────────────────────────────────────────────────────────────
# PANELBOARD SCHEDULE PIPELINE (Gurtz / multi-column BIM schedules)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_panelboard(df: pd.DataFrame, header_idx: int):
    """
    Handles multi-column panelboard schedules where each row = 1 panel (Count=1).
    Key columns: Level, Panel Name, Type, MCB Rating, Mains Type, Voltage
    Returns (df_clean, df_family, df_issues, total_units, schema_label)
    """
    header_row = [_clean_str(v) for v in df.iloc[header_idx]]

    # Also check row above (sometimes sub-headers span 2 rows)
    if header_idx > 0:
        sub_header = [_clean_str(v) for v in df.iloc[header_idx - 1]]
    else:
        sub_header = [""] * len(header_row)

    # Build column name map: use header_row, fall back to sub_header
    col_names = []
    for j in range(len(header_row)):
        name = header_row[j] if header_row[j] else sub_header[j] if j < len(sub_header) else ""
        col_names.append(name)

    df_data = df.iloc[header_idx + 1:].copy()
    df_data.columns = list(range(len(df_data.columns)))

    # Find column indices for the fields we care about
    def find_col(candidates):
        for c in candidates:
            for j, name in enumerate(col_names):
                if name.strip().lower() == c.lower():
                    return j
        return None

    idx_level      = find_col(["Level"])
    idx_location   = find_col(["Location"])
    idx_closet     = find_col(["Closet Type"])
    idx_panel      = find_col(["Panel Name"])
    idx_supply     = find_col(["Supply From"])
    idx_type       = find_col(["Type"])
    idx_mcb        = find_col(["MCB Rating"])
    idx_mains      = find_col(["Mains Type"])
    idx_voltage    = find_col(["Voltage"])
    idx_tub        = find_col(["Tub Number"])
    idx_qaqc       = find_col(["QA/QC Status", "Wattage"])  # last real col often

    def get_col(row, idx):
        if idx is None or idx >= len(row):
            return ""
        return _clean_str(row.iloc[idx])

    rows = []
    for _, row in df_data.iterrows():
        level  = get_col(row, idx_level)
        panel  = get_col(row, idx_panel)
        ptype  = get_col(row, idx_type)

        # Skip subtotal rows, fully blank rows, and section header rows
        if not level and not panel and not ptype:
            continue                          # completely empty row
        if _is_subtotal_row(level):
            continue                          # e.g. "PLAZA LEVEL: 144" or ": 1"
        if level and not panel and not ptype:
            continue                          # section label row e.g. "PLAZA LEVEL"
        # Keep rows where at least Type OR Panel Name has data
        # (handles panels with empty Level like basement/pre-floor entries)
        if not panel and not ptype:
            continue

        # Skip the grand total row
        if re.search(r'grand total', level, re.IGNORECASE):
            continue

        rows.append({
            "Level":      level,
            "Location":   get_col(row, idx_location),
            "Closet Type":get_col(row, idx_closet),
            "Panel Name": panel,
            "Supply From":get_col(row, idx_supply),
            "Type":       ptype,
            "Tub Number": get_col(row, idx_tub),
            "MCB Rating": get_col(row, idx_mcb),
            "Mains Type": get_col(row, idx_mains),
            "Voltage":    get_col(row, idx_voltage),
            "Count":      1,   # each row = one panel
        })

    df_work = pd.DataFrame(rows)
    if df_work.empty:
        return df_work, pd.DataFrame(), pd.DataFrame(), 0

    # Normalize Level: strip trailing subtotal artefacts like "PLAZA LEVEL: 144"
    df_work["Level"] = df_work["Level"].apply(
        lambda x: re.sub(r':\s*\d+\s*$', '', x).strip()
    )

    return _build_outputs(df_work, schema='panelboard')


# ──────────────────────────────────────────────────────────────────────────────
# GENERIC FALLBACK PARSER
# ──────────────────────────────────────────────────────────────────────────────

def _parse_generic(df: pd.DataFrame, header_idx: int):
    """
    Generic fallback: use whatever columns exist, try to find Level/Name/Type/Count.
    """
    header_row = [_clean_str(v) for v in df.iloc[header_idx]]
    df_data = df.iloc[header_idx + 1:].copy()
    df_data.columns = header_row[:len(df_data.columns)]

    # Drop fully empty rows
    df_data = df_data.dropna(how='all')
    df_data = df_data.reset_index(drop=True)

    # Try to find a Count column
    count_col = None
    for c in df_data.columns:
        if 'count' in str(c).lower() or 'qty' in str(c).lower() or 'quantity' in str(c).lower():
            count_col = c
            break

    if count_col:
        df_data[count_col] = pd.to_numeric(df_data[count_col], errors='coerce')
        df_data = df_data[df_data[count_col].notna() & (df_data[count_col] > 0)]
        df_data[count_col] = df_data[count_col].astype(int)
        df_data = df_data.rename(columns={count_col: "Count"})
    else:
        df_data["Count"] = 1

    # Rename first few columns to standard names where possible
    rename_map = {}
    for col in df_data.columns:
        cl = str(col).lower()
        if 'family' in cl and 'Family' not in rename_map.values():
            rename_map[col] = 'Family'
        elif ('type' in cl or 'model' in cl) and 'Type' not in rename_map.values():
            rename_map[col] = 'Type'
        elif 'panel' in cl and 'Panel Name' not in rename_map.values():
            rename_map[col] = 'Panel Name'
        elif 'level' in cl and 'Level' not in rename_map.values():
            rename_map[col] = 'Level'

    df_data = df_data.rename(columns=rename_map)

    # Ensure base columns exist
    for col in ['Family', 'Type', 'Panel Name', 'Level', 'Count']:
        if col not in df_data.columns:
            df_data[col] = ""

    df_data = df_data[df_data["Family"] != ""].copy()
    df_data = df_data.reset_index(drop=True)

    return _build_outputs(df_data, schema='generic')


# ──────────────────────────────────────────────────────────────────────────────
# SHARED OUTPUT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_outputs(df_work: pd.DataFrame, schema: str):
    """
    Given a cleaned working DataFrame, build:
      - df_clean  : full validated BOM with Issues column
      - df_family : aggregated by equipment family/type
      - df_issues : rows with missing critical attributes
      - total_units: integer sum of all Count values
    """
    if df_work.empty:
        return df_work, pd.DataFrame(columns=["Family","Count"]), pd.DataFrame(), 0

    # Determine the "family" grouping column
    if schema == 'revit_ee':
        family_col = "Family"
        critical_cols = ["Type", "Panel Name", "Level"]
    elif schema == 'panelboard':
        family_col = "Type"       # group by panelboard type (e.g. "225 A (B56)")
        critical_cols = ["Level", "Panel Name"]
    else:
        family_col = "Family" if "Family" in df_work.columns else df_work.columns[0]
        critical_cols = [c for c in ["Type", "Level", "Panel Name"] if c in df_work.columns]

    # Build Issues column
    def get_issues(row):
        errors = []
        if "Type" in row.index and not _clean_str(row.get("Type", "")):
            errors.append("Type undefined")
        if "Panel Name" in row.index and not _clean_str(row.get("Panel Name", "")):
            errors.append("Panel not assigned")
        if "Level" in row.index and not _clean_str(row.get("Level", "")):
            errors.append("Level missing")
        return "; ".join(errors)

    df_work["Issues"] = df_work.apply(get_issues, axis=1)

    # Data quality issues
    df_issues = df_work[df_work["Issues"] != ""].copy()

    # Family summary — DETERMINISTIC aggregation (no AI arithmetic)
    df_family = (
        df_work.groupby(family_col, dropna=False)["Count"]
        .sum()
        .reset_index()
        .rename(columns={family_col: "Family"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    df_family["Count"] = df_family["Count"].astype(int)

    # Total — computed once, passed read-only to AI
    total_units = int(df_work["Count"].sum())

    return df_work, df_family, df_issues, total_units


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def clean_bim_csv(uploaded_file):
    """
    Universal BIM CSV parser. Detects format automatically.
    Returns (df_clean, df_family, df_issues, total_units, schema_info_str)
    """
    df_raw = _read_raw(uploaded_file)

    header_idx, schema = _detect_header_row(df_raw)

    if header_idx is None:
        st.error(
            "❌ No se encontró una cabecera válida en las primeras 20 filas del CSV. "
            "Asegúrate de que el archivo es un schedule exportado desde Revit u otro BIM."
        )
        st.stop()

    if schema == 'revit_ee':
        schema_label = "Revit Electrical Equipment Schedule"
        result = _parse_revit_ee(df_raw, header_idx)
    elif schema == 'panelboard':
        schema_label = "Panelboard / BIM Multi-Column Schedule"
        result = _parse_panelboard(df_raw, header_idx)
    else:
        schema_label = "Generic BIM Schedule (auto-mapped)"
        result = _parse_generic(df_raw, header_idx)

    df_clean, df_family, df_issues, total_units = result
    return df_clean, df_family, df_issues, total_units, schema_label


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(total_units: int, n_rows: int, n_issues: int,
                 df_family: pd.DataFrame, schema_label: str) -> str:
    """
    Builds the v3 prompt (zero hallucinations).
    All quantities come from the deterministic Python pipeline.
    AI receives them READ-ONLY — no arithmetic allowed.
    """
    # Build family inventory string with verified totals
    family_lines = []
    running_check = 0
    for _, row in df_family.iterrows():
        qty = int(row["Count"])
        running_check += qty
        family_lines.append(f"  • {row['Family']}: {qty} units")
    family_data = "\n".join(family_lines)

    return f"""You are an office-based Senior Procurement Engineer on an MEP/Electrical project.
You have received a validated BOM processed from a {schema_label} exported via Autodesk Revit / BIM software.
The data below was validated by a deterministic Python pipeline. Your role is to add strategic procurement value.

═══ VALIDATED DATA — READ-ONLY, DO NOT RECALCULATE ═══
• Total equipment units : {total_units}
• BOM line items         : {n_rows}
• Data quality issues    : {n_issues}
• Cross-check sum        : {running_check}  ← must match Total (verified by pipeline)

Equipment inventory (exact quantities — do not modify, recalculate, or add up):
{family_data}

═══ YOUR TASKS ═══

**1. RFQ LOTS**
Group the equipment families above into logical procurement batches by supplier specialisation.

CRITICAL RULES — violation makes the output unusable:
  (a) MUTUALLY EXCLUSIVE: assign each equipment family to ONE AND ONLY ONE lot. Never list the same item in two lots.
  (b) EXACT QUANTITIES: copy the exact quantity from the inventory above for each item. Do not compute totals or subtotals.
  (c) Do NOT write any additions (e.g. "30+7") or lot-level totals. List items only.

**2. 4-WEEK OFFICE-BASED ACTION PLAN**
Use procurement milestones appropriate for an office-based engineer:
  - Week 1: RFQ preparation and supplier shortlisting
  - Week 2: RFQ issue and supplier follow-up
  - Week 3: Quotation analysis, technical evaluation, and negotiation
  - Week 4: Purchase order placement and delivery scheduling by floor level
  If data quality issues exist ({n_issues} flagged), escalate the Data Quality Log to the BIM Manager for correction in Revit.
  DO NOT suggest physical site visits to fix BIM data — that is the BIM Manager's responsibility.

**3. TOP 3 PROCUREMENT RISKS + MITIGATIONS**
Focus on: lead times, supplier capacity, incomplete BIM data blocking orders, and floor-level delivery sequencing.

═══ ABSOLUTE RULES ═══
• No recalculation of any numbers
• No tables — bullet points only
• No site visits in the action plan
• Professional tone
• If data quality issues > 0, always reference escalation to BIM Manager in Week 1
"""


def call_ai(ai_provider: str, api_key: str, model_name: str, prompt: str) -> str:
    if ai_provider == "Ollama (local)":
        try:
            import ollama
            response = ollama.chat(
                model=model_name or "llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[Ollama error] {e}\n\nAsegúrate de que Ollama está corriendo con el modelo '{model_name}'."

    elif ai_provider == "Groq (free)":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            response = client.chat.completions.create(
                model=model_name or "llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.2,   # lower temperature → more deterministic output
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Groq error] {e}"

    elif ai_provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI error] {e}"

    elif ai_provider == "Claude (Anthropic)":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model_name or "claude-haiku-4-5-20251001",
                max_tokens=1800,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[Anthropic error] {e}"

    return "No AI provider configured."


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_excel(df_clean: pd.DataFrame, df_family: pd.DataFrame,
                df_issues: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_clean.drop(columns=["Issues"], errors="ignore").to_excel(
            writer, sheet_name="Clean_BOM", index=False)
        df_family.to_excel(writer, sheet_name="Summary_by_Family", index=False)
        df_issues.to_excel(writer, sheet_name="Data_Quality_Log", index=False)
    return output.getvalue()


def build_word(df_clean: pd.DataFrame, df_family: pd.DataFrame,
               df_issues: pd.DataFrame, total_units: int,
               ai_strategy: str, schema_label: str) -> bytes:
    from docx import Document
    doc = Document()
    doc.add_heading("Procurement Engineering Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph(f"Source format: {schema_label}")

    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        f"Total equipment units: {total_units}\n"
        f"BOM line items: {len(df_clean)}\n"
        f"Data quality issues: {len(df_issues)}\n"
        f"Equipment families / types: {len(df_family)}"
    )

    doc.add_heading("2. AI-Generated Procurement Strategy", level=1)
    doc.add_paragraph(ai_strategy)

    doc.add_heading("3. Summary by Equipment Family / Type", level=1)
    for _, row in df_family.iterrows():
        doc.add_paragraph(f"{row['Family']}: {int(row['Count'])} units",
                          style="List Bullet")

    doc.add_heading("4. Detailed BOM", level=1)
    for _, row in df_clean.iterrows():
        p = doc.add_paragraph(style="List Bullet")
        # Build line from whatever columns exist
        parts = []
        for col in df_clean.columns:
            if col == "Issues":
                continue
            val = row.get(col, "")
            if val and str(val).strip():
                parts.append(f"{col}: {val}")
        p.add_run(" | ".join(parts))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_html(df_clean: pd.DataFrame, df_family: pd.DataFrame,
               df_issues: pd.DataFrame, total_units: int,
               ai_strategy: str, schema_label: str) -> str:
    family_list = "".join(
        f"<li><strong>{r['Family']}</strong>: {int(r['Count'])} units</li>"
        for _, r in df_family.iterrows()
    )
    items_html = ""
    for _, r in df_clean.iterrows():
        parts = " | ".join(
            f"<strong>{col}</strong>: {r[col]}"
            for col in df_clean.columns
            if col != "Issues" and str(r.get(col, "")).strip()
        )
        items_html += f"<div class='item-row'>{parts}</div>"

    issues_list = "".join(
        f"<li>{r.get('Panel Name', r.get('Family',''))} — {r.get('Issues','')}</li>"
        for _, r in df_issues.iterrows()
    ) if len(df_issues) > 0 else "<li>✅ No issues found.</li>"

    ai_escaped = ai_strategy.replace("<", "&lt;").replace(">", "&gt;")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Procurement Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 40px;
         background: #f8f9fa; line-height: 1.6; }}
  h1   {{ color: #2c3e50; }}
  .card {{ background: white; padding: 25px; border-radius: 12px;
           box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 25px; }}
  .stat-grid {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat {{ flex: 1; min-width: 140px; padding: 20px; color: white;
           border-radius: 8px; text-align: center; }}
  .ai-box {{ border-left: 5px solid #f1c40f; padding-left: 15px;
             background: #fffdf0; white-space: pre-wrap; font-size: 0.9rem; }}
  .item-row {{ padding: 8px 4px; border-bottom: 1px solid #eee; font-size: 0.85rem; }}
  .item-row:hover {{ background: #f1f3f5; }}
  .schema-tag {{ background: #e8f4fd; border: 1px solid #bee3f8;
                 border-radius: 4px; padding: 4px 10px;
                 font-size: 0.8rem; color: #2b6cb0; }}
</style>
</head>
<body>
<h1>⚡ BIM Procurement Dashboard</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;
   <span class="schema-tag">📄 {schema_label}</span></p>
<div class="stat-grid">
  <div class="stat" style="background:#3498db;"><h2>{total_units}</h2><p>Total Units</p></div>
  <div class="stat" style="background:#2ecc71;"><h2>{len(df_clean)}</h2><p>BOM Lines</p></div>
  <div class="stat" style="background:#e74c3c;"><h2>{len(df_issues)}</h2><p>Issues</p></div>
  <div class="stat" style="background:#9b59b6;"><h2>{len(df_family)}</h2><p>Families</p></div>
</div>
<div class="card" style="border-top:5px solid #f1c40f;">
  <h2>🤖 AI Procurement Strategy</h2>
  <div class="ai-box">{ai_escaped}</div>
</div>
<div class="card">
  <h2>📦 Summary by Family / Type</h2>
  <ul>{family_list}</ul>
</div>
<div class="card">
  <h2>⚠️ Data Quality Issues ({len(df_issues)})</h2>
  <ul>{issues_list}</ul>
</div>
<div class="card">
  <h2>📋 Full BOM ({len(df_clean)} lines)</h2>
  <div style="max-height:500px;overflow-y:auto;">{items_html}</div>
</div>
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

    ai_provider = st.selectbox(
        "Select AI",
        ["Groq (free)", "Ollama (local)", "OpenAI", "Claude (Anthropic)"],
        help="Groq is free and runs in the cloud. Ollama runs locally."
    )

    api_key = ""
    model_name = ""

    if ai_provider == "Groq (free)":
        api_key   = st.text_input("Groq API Key", type="password",
                                  help="Free at console.groq.com → API Keys")
        model_name = st.selectbox("Model", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ])
        st.success("✅ Groq is free — no credit card needed!")

    elif ai_provider == "Ollama (local)":
        model_name = st.text_input("Ollama model", value="llama3")

    elif ai_provider == "OpenAI":
        api_key    = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])

    elif ai_provider == "Claude (Anthropic)":
        api_key    = st.text_input("Anthropic API Key", type="password")
        model_name = st.selectbox("Model", [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
        ])

    st.divider()
    st.markdown("""
**Compatible formats:**
- Revit Electrical Equipment Schedule
- Panelboard / multi-column BIM schedule
- Any BIM CSV with Level / Panel / Type columns

**How to use:**
1. Upload your Revit / BIM CSV
2. Enter your API key
3. Click **Run Pipeline**
4. Download all outputs
""")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
col_upload, col_run = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📂 Upload BIM Equipment / Panelboard Schedule (CSV)",
        type=["csv"],
        help="Export directly from Revit or any BIM tool without manual editing."
    )

with col_run:
    st.write("")
    st.write("")
    run_btn = st.button(
        "🚀 Run Pipeline", type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

# ── PIPELINE EXECUTION ────────────────────────────────────────────────────────
if run_btn and uploaded_file:

    with st.spinner("Step 1/3 — Parsing & cleaning BIM data..."):
        df_clean, df_family, df_issues, total_units, schema_label = clean_bim_csv(uploaded_file)

    # Show detected format
    st.markdown(
        f"<div class='info-box'>📄 <strong>Detected format:</strong> {schema_label} &nbsp;·&nbsp; "
        f"{len(df_clean)} BOM lines · {total_units} total units · {len(df_issues)} issues</div>",
        unsafe_allow_html=True
    )

    # ── KPI METRICS ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Total Units",   total_units)
    c2.metric("📋 BOM Lines",    len(df_clean))
    c3.metric("⚠️ Issues",       len(df_issues))
    c4.metric("📦 Families",     len(df_family))

    # ── PREVIEW TABS ─────────────────────────────────────────────────────────
    st.subheader("📊 Data Preview")
    tab1, tab2, tab3 = st.tabs(["✅ Clean BOM", "📦 By Family / Type", "⚠️ Quality Issues"])

    with tab1:
        st.dataframe(
            df_clean.drop(columns=["Issues"], errors="ignore"),
            use_container_width=True, height=320
        )

    with tab2:
        st.bar_chart(df_family.set_index("Family")["Count"])
        st.dataframe(df_family, use_container_width=True)

    with tab3:
        if len(df_issues) > 0:
            st.warning(f"{len(df_issues)} rows with missing procurement-critical attributes.")
            issue_cols = [c for c in ["Family", "Type", "Level", "Panel Name",
                                      "Count", "Issues"] if c in df_issues.columns]
            st.dataframe(df_issues[issue_cols], use_container_width=True)
        else:
            st.success("✅ No data quality issues found — BOM is complete!")

    # ── AI STRATEGY ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Procurement Strategy")

    with st.spinner(f"Step 2/3 — Calling {ai_provider}..."):
        prompt = build_prompt(total_units, len(df_clean), len(df_issues),
                              df_family, schema_label)
        ai_strategy = call_ai(ai_provider, api_key, model_name, prompt)

    st.markdown(f"<div class='ai-box'>{ai_strategy}</div>", unsafe_allow_html=True)

    # ── EXPORT FILES ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Download Outputs")

    with st.spinner("Step 3/3 — Building export files..."):
        excel_bytes = build_excel(df_clean, df_family, df_issues)
        word_bytes  = build_word(df_clean, df_family, df_issues,
                                 total_units, ai_strategy, schema_label)
        html_str    = build_html(df_clean, df_family, df_issues,
                                 total_units, ai_strategy, schema_label)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("procurement_outputs/tfm_outputs_procurement.xlsx", excel_bytes)
            zf.writestr("procurement_outputs/Procurement_Technical_Annex.docx", word_bytes)
            zf.writestr("procurement_outputs/Procurement_Report.html",
                        html_str.encode("utf-8"))
        zip_bytes = zip_buf.getvalue()

    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        st.download_button(
            "📊 Download Excel", data=excel_bytes,
            file_name="tfm_outputs_procurement.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📝 Download Word", data=word_bytes,
            file_name="Procurement_Technical_Annex.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "🌐 Download HTML", data=html_str.encode("utf-8"),
            file_name="Procurement_Report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl4:
        st.download_button(
            "📦 Download All (ZIP)", data=zip_bytes,
            file_name="procurement_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.success("✅ Pipeline complete! All files ready to download.")

elif not uploaded_file:
    st.info("👆 Upload a Revit or BIM CSV file to get started.")
