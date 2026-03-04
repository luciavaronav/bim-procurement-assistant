"""
BIM Procurement Assistant — Streamlit App
TFM Tool: BIM Schedule → Procurement Pipeline + AI Strategy
Run with: streamlit run procurement_app.py
"""

import streamlit as st
import pandas as pd
import io
import os
import zipfile
from datetime import datetime

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
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
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-number { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .metric-label { color: #666; font-size: 0.9rem; margin: 0; }
    .ai-box {
        background: #fffdf0;
        border-left: 5px solid #f1c40f;
        padding: 20px;
        border-radius: 8px;
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .section-header {
        background: white;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EXPECTED_COLS = ["Family", "Type", "Panel Name", "Level", "Count"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_bim_csv(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Full cleaning pipeline, returns (df_clean, df_family, df_issues, total_units)."""
    df_raw = pd.read_csv(uploaded_file, header=None, names=range(5),
                         dtype=str, sep=',', encoding='utf-8')

    # Detect header row
    header_row_idx = None
    for i in range(min(20, len(df_raw))):
        row_vals = [str(v).strip().lower() for v in df_raw.iloc[i].fillna("")]
        if "family" in row_vals[0]:
            header_row_idx = i
            break

    if header_row_idx is None:
        st.error("❌ No se encontró la cabecera esperada ('Family') en las primeras 20 filas.")
        st.stop()

    df = df_raw.iloc[header_row_idx + 1:].copy()
    df.columns = EXPECTED_COLS

    for col in EXPECTED_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["Count"] = pd.to_numeric(df["Count"], errors='coerce')
    df_clean = df[df["Count"].notna()].copy()
    df_clean["Count"] = df_clean["Count"].astype(int)

    # Remove subtotals / empty families
    mask_subtotal = (df_clean["Type"] == "") & (df_clean["Panel Name"] == "") & (df_clean["Level"] == "")
    df_clean = df_clean[~mask_subtotal].copy()
    df_clean = df_clean[df_clean["Family"] != ""].copy()

    # Issues
    def get_issues(row):
        errors = []
        if not row["Type"]:       errors.append("Type undefined")
        if not row["Panel Name"]: errors.append("Panel not assigned")
        if not row["Level"]:      errors.append("Level missing")
        return "; ".join(errors)

    df_clean["Issues"] = df_clean.apply(get_issues, axis=1)
    df_issues = df_clean[df_clean["Issues"] != ""].copy()
    df_family = df_clean.groupby("Family")["Count"].sum().reset_index().sort_values("Count", ascending=False)
    total_units = int(df_clean["Count"].sum())

    return df_clean, df_family, df_issues, total_units


def call_ai(ai_provider, api_key, model_name, prompt) -> str:
    """Call the chosen AI provider."""
    if ai_provider == "Ollama (local)":
        try:
            import ollama
            response = ollama.chat(
                model=model_name or "llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[Ollama error] {e}\n\nMake sure Ollama is running locally with the model '{model_name}'."

    elif ai_provider == "Groq (free)":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            response = client.chat.completions.create(
                model=model_name or "llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
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
                max_tokens=1500,
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
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[Anthropic error] {e}"

    return "No AI provider configured."


def build_prompt(total_units, n_rows, n_issues, df_family) -> str:
    return f"""Act as a Senior Procurement Engineer.
Summary: {total_units} total units, {n_rows} line items, {n_issues} data quality issues.
Inventory by family:
{df_family.to_string(index=False)}

TASKS:
1. RFQ purchase lots — group equipment into logical procurement batches.
2. 4-week action plan — concrete weekly milestones.
3. Critical risks — top 3 risks with mitigation actions.

RULE: DO NOT USE TABLES. Use bullet points or dashes only.
"""


def build_excel(df_clean, df_family, df_issues) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_clean.drop(columns=["Issues"], errors="ignore").to_excel(writer, sheet_name="Clean_BOM", index=False)
        df_family.to_excel(writer, sheet_name="Summary_by_Family", index=False)
        df_issues.to_excel(writer, sheet_name="Data_Quality_Log", index=False)
    return output.getvalue()


def build_word(df_clean, df_family, df_issues, total_units, ai_strategy) -> bytes:
    from docx import Document
    doc = Document()
    doc.add_heading("Procurement Engineering Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        f"Total equipment units: {total_units}\n"
        f"Order lines (BOM rows): {len(df_clean)}\n"
        f"Data quality issues: {len(df_issues)}"
    )

    doc.add_heading("2. AI-Generated Procurement Strategy", level=1)
    doc.add_paragraph(ai_strategy)

    doc.add_heading("3. Summary by Equipment Family", level=1)
    for _, row in df_family.iterrows():
        doc.add_paragraph(f"{row['Family']}: {row['Count']} units", style="List Bullet")

    doc.add_heading("4. Detailed BOM Line Items", level=1)
    for _, row in df_clean.iterrows():
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{row['Family']}").bold = True
        p.add_run(
            f" | Type: {row['Type']} | Panel: {row['Panel Name']} "
            f"| Level: {row['Level']} | Qty: {row['Count']}"
        )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_html(df_clean, df_family, df_issues, total_units, ai_strategy) -> str:
    family_list = "".join(
        f"<li><strong>{r['Family']}</strong>: {r['Count']} units</li>"
        for _, r in df_family.iterrows()
    )
    items_list = "".join(
        f"<div class='item-row'><strong>{r['Family']}</strong> — {r['Type']}<br>"
        f"<small>Panel: {r['Panel Name']} | Level: {r['Level']} | Qty: {r['Count']}</small></div>"
        for _, r in df_clean.iterrows()
    )
    issues_list = "".join(
        f"<li>{r['Family']} ({r['Type']}): {r['Issues']}</li>"
        for _, r in df_issues.iterrows()
    ) if len(df_issues) > 0 else "<li>No issues found.</li>"

    return f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<title>Procurement Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; line-height: 1.6; }}
  h1 {{ color: #2c3e50; }}
  .card {{ background: white; padding: 25px; border-radius: 12px;
           box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 25px; }}
  .stat-grid {{ display: flex; gap: 15px; margin-bottom: 20px; }}
  .stat {{ flex: 1; padding: 20px; color: white; border-radius: 8px; text-align: center; }}
  .ai-box {{ border-left: 5px solid #f1c40f; padding-left: 15px;
             background: #fffdf0; white-space: pre-wrap; }}
  .item-row {{ padding: 10px; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  .item-row:hover {{ background: #f1f3f5; }}
</style>
</head>
<body>
<h1>⚡ Procurement Dashboard</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="stat-grid">
  <div class="stat" style="background:#3498db;"><h2>{total_units}</h2><p>Total Units</p></div>
  <div class="stat" style="background:#2ecc71;"><h2>{len(df_clean)}</h2><p>BOM Lines</p></div>
  <div class="stat" style="background:#e74c3c;"><h2>{len(df_issues)}</h2><p>Issues</p></div>
</div>
<div class="card" style="border-top:5px solid #f1c40f;">
  <h2>🤖 AI Procurement Strategy</h2>
  <div class="ai-box">{ai_strategy}</div>
</div>
<div class="card">
  <h2>📦 Summary by Family</h2>
  <ul>{family_list}</ul>
</div>
<div class="card">
  <h2>⚠️ Data Quality Issues</h2>
  <ul>{issues_list}</ul>
</div>
<div class="card">
  <h2>📋 Detailed BOM ({len(df_clean)} lines)</h2>
  <div style="max-height:400px;overflow-y:auto;">{items_list}</div>
</div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚡ BIM Procurement Assistant")
st.caption("TFM Tool — BIM Schedule → Procurement Pipeline + AI Strategy")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🤖 AI Provider")
    ai_provider = st.selectbox(
        "Select AI",
        ["Groq (free)", "Ollama (local)", "OpenAI", "Claude (Anthropic)"],
        help="Groq is free and works in the cloud. Ollama runs locally."
    )

    api_key = ""
    model_name = ""

    if ai_provider == "Groq (free)":
        api_key = st.text_input("Groq API Key", type="password",
                                help="Free at console.groq.com → API Keys")
        model_name = st.selectbox("Model", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ])
        st.success("✅ Groq is free — no credit card needed!")

    elif ai_provider == "Ollama (local)":
        model_name = st.text_input("Ollama model", value="llama3",
                                   help="Model must be pulled locally (ollama pull llama3)")

    elif ai_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])

    elif ai_provider == "Claude (Anthropic)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model_name = st.selectbox("Model", [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
        ])

    st.divider()
    st.info(
        "**How to use:**\n"
        "1. Upload your Revit CSV\n"
        "2. Click 'Run Pipeline'\n"
        "3. Review results\n"
        "4. Download outputs"
    )

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
col_upload, col_run = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📂 Upload Revit Electrical Equipment Schedule (CSV)",
        type=["csv"],
        help="Export directly from Revit without any manual editing."
    )

with col_run:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True,
                        disabled=(uploaded_file is None))

# ── PIPELINE ──────────────────────────────────────────────────────────────────
if run_btn and uploaded_file:

    with st.spinner("Step 1/3 — Cleaning BIM data..."):
        df_clean, df_family, df_issues, total_units = clean_bim_csv(uploaded_file)

    # ── KPI METRICS ──────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Total Units", total_units)
    c2.metric("📋 BOM Lines", len(df_clean))
    c3.metric("⚠️ Issues", len(df_issues))
    c4.metric("📦 Families", len(df_family))

    # ── PREVIEW TABS ─────────────────────────────────────────────────────────
    st.subheader("📊 Data Preview")
    tab1, tab2, tab3 = st.tabs(["✅ Clean BOM", "📦 By Family", "⚠️ Quality Issues"])

    with tab1:
        st.dataframe(df_clean.drop(columns=["Issues"], errors="ignore"),
                     use_container_width=True, height=300)

    with tab2:
        st.bar_chart(df_family.set_index("Family")["Count"])
        st.dataframe(df_family, use_container_width=True)

    with tab3:
        if len(df_issues) > 0:
            st.warning(f"{len(df_issues)} rows with missing procurement attributes.")
            st.dataframe(df_issues[["Family", "Type", "Panel Name", "Level", "Count", "Issues"]],
                         use_container_width=True)
        else:
            st.success("No data quality issues found!")

    # ── AI STRATEGY ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Procurement Strategy")

    with st.spinner(f"Step 2/3 — Calling {ai_provider}..."):
        prompt = build_prompt(total_units, len(df_clean), len(df_issues), df_family)
        ai_strategy = call_ai(ai_provider, api_key, model_name, prompt)

    st.markdown(
        f"<div class='ai-box'>{ai_strategy}</div>",
        unsafe_allow_html=True
    )

    # ── EXPORTS ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Download Outputs")

    with st.spinner("Step 3/3 — Building export files..."):
        excel_bytes = build_excel(df_clean, df_family, df_issues)
        word_bytes  = build_word(df_clean, df_family, df_issues, total_units, ai_strategy)
        html_str    = build_html(df_clean, df_family, df_issues, total_units, ai_strategy)

        # Bundle into ZIP
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("procurement_outputs/tfm_outputs_procurement.xlsx", excel_bytes)
            zf.writestr("procurement_outputs/Procurement_Technical_Annex.docx", word_bytes)
            zf.writestr("procurement_outputs/Procurement_Report.html", html_str.encode("utf-8"))
        zip_bytes = zip_buf.getvalue()

    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        st.download_button(
            "📊 Download Excel",
            data=excel_bytes,
            file_name="tfm_outputs_procurement.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📝 Download Word",
            data=word_bytes,
            file_name="Procurement_Technical_Annex.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "🌐 Download HTML",
            data=html_str.encode("utf-8"),
            file_name="Procurement_Report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl4:
        st.download_button(
            "📦 Download All (ZIP)",
            data=zip_bytes,
            file_name="procurement_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.success("✅ Pipeline complete! All files ready to download.")

elif not uploaded_file:
    st.info("👆 Upload a Revit CSV file to get started.")
