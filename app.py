"""
Streamlit demo UI for the message intelligence pipeline (classification,
task/event extraction, sensitive-info detection).

Privacy-by-design: this app ships with NO dataset baked in. It only ever
processes a CSV you upload in your own browser session (or the small
fictional data/sample_messages.csv bundled for a first look). Nothing is
sent to an external AI service, and uploaded data is never written to disk
or committed anywhere by this app — it is processed in memory for the
current session only.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import process_dataframe  # noqa: E402

st.set_page_config(page_title="Message Intelligence System", layout="wide")

REQUIRED_COLUMNS = {"message_id", "timestamp", "sender", "message"}

CATEGORY_LABELS = {
    "action_required": "Action Required",
    "meeting_or_event": "Meeting or Event",
    "personal_information": "Personal Information",
    "general_information": "General Information",
    "promotional": "Promotional",
    "sensitive_information": "Sensitive Information",
}

st.title("Message Intelligence System")
st.caption(
    "Rule-based classification, task/event extraction, and sensitive-information "
    "detection & masking — runs entirely locally, no external AI calls."
)

with st.expander("How this works (click to expand)", expanded=False):
    st.markdown(
        """
**Part 1 — Classification.** Every message is scored against an ordered set of
regex/keyword rules (most specific first: sensitive → promotional → meeting/event
→ action required → personal → general fallback). Each result carries a
confidence score and a plain-language reason tied to the exact rule that fired.

**Part 2 — Task & event extraction.** For messages classified as
`action_required` or `meeting_or_event`, regex templates pull out title,
description, deadline, time, person, and priority. Vague/relative phrases
("tomorrow", "next week") are stored as `"unresolved"` rather than guessed.

**Part 3 — Sensitive-information detection.** A separate set of regex rules
matches concrete secret/PII patterns (passwords, OTPs, tokens, card & bank
numbers, IDs, addresses, phone numbers, health details). Every match is
masked before being shown or stored anywhere.

No trained/black-box ML model is used — the dataset ships with no labels, so
there is nothing to train on, and rule-based logic keeps every decision
traceable and explainable, which the assignment requires.
        """
    )

st.divider()

st.subheader("1. Load messages")
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload your messages CSV", type=["csv"])
with col2:
    use_sample = st.button("Use bundled sample data instead", use_container_width=True)

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {sorted(missing)}")
    else:
        st.session_state.df = df
        st.session_state.source = uploaded.name
elif use_sample:
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "sample_messages.csv")
    st.session_state.df = df
    st.session_state.source = "sample_messages.csv (bundled fictional demo data)"

if st.session_state.df is None:
    st.info(
        "Upload a CSV with columns message_id, timestamp, sender, message — "
        "or click 'Use bundled sample data' for a quick look. Your file is "
        "processed only in this browser session; it is never saved to the "
        "repository or sent to an external AI service."
    )
    st.stop()

df = st.session_state.df
st.success(f"Loaded {len(df)} messages from {st.session_state.source}.")

with st.spinner("Processing..."):
    classifications, items, sensitive_records = process_dataframe(df)

clf_df = pd.DataFrame(classifications)
items_df = pd.DataFrame(items)
sens_df = pd.DataFrame(sensitive_records)

st.divider()
st.subheader("2. Classification results (Part 1)")

counts = clf_df["category"].value_counts()
cols = st.columns(len(CATEGORY_LABELS))
for col, (cat, label) in zip(cols, CATEGORY_LABELS.items()):
    col.metric(label, int(counts.get(cat, 0)))

selected_cats = st.multiselect(
    "Filter by category",
    options=list(CATEGORY_LABELS.keys()),
    format_func=lambda c: CATEGORY_LABELS[c],
    default=list(CATEGORY_LABELS.keys()),
)
filtered_clf = clf_df[clf_df["category"].isin(selected_cats)]
st.dataframe(filtered_clf, use_container_width=True, hide_index=True)
st.download_button(
    "Download classifications (JSON)",
    clf_df.to_json(orient="records", indent=2, force_ascii=False),
    file_name="classifications.json",
    mime="application/json",
)

st.divider()
st.subheader("3. Extracted tasks & events (Part 2)")
if items_df.empty:
    st.write("No task/event items extracted from this data.")
else:
    type_filter = st.radio("Show", ["All", "Tasks only", "Events only"], horizontal=True)
    view = items_df
    if type_filter == "Tasks only":
        view = items_df[items_df["type"] == "task"]
    elif type_filter == "Events only":
        view = items_df[items_df["type"] == "event"]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download tasks & events (JSON)",
        items_df.to_json(orient="records", indent=2, force_ascii=False),
        file_name="tasks_events.json",
        mime="application/json",
    )

st.divider()
st.subheader("4. Sensitive information detected & masked (Part 3)")
if sens_df.empty:
    st.write("No sensitive information detected in this data.")
else:
    st.dataframe(
        sens_df[["message_id", "sensitivity_type", "risk", "masked_text",
                 "recommended_action", "reason"]],
        use_container_width=True, hide_index=True,
    )
    st.caption("Values are masked before display — the underlying secret is never rendered.")
    st.download_button(
        "Download sensitive findings (JSON)",
        sens_df.to_json(orient="records", indent=2, force_ascii=False),
        file_name="sensitive_findings.json",
        mime="application/json",
    )

st.divider()
st.subheader("5. Check specific / mandatory message IDs")
id_text = st.text_area(
    "Paste message IDs (one per line) to inspect across all three parts",
    placeholder="MSG_0001\nMSG_0002\n...",
)
mand_upload = st.file_uploader("...or upload a mandatory_demo_ids.csv", type=["csv"], key="mand")

ids_to_check = []
if mand_upload is not None:
    ids_to_check = pd.read_csv(mand_upload)["message_id"].astype(str).tolist()
elif id_text.strip():
    ids_to_check = [x.strip() for x in id_text.splitlines() if x.strip()]

if ids_to_check:
    st.write(f"Checking {len(ids_to_check)} message ID(s):")
    for mid in ids_to_check:
        clf_row = clf_df[clf_df["message_id"] == mid]
        if clf_row.empty:
            st.warning(f"{mid}: not found in the loaded dataset.")
            continue
        row = clf_row.iloc[0]
        with st.container(border=True):
            st.markdown(f"**{mid}** — `{CATEGORY_LABELS[row['category']]}` "
                        f"(confidence {row['confidence']:.2f})")
            st.caption(row["reason"])
            related_items = items_df[items_df["source_message_id"] == mid] if not items_df.empty else pd.DataFrame()
            if not related_items.empty:
                st.dataframe(related_items, use_container_width=True, hide_index=True)
            related_sens = sens_df[sens_df["message_id"] == mid] if not sens_df.empty else pd.DataFrame()
            if not related_sens.empty:
                st.dataframe(
                    related_sens[["sensitivity_type", "risk", "masked_text", "recommended_action"]],
                    use_container_width=True, hide_index=True,
                )
