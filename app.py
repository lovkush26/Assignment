"""
Streamlit demo UI for the message intelligence pipeline.

L1 (Parts 1-3): classification, task/event extraction, sensitive-info
detection & masking.
L2 (Parts 1-3 extension): priority & action engine, related-message
grouping, privacy-aware routing, and a semantic-search assistant.

Privacy-by-design: this app ships with NO real dataset baked in — only a
small fictional sample pair (data/sample_messages.csv /
data/sample_l2_messages.csv) for a first look on the public cloud demo.
Everything else is processed only from a CSV you upload in your own browser
session; nothing is sent to an external AI service, and uploaded data is
never written to disk by this app.

Multiple batches (L1, L2, and any further "new unseen batch" you upload
live, e.g. l2_demo_messages.csv) accumulate in session state and are
reprocessed together on every run — the whole pipeline finishes in well
under a second at this dataset's scale (see outputs/benchmark_report.md),
so no incremental-state machinery was needed.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import process_full  # noqa: E402
from search import Assistant  # noqa: E402

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

ROUTE_LABELS = {
    "process_locally": "Processed locally",
    "ask_for_confirmation": "Needs confirmation",
    "blocked": "Blocked",
}

st.title("Message Intelligence System")
st.caption(
    "L1: rule-based classification, task/event extraction, sensitive-information detection & masking. "
    "L2: priority & action engine, related-message grouping, privacy-aware routing, semantic-search assistant. "
    "Runs entirely locally — no external AI calls."
)

with st.expander("How this works (click to expand)", expanded=False):
    st.markdown(
        """
**L1 — Part 1 Classification.** Every message is scored against an ordered set of regex/keyword
rules. Each result carries a confidence score and a plain-language reason.

**L1 — Part 2 Task & event extraction.** Regex templates pull out title, description, deadline,
time, person from `action_required`/`meeting_or_event` messages. Vague phrases are stored as
`"unresolved"` rather than guessed.

**L1 — Part 3 Sensitive-information detection.** Regex rules match concrete secret/PII patterns;
every match is masked before being shown or stored anywhere.

**L2 — Part 1 Priority & Action Engine.** Every message that touches a task/event gets a
Critical/High/Medium/Low priority, recomputed at each touch-point from deadline proximity (as of
that message's own timestamp), explicit urgency language, conflicting deadlines, ambiguous status,
whether a response is still required, sender, and sensitivity — never a single keyword.

**L2 — Part 2 Related-message grouping.** Messages about the same task/event/subject are linked
first by exact normalized-subject match, then by TF-IDF cosine similarity as a meaning-based
fallback (e.g. "our earlier model-results review" -> "Review the model results"). A group's status
follows "latest message wins": completion/cancellation/reschedule/ambiguous signals move it
forward or back to "unclear" over time.

**L2 — Part 3 Semantic search & assistant.** An ordered intent-rule layer plus a TF-IDF fallback
answers natural-language questions using only the structured output above — every answer carries
supporting message IDs, related task/event/group IDs, relevance scores, and a reason. If there
isn't enough evidence, it says so rather than guessing.

**Privacy-aware routing** turns Part 3's per-value findings into one decision per message:
`process_locally` (nothing sensitive), `ask_for_confirmation` (medium risk - phone/address/health),
or `blocked` (high risk - password/OTP/token/card/bank/ID) — never softer than the riskiest value
found in that message.

No trained/black-box ML model is used anywhere — classification/extraction/sensitivity/priority are
transparent rule engines; grouping/search use a local TF-IDF vector-space model (scikit-learn,
in-process, no network call) as the one "meaning-based" component.
        """
    )

st.divider()

# ------------------------------------------------------------------ #
# 1. Load messages — batches accumulate in session state
st.subheader("1. Load messages")
st.caption(
    "Upload the real L1 dataset and the L2 batch (both are excluded from the public repo per the "
    "assignment's rules), then optionally upload a further 'new unseen batch' live — e.g. "
    "l2_demo_messages.csv — to see it get classified, prioritized, and grouped against everything "
    "already loaded. Or click the sample-data button for a quick, fully self-contained fictional demo."
)

if "batches" not in st.session_state:
    st.session_state.batches = []  # list of (label, df)

col1, col2, col3 = st.columns(3)
with col1:
    up = st.file_uploader("Upload a messages CSV", type=["csv"], key="uploader")
    batch_label = st.text_input("Label for this batch", value="new batch")
    if up is not None and st.button("Add this batch"):
        df_new = pd.read_csv(up)
        missing = REQUIRED_COLUMNS - set(df_new.columns)
        if missing:
            st.error(f"CSV is missing required columns: {sorted(missing)}")
        else:
            st.session_state.batches.append((batch_label or up.name, df_new))
            st.success(f"Added batch '{batch_label or up.name}' ({len(df_new)} messages).")
with col2:
    if st.button("Use bundled sample data (fictional)", use_container_width=True):
        root = Path(__file__).resolve().parent
        st.session_state.batches = [
            ("sample L1", pd.read_csv(root / "data" / "sample_messages.csv")),
            ("sample L2", pd.read_csv(root / "data" / "sample_l2_messages.csv")),
        ]
        st.success("Loaded the small fictional sample L1 + L2 batches.")
with col3:
    if st.button("Clear all loaded batches", use_container_width=True):
        st.session_state.batches = []

if st.session_state.batches:
    st.write("Loaded batches (processed together, in chronological order):")
    for label, bdf in st.session_state.batches:
        st.write(f"- **{label}**: {len(bdf)} messages")
else:
    st.info("No batches loaded yet. Upload a CSV or use the bundled sample data above.")
    st.stop()

df = pd.concat([bdf for _, bdf in st.session_state.batches], ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)

with st.spinner("Processing full pipeline (Parts 1-3 + priority/grouping/routing)..."):
    result = process_full(df)

clf_df = pd.DataFrame(result["classifications"])
items_df = pd.DataFrame(result["items"])
sens_df = pd.DataFrame(result["sensitive_records"])
priority_df = pd.DataFrame(result["priority_records"])
groups_df = pd.DataFrame(result["groups"])
routing_df = pd.DataFrame(result["routing_records"])

st.success(f"Processed {len(df)} messages across {len(st.session_state.batches)} batch(es).")

# ------------------------------------------------------------------ #
st.divider()
st.subheader("2. Classification (L1 Part 1)")
counts = clf_df["category"].value_counts()
cols = st.columns(len(CATEGORY_LABELS))
for col, (cat, label) in zip(cols, CATEGORY_LABELS.items()):
    col.metric(label, int(counts.get(cat, 0)))
selected_cats = st.multiselect(
    "Filter by category", options=list(CATEGORY_LABELS.keys()),
    format_func=lambda c: CATEGORY_LABELS[c], default=list(CATEGORY_LABELS.keys()),
)
st.dataframe(clf_df[clf_df["category"].isin(selected_cats)], use_container_width=True, hide_index=True)

st.divider()
st.subheader("3. Extracted tasks & events (L1 Part 2)")
if items_df.empty:
    st.write("No task/event items extracted.")
else:
    type_filter = st.radio("Show", ["All", "Tasks only", "Events only"], horizontal=True, key="type_filter")
    view = items_df
    if type_filter == "Tasks only":
        view = items_df[items_df["type"] == "task"]
    elif type_filter == "Events only":
        view = items_df[items_df["type"] == "event"]
    st.dataframe(view, use_container_width=True, hide_index=True)

st.divider()
st.subheader("4. Sensitive information detected & masked (L1 Part 3)")
if sens_df.empty:
    st.write("No sensitive information detected.")
else:
    st.dataframe(
        sens_df[["message_id", "sensitivity_type", "risk", "masked_text", "recommended_action", "reason"]],
        use_container_width=True, hide_index=True,
    )
    st.caption("Values are masked before display — the underlying secret is never rendered.")

# ------------------------------------------------------------------ #
st.divider()
st.subheader("5. Priority & Action Engine (L2 Part 1)")
st.caption(
    "One record per (message, item) touch-point — priority is recomputed every time a message "
    "creates or updates a tracked task/event, so the same item's priority can rise or fall over time."
)
if priority_df.empty:
    st.write("No priority-relevant messages.")
else:
    prio_filter = st.multiselect("Filter by priority", ["critical", "high", "medium", "low"],
                                  default=["critical", "high", "medium", "low"])
    view = priority_df[priority_df["priority"].isin(prio_filter)]
    st.dataframe(view[["message_id", "item_id", "group_id", "priority", "confidence", "reason", "signals"]],
                 use_container_width=True, hide_index=True)

st.divider()
st.subheader("6. Related-Message Groups (L2 Part 2)")
st.caption(
    "Exact normalized-subject match (high confidence) plus a TF-IDF meaning-based fallback (lower "
    "confidence) for differently-worded references to the same task/event."
)
if groups_df.empty:
    st.write("No related-message groups.")
else:
    multi_only = st.checkbox("Show only groups with more than one message", value=True)
    view = groups_df[groups_df["related_message_ids"].apply(len) > 1] if multi_only else groups_df
    for _, g in view.iterrows():
        with st.container(border=True):
            st.markdown(f"**{g['group_id']} - {g['title']}** — status: `{g['status']}`"
                        + (f", latest deadline: {g['latest_deadline']}" if g["latest_deadline"] else "")
                        + f" (confidence {g['confidence']:.2f})")
            st.caption(g["summary"])
            st.write("Messages:", ", ".join(g["related_message_ids"]))
            if g["conflicting_deadlines"]:
                st.warning(f"Conflicting deadlines reported: {g['conflicting_deadlines']}")

# ------------------------------------------------------------------ #
st.divider()
st.subheader("7. Privacy-aware routing")
st.caption(
    "Every message routes to exactly one of three states, driven by the highest-risk sensitive "
    "value found in it (Part 3 detection is untouched — this only adds a routing decision on top)."
)
route_counts = routing_df["route"].value_counts()
rcols = st.columns(3)
for col, (route, label) in zip(rcols, ROUTE_LABELS.items()):
    col.metric(label, int(route_counts.get(route, 0)))

example_cols = st.columns(3)
for col, (route, label) in zip(example_cols, ROUTE_LABELS.items()):
    subset = routing_df[routing_df["route"] == route]
    with col:
        st.markdown(f"**Example: {label}**")
        if subset.empty:
            st.write("(none in this batch)")
        else:
            ex = subset.iloc[0]
            st.write(f"`{ex['message_id']}`")
            st.caption(ex["reason"])

with st.expander("Full privacy-routing table"):
    st.dataframe(routing_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
st.divider()
st.subheader("8. Intelligent Assistant (L2 Part 3 — semantic search & Q&A)")
st.caption(
    "Answers are built only from the structured output above (plus masked message text where "
    "routing permits) — every answer includes supporting message IDs, related item/group IDs, "
    "relevance scores, and a reason. If there isn't enough evidence, it says so."
)
assistant = Assistant(
    result["messages_df"], result["classifications"], result["items"], result["sensitive_records"],
    result["priority_records"], result["groups"], result["routing_records"],
)

example_queries = [
    "What tasks should I complete today?",
    "Which critical or high-priority tasks are still pending?",
    "What meetings were rescheduled?",
    "Which tasks have been completed?",
    "Which messages require confirmation?",
    "What deadlines have changed?",
]
query = st.selectbox("Try an example question, or type your own below", [""] + example_queries)
typed = st.text_input("Ask a question", value=query)

if typed.strip():
    ans = assistant.answer(typed)
    st.markdown(f"**Answer:** {ans['answer']}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Supporting message IDs:", ans.get("supporting_message_ids") or "—")
        st.write("Related item IDs:", ans.get("related_item_ids") or "—")
        if ans.get("group_id"):
            st.write("Group ID:", ans["group_id"])
    with c2:
        st.write("Relevance scores:", ans.get("relevance_scores") or "—")
    st.caption(f"Reason: {ans['reason']}")

st.divider()
st.subheader("9. Check specific / mandatory message IDs")
id_text = st.text_area(
    "Paste message IDs (one per line) to inspect across every part",
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
            st.warning(f"{mid}: not found in the loaded batches.")
            continue
        row = clf_row.iloc[0]
        with st.container(border=True):
            st.markdown(f"**{mid}** — `{CATEGORY_LABELS[row['category']]}` (confidence {row['confidence']:.2f})")
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
            prio_row = priority_df[priority_df["message_id"] == mid] if not priority_df.empty else pd.DataFrame()
            if not prio_row.empty:
                st.dataframe(prio_row[["item_id", "priority", "confidence", "reason"]],
                             use_container_width=True, hide_index=True)
            route_row = routing_df[routing_df["message_id"] == mid] if not routing_df.empty else pd.DataFrame()
            if not route_row.empty:
                r = route_row.iloc[0]
                st.write(f"Privacy route: **{ROUTE_LABELS[r['route']]}** — {r['reason']}")
            group_row = groups_df[groups_df["related_message_ids"].apply(lambda ids: mid in ids)] \
                if not groups_df.empty else pd.DataFrame()
            if not group_row.empty:
                g = group_row.iloc[0]
                st.write(f"Related-message group: **{g['group_id']} - {g['title']}** (status: {g['status']})")
