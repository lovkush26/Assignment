# Message Intelligence System

A local, rule-based pipeline that takes a chronological CSV of messages and produces:

1. **Classification** into six categories, with confidence + reason (Part 1)
2. **Task/event extraction** — title, description, deadline, time, person, priority (Part 2)
3. **Sensitive-information detection & masking** — type, risk, masked text, recommended action (Part 3)

**L2 extends this** (see [L2 extension](#l2-extension) below) with:

4. **Priority & Action Engine** — Critical/High/Medium/Low per actionable message, updated over time
5. **Related-Message Grouping** — links follow-ups/reschedules/completions/etc. to the same task/event
6. **Semantic search & assistant** — natural-language Q&A over everything above
7. **Privacy-aware routing** — process locally / ask for confirmation / block, per message

A Streamlit UI (`app.py`) wraps the full L1+L2 pipeline for interactive/cloud demonstration.

> **The datasets are not included in this repository.** `data/messages.csv` (L1, 900 msgs),
> `data/l2_messages.csv` (L2, 180 msgs), `data/l2_demo_messages.csv`/`l2_demo_queries.csv`
> (the recorded-demo batch), and `data/mandatory_demo_ids.csv` are all gitignored on purpose
> (see *Assumptions and limitations*). `data/sample_messages.csv` +
> `data/sample_l2_messages.csv` are a small **fictional** L1+L2 pair used only so the deployed
> demo has something to process out of the box.

## Quick start

```bash
pip install -r requirements.txt

# CLI: place the real datasets at data/messages.csv (L1) and data/l2_messages.csv (L2), then:
python src/pipeline.py          # writes every outputs/*.json and outputs/*.csv (L1+L2 combined)

# Benchmark (L1 vs L2 timing, TF-IDF index size, fuzzy-match quality delta):
python src/benchmark.py         # writes outputs/benchmark_report.{json,md}

# Interactive UI:
streamlit run app.py            # upload L1/L2 CSVs (or more), or click "Use bundled sample data"
```

No network calls are made anywhere in this codebase — `src/classify.py`, `src/extract.py`,
`src/sensitive.py`, `src/l2_patterns.py`, `src/priority.py`, `src/routing.py` are pure
`re`/`datetime` logic; `src/grouping.py` and `src/search.py` add one local, in-process TF-IDF
vector-space model (scikit-learn) for meaning-based matching — no trained/black-box model, no
external AI service, per the assignment's rules.

## How message classification works

There are no labels in the dataset, so there's nothing to train a supervised
model on. Frequency analysis over all 900 messages (`src/analyze_templates.py`)
showed the data is **highly templated**: ~125 distinct core sentence patterns,
each repeated with different names/dates/numbers, and each randomly wrapped in
one of nine "framing" openers ("For today:", "FYI:", "Quick update:", "Can you
help?", …) that carry **no** category signal — the same opener prefixes tasks,
events, personal notes, and plain FYIs alike.

Given that, `src/classify.py` uses an **ordered rule engine**: regex/keyword
rules grouped by category, evaluated most-specific-first so overlapping
signals resolve predictably:

1. `sensitive_information` — a concrete secret/PII value is detected (reuses
   the Part 3 detector, see below)
2. `promotional` — marketing language (discount/sale/coupon/promo code/etc.),
   boosted by `sender == "Promotions"`
3. `meeting_or_event` — scheduled-event language ("Calendar update:",
   "happens on", "scheduled for", "please join the …", "available for the …")
4. `action_required` — an explicit instruction/deadline directed at the
   recipient ("don't forget to", "I need you to", "please submit/reply/…",
   "is due on", "deadline is")
5. `personal_information` — a non-sensitive personal fact/preference
   ("for my profile,", "personal note:", "remember that", "prefer",
   "favourite", …)
6. `general_information` — fallback: no other signal matched (plain
   FYI/status statement)

Every result carries a **confidence score** (higher for an exact template
match, lower for keyword-only or hedged/tentative language — e.g. "The review
could be Friday afternoon." scores 0.60, not 0.93, because of the "could be"
hedge and missing exact date) and a **reason** string naming the signal that
fired. This ordering matters in practice: e.g. *"You can contact me on 98765
43210-86."* mentions a phone number and could look personal, but the sensitive
phone-number rule (step 1) fires first, so it's correctly filed as
`sensitive_information`, not `personal_information`.

Verified against every one of the 125 unique templates (see
`src/analyze_templates.py` output) — all map to the intended category. Full
900-message run: `action_required=240, general_information=170,
meeting_or_event=170, personal_information=110, promotional=110,
sensitive_information=100`.

### Personal vs. Sensitive — where's the line?

The assignment lists both as separate Part-1 categories, so a boundary had to
be picked and documented:

- **`sensitive_information`**: a concrete, specific secret or identifying
  value is present in the text — password, OTP, recovery code, access token,
  card/bank number, ID number, a full home address, a phone number, or a
  health-test detail.
- **`personal_information`**: a personal fact/preference/opinion with **no**
  such concrete value — e.g. "my emergency contact is my brother" (no
  number), "I live near the central library" (no exact address), "my
  favourite language is Python."

## How tasks and events are extracted

`src/extract.py` only looks at messages already classified as
`action_required` or `meeting_or_event`, and only pulls out fields that are
**literally present** in the text — nothing is inferred.

- Noise-prefix stripping (`src/prefixes.py`) removes the nine random openers
  first, so e.g. `"For today: Complete the onboarding form is due on
  2026-09-10."` reduces to a clean core sentence before regex matching.
- A small number of regex templates (~10 for tasks, 5 generic ones for
  events) cover the recurring sentence structures, e.g.:
  `Please join the {event} on {date}, {time} at {location}.` or
  `I need you to {action} by {date}.`
  These were derived directly from the frequency analysis, not guessed —
  every action/event template in the dataset is covered; a 900-message dry
  run extracts exactly 240 tasks + 170 events, matching the classification
  counts 1:1 (i.e. **no action_required/meeting_or_event message is silently
  dropped**).
- **Person** is only filled in when one of the known sender names is
  explicitly mentioned inside the message body itself (e.g. "Please call
  Maya when you are free." → `person: "Maya"`) — never guessed from the
  `sender` column, since the example in the assignment brief shows `person:
  null` even for a message with an obvious sender.
- **Priority** is derived from the gap between the message's own timestamp
  and the resolved deadline date: `≤2 days → high`, `3–7 days → medium`,
  `>7 days or unresolved → low`. This is a stated, reproducible heuristic,
  not a guess about business importance.
- **Vague/relative dates are never resolved to a guessed absolute date.**
  Phrases like *"tomorrow"*, *"next week"*, *"soon"*, *"Friday afternoon"*
  are stored as the string `"unresolved"` (not `null`, since something *was*
  said — just not resolvable — and not a guessed date either), with the raw
  phrase preserved in `description`. Example: *"Let us meet sometime next
  week."* → `deadline: "unresolved"`, `priority: "low"` (can't judge urgency
  without a date). This is the deliberately-uncertain example used in the
  video demo.

## How sensitive information is detected and masked

`src/sensitive.py` runs independently of classification (and classification
re-uses its output for the `sensitive_information` category, so the two
parts can never disagree). It is a list of `(type, risk, action, regex,
reason)` rules, each tied to one concrete pattern seen in the data:
password, OTP, account-recovery code, access token, card number, bank
account number, personal ID number, phone number, home address, and a
health-test detail.

Each regex has a **named capture group** around exactly the sensitive
substring (not the whole message). Masking replaces only that span with a
fixed `******` marker (fixed-length, so the mask itself never leaks how long
the original value was) and leaves the rest of the sentence intact, e.g.:

```
"Your OTP is 482193-50. It expires in 10 minutes."
→ "Your OTP is ******. It expires in 10 minutes."
```

**Recommended action** is mapped by severity, documented here so the mapping
is auditable rather than arbitrary:

| Type | Risk | Recommended action | Why |
|---|---|---|---|
| password, OTP, recovery code, access token, card number, bank account number, ID number | high | `do_not_store` | credentials/financial secrets/government-style IDs must never be persisted |
| home address, phone number | medium | `ask_for_confirmation` | private contact info that may legitimately be needed, but a human should confirm before it's stored or acted on |
| health information | medium | `ask_for_confirmation` | personal health detail; not explicitly named in the assignment's example list, added as an extended sensitivity type (see below) |

Regardless of the per-message action, the pipeline as a whole **never** sends
raw message text to an external AI/service — that's a system-level rule
enforced by never making a network call anywhere in this codebase, not a
per-message decision.

**Extended category:** the assignment's sensitive-type list is introduced
with "such as," so it isn't meant to be exhaustive. "My recent test result
says vitamin D deficiency-97." isn't a password/PIN/bank detail/token/address
by the letter of the examples, but it's clearly personal health data, so it's
flagged as `health_information` at medium risk with the same masking
treatment, rather than left undetected.

## L2 extension

L2 processes the same 900 L1 messages plus 180 new chronological L2 messages
(`data/l2_messages.csv`, `MSG_0901`-`MSG_1080`) **after** them — `pipeline.py`
concatenates the two CSVs and sorts once by timestamp; L1's dates (2026-09-01
onward) are all earlier than L2's (2026-09-26 onward), so this alone
preserves chronological order and lets `TASK_`/`EVENT_` IDs continue
seamlessly from 240/170 into the L2 range with no extra bookkeeping.

Everything from L1 is **extended, not replaced**: `classify.py`, `extract.py`,
and `sensitive.py` keep every original rule; L2 only *adds* rules (verified —
a fresh L1-only run still yields the exact documented counts:
`action_required=240, meeting_or_event=170, general_information=170,
personal_information=110, promotional=110, sensitive_information=100`). The
180 new L2 messages introduce ~30 new sentence templates (follow-ups,
completions, cancellations, reschedules, deadline changes/conflicts,
ambiguous hedges, new tasks/events, and new sensitive-value phrasings) —
catalogued the same way L1's were, via a frequency-analysis script
(`src/analyze_l2_templates.py`, the L2 analogue of `src/analyze_templates.py`)
run over `l2_messages.csv` and `l2_demo_messages.csv` before writing a single
regex.

New modules (`src/`):

| File | Role |
|---|---|
| `l2_patterns.py` | Recognizes the ~30 new L2 sentence templates and classifies each into a signal type (status-check, completed, cancelled, rescheduled, deadline-urgent/conflict/extended, ambiguous, new-task, new-event) with the literal subject phrase, date/time, urgency/conflict flags, a reason, and a confidence — the single source of truth consumed by `classify.py`, `extract.py`, `priority.py`, and `grouping.py`. |
| `textnorm.py` | `normalize_subject()` — lowercase/de-article/de-punctuate an action or event phrase into one comparable key (handles the dataset's "the the X" double-article typo too). |
| `retrieval.py` | Shared local TF-IDF + cosine-similarity corpus (`Corpus` class) used by both `grouping.py`'s fuzzy fallback and `search.py`'s semantic search. |
| `grouping.py` | Part 2 — related-message grouping (see below). |
| `priority.py` | Part 1 — priority & action engine (see below). |
| `routing.py` | Privacy-aware routing (see below). |
| `search.py` | Part 3 — semantic search & assistant (see below). |
| `benchmark.py` | L1-vs-L2 timing / index-size / fuzzy-match quality-delta report. |

### How priority is calculated and updated (Part 1)

`priority.py` scores every message that creates or touches a tracked task/event with a small
**weighted-signal score**, never a single keyword or a random draw:

```
+4  overdue                        +3  explicitly marked urgent ("Treat this as urgent")
+4  deadline is today              +2  a conflicting deadline was just reported
+3  deadline within 2 days         +1  status is currently "unclear" (ambiguous/hedged)
+2  deadline within a week         +1  a response is still required (not a pure status report)
+1  deadline > a week out /        +1  sent by an authoritative sender (Project Lead/HR Team/Mentor)
    unresolved/unknown             +1  the message also overlaps a detected sensitive value
```
`score >= 7` -> **critical**, `>= 5` -> **high**, `>= 3` -> **medium**, else -> **low**. An item
already `completed`/`cancelled` is always scored `low` regardless of the above (nothing further is
needed). Every record names the exact signals that fired (e.g.
`["overdue", "explicitly_marked_urgent", "conflicting_deadline_reported", "status_unclear"]`) and a
confidence (`0.55 + 0.06 × signal_count`, discounted when the status is ambiguous or the deadline is
unresolved) — this satisfies "not randomly or only using one keyword" the same way L1's day-gap
heuristic was a stated, reproducible rule rather than a guess about business importance.

**Deadline proximity is measured against the touching message's own timestamp**, not a fixed
"today" — so the very same item's priority naturally rises as time passes without any new message
(an unresolved task just gets more overdue), and one priority record is produced **per
(message, item) touch-point**, using `grouping.py`'s per-message history snapshots. That is what
lets priority genuinely change over time: e.g. the dataset's "Confirm the interview slot" task
(`TASK_006`) scores `medium` at its first message, climbs to `high` as its original deadline
approaches, hits `critical` when a deadline-today message lands, eases back to `high` once that
date passes without resolution, then returns to `critical` for good once a later message marks it
explicitly urgent with a conflicting deadline and an ambiguous confirmation — exactly the kind of
"which task became critical" trajectory `outputs/priority.json` and the assistant's
`became-critical` intent are built to answer, not to hardcode.

### How related messages are identified (Part 2)

`grouping.py` links messages to the same task/event/subject in two tiers, literal first:

1. **Exact subject match** (confidence up to 1.0): every extracted task/event seeds a group keyed
   by `normalize_subject(title)`. A later message's `l2_patterns` signal (follow-up, completion,
   cancellation, reschedule, deadline change...) joins that group by the same key. Two items that
   literally share a subject — the dataset deliberately announces "prepare the offline inference
   demo" once as an event and once as a task — are **merged into one group** instead of duplicated.
2. **TF-IDF fallback** (confidence scaled by cosine similarity, threshold 0.35): when the subject
   phrase doesn't exactly match any known title — e.g. "our earlier model-results review" vs. the
   task titled "Review the model results", or "the assignment" vs. "Upload the assignment" — cosine
   similarity over group titles finds the best match. Below the threshold, the message is left
   **unresolved** (`outputs/unresolved_references.json`) rather than guessed into a group; the
   assignment requires "must not invent," and an explicit "no confident match" is more honest than
   a forced link.

A group's `status` follows **latest message wins**: `completed`/`cancelled` signals set it directly;
`rescheduled` updates status + deadline/time (a time-only "the date stays the same, time is now
X" message keeps the prior date); a `deadline_conflict` or `ambiguous` signal — even one arriving
*after* a completion — can move status back to `unclear`, because a later hedge genuinely
re-introduces doubt about an already-reported outcome. Every conflicting deadline is logged with
its message ID and reason in `conflicting_deadlines`, separate from `status`, so "which messages
have conflicting deadlines" (mandatory query DQ04) can be answered directly instead of re-derived.

### How semantic retrieval works (Part 3)

`search.py`'s `Assistant` answers a free-text question in two tiers, the same literal-first,
meaning-based-fallback shape as grouping:

1. **Intent rules** — a small ordered set of regex matchers (`_INTENT_HANDLERS`), one per shape of
   question the assignment lists (today's tasks, critical/high pending, rescheduled, completed/
   cancelled, blocked, needs confirmation, why-critical, deadlines changed, conflicting/uncertain,
   latest status, "became critical", related-to-X). Each handler queries the structured registries
   (priority records, groups, routing decisions) directly — the answer's IDs are always real IDs
   pulled from those registries, never generated text.
2. **TF-IDF fallback** (`retrieval.Corpus`, the same technique `grouping.py` uses) — for open-ended
   "show me / tell me about X" questions, or to resolve *which* group a question refers to, cosine
   similarity over group titles/summaries (and, as a last resort, raw message text) finds the best
   evidence. This is a classic, fully local, explainable vector-space technique — not a trained or
   black-box model, no network call — kept deliberately lightweight and consistent with the rest of
   the pipeline's "no external AI service" rule.

A **stricter threshold** (0.4, vs. grouping's 0.35) is used for the assistant's own fuzzy subject
resolution: a single shared generic word can clear a low bar without the messages being genuinely
related. This was caught concretely during testing — the mandatory query DQ08 ("Was the compliance
form approved by the finance director?", the dataset's deliberate no-evidence test) initially fuzzy-
matched to the unrelated "Complete the onboarding form" task purely because both phrases contain the
word "form" (cosine 0.36). Raising the threshold to 0.4 fixed it. A second, related fix: a
near-verbatim match between the question and a message (cosine ~1.0) means the dataset contains a
message that *asks* the question, not one that *answers* it — DQ08's own text is injected into the
dataset as `DEMO_022`, so without this check the assistant would confidently cite the question back
to itself as if that were evidence. Both fixes make the assistant answer "I don't have sufficient
evidence" for DQ08, correctly, rather than guessing.

Every answer carries the fields the assignment requires: `answer`, `supporting_message_ids`,
`related_item_ids` (and `group_id` where relevant), `relevance_scores`, and a `reason` explaining
which evidence was used and why.

### How privacy-aware routing works

`routing.py` turns Part 3's per-value findings into **one routing decision per message**:
`process_locally` (nothing sensitive detected), `ask_for_confirmation` (a medium-risk value —
phone/address/health), or `blocked` (a high-risk value — password/OTP/token/card/bank/ID). A
message can trigger multiple findings; the route is always driven by the single highest-risk one
present, so the decision is never softer than the most sensitive value actually found. This adds no
new detection logic — it's a routing rule on top of `sensitive.py`'s already-computed output, so it
can never disagree with what's shown as masked elsewhere in the UI.

### What component was optimized, and how benchmarking was performed

The "optimized component" is the **TF-IDF fuzzy-match layer** (`retrieval.py`'s `Corpus`, used by
both `grouping.py` and `search.py`) — the piece that upgrades pure literal rule-matching into
something that also catches differently-worded references. `src/benchmark.py` measures this with an
ablation (grouping run twice over the same L1+L2 data, once with the fuzzy fallback enabled, once
with it disabled) plus end-to-end timing, and writes `outputs/benchmark_report.{json,md}`. On this
machine: the fuzzy layer resolves **12 additional messages** that would otherwise be left
unresolved (27 -> 15), using a 51-document / ~190-term TF-IDF index of a few KB, at a timing cost
indistinguishable from noise (both configurations run in ~0.2s). The full L2 pipeline — six parts
over 1,080 messages — completes in under 2 seconds end to end, so no incremental/caching
architecture was implemented; see *Assumptions and limitations*.

## Assumptions and limitations

- **Rule-based, not statistical.** This is a deliberate choice given the
  dataset (no labels, highly templated) — see the classification section
  above. It will **not** generalize well to free-form, non-templated
  messages outside this dataset's style without adding more rules; it is not
  a general-purpose NLP classifier.
- **Dates/times only recognized in the formats present in the data**
  (`YYYY-MM-DD`, `H:MM`/`HH:MM`). A message using a different date format
  (e.g. "Sept 9th") would not be matched by the current regexes and its
  fields would come back `null`, not a guess.
- **Priority is a stated heuristic** (day-gap thresholds), not a judgment of
  real-world importance — a low-priority-sounding but same-day task will
  still score `high` because of the date-proximity rule, and this is by
  design (documented, not silent).
- **Person extraction** only recognizes the ten known first names present in
  this dataset's `sender` column; it wouldn't recognize an arbitrary new name
  without extending `KNOWN_NAMES` in `src/extract.py`.
- **One incorrect/uncertain case** kept deliberately for the demo:
  `meeting_or_event` matches for *"Let us meet sometime next week."* and
  *"The review could be Friday afternoon."* are scored at only 0.60
  confidence because the language is genuinely ambiguous (could arguably be
  `action_required` instead — a hint to schedule something — rather than a
  meeting itself). This is flagged, not hidden.
- **Streamlit Community Cloud** is a static/compute host, not an AI service —
  uploading a CSV to your own deployed instance during a demo does not
  violate the "no external AI services" rule. No data is stored server-side
  beyond the current session; the app never writes uploaded content to disk.
- **Two different "priority" fields, on purpose.** `tasks_events.csv`'s
  `priority` column is L1's original, unchanged day-gap heuristic (part of
  Part 2's extraction schema). `outputs/priority.json`/`.csv` is L2's new,
  separate multi-signal Critical/High/Medium/Low engine (Part 1) — richer,
  updates over time, and is what the assistant and the video demo use. They
  are deliberately not merged into one field, since they answer different
  questions (a rough same-message estimate vs. a running current
  assessment).
- **"Today"/"now" has no real wall-clock meaning** inside this fictional
  dataset's timeline (2026-09 through 2026-10). Both the priority engine's
  deadline-proximity scoring and the assistant's "what's due today" query
  use **the latest message timestamp in the currently processed batch** as
  "now" — documented here rather than left implicit, since it's a modelling
  choice, not something stated in any message.
- **Fuzzy-match thresholds are a deliberate conservative choice**
  (`grouping.py`: 0.35 for linking an L2 signal's named subject to a
  group; `search.py`: 0.4 for the assistant's own free-text subject
  resolution, discussed above). Favoring "insufficient evidence"/"left
  unresolved" over a confident-looking wrong guess was chosen deliberately
  over maximizing recall, per the assignment's "must not invent" rule — the
  benchmark report's ablation quantifies exactly what this threshold trades
  off (12 extra correct links at 0.35; the DQ08 false-positive avoided at
  0.4).
- **No incremental/caching architecture.** Every run (CLI or each Streamlit
  interaction) reprocesses the full accumulated message history from
  scratch. At this dataset's scale (up to ~1,100 messages) a full L2 run
  takes well under 2 seconds (see `outputs/benchmark_report.md`), so this
  was a deliberate simplicity-over-premature-optimization tradeoff, not an
  oversight — it would need revisiting for a much larger, continuously
  growing message stream.
- **`l2_patterns.py`'s ~30 templates are, like L1's, tied to this dataset's
  exact phrasing** (verified via `src/analyze_l2_templates.py`'s frequency
  dump over both `l2_messages.csv` and `l2_demo_messages.csv` — every
  distinct template is covered). A follow-up/reschedule/cancellation
  message phrased completely differently from what's in this dataset would
  not be recognized without extending those patterns — the same documented
  limitation L1 already carries for its own templates.
- **Sample data is fictional, not a subset of the real dataset.**
  `data/sample_messages.csv` + `data/sample_l2_messages.csv` are a small
  (30-message) hand-written L1+L2 pair used only so the public cloud demo
  has something to process without the real, gitignored data files. They
  exercise every part of the pipeline (classification through the
  assistant) but are not derived from the real 1,080-message dataset in any
  way.

## AI-tool usage declaration

This project (both L1 and its L2 extension) was built with **Claude Code**
(Anthropic) as a coding assistant: for L1, it helped analyze the dataset's
message templates and draft the classification/extraction/masking rule sets.
For L2, it additionally helped run the same frequency-analysis methodology
over the L2 batch (`src/analyze_l2_templates.py`), design and implement
`l2_patterns.py`'s ~30 new template rules, the priority-scoring formula
(`priority.py`), the two-tier exact/TF-IDF grouping logic (`grouping.py`,
`retrieval.py`), the intent-rule + TF-IDF assistant (`search.py`), the
privacy-routing layer (`routing.py`), the benchmark script, the Streamlit UI
extension, and this README. Every rule, regex, threshold, scoring formula,
and category/status/route boundary was reviewed and tested by the author —
against the full L1+L2 dataset, the L2 demo batch, and all 8 mandatory demo
queries (`data/l2_demo_queries.csv`) — and is understood and explainable
(see the sections above for the exact reasoning behind each decision,
including two bugs the author caught and fixed during testing: the DQ08
false-positive fuzzy match, and a priority-history "became critical"
detection bug). No message content was sent to any external AI/LLM service
as part of the pipeline itself at any point — Claude Code was used only as a
local development tool during authoring, never as a runtime component that
processes messages.

## Project structure

```
message-intel-system/
├── app.py                       # Streamlit demo UI (L1 + L2)
├── requirements.txt
├── src/
│   ├── classify.py              # Part 1: category rules (+ L2 signal routing)
│   ├── extract.py               # Part 2: task/event regex extraction (+ L2 new-task/event signals)
│   ├── sensitive.py             # Part 3: sensitive-value detection + masking (+ L2 phrasings)
│   ├── prefixes.py              # shared noise-prefix stripping (+ L2 wrapper labels)
│   ├── l2_patterns.py           # L2: recognizes follow-up/reschedule/completion/etc. templates
│   ├── textnorm.py              # L2: subject-phrase normalization shared by extract/grouping
│   ├── retrieval.py             # L2: shared local TF-IDF + cosine-similarity corpus
│   ├── priority.py              # L2 Part 1: priority & action engine
│   ├── grouping.py              # L2 Part 2: related-message grouping
│   ├── routing.py               # L2: privacy-aware routing (local/confirm/block)
│   ├── search.py                # L2 Part 3: semantic search & assistant
│   ├── pipeline.py              # orchestration (CLI + shared by app.py)
│   ├── benchmark.py             # L1-vs-L2 timing / index-size / quality-delta report
│   ├── analyze_templates.py     # L1 dataset template/frequency analysis
│   └── analyze_l2_templates.py  # L2 dataset template/frequency analysis
├── data/
│   ├── sample_messages.csv      # small fictional L1 demo file (real dataset gitignored)
│   └── sample_l2_messages.csv   # small fictional L2 demo file (real dataset gitignored)
└── outputs/                     # generated JSON/CSV/MD — committed as required submission deliverables
```

## Outputs

`python src/pipeline.py` writes, per message/item/finding/group/decision (L1+L2 combined). These
files are **committed to the repo** (unlike the raw datasets) since the assignment requires the
priority/grouping/routing/benchmark outputs as submission deliverables in their own right:

- `outputs/classifications.json` / `.csv` — Part 1
- `outputs/tasks_events.json` / `.csv` — Part 2
- `outputs/sensitive_findings.json` / `.csv` — Part 3 (values pre-masked)
- `outputs/priority.json` / `.csv` — **L2 Part 1**: one record per (message, item) touch-point
- `outputs/related_message_groups.json` / `.csv` — **L2 Part 2**: every related-message group
- `outputs/privacy_routing.json` / `.csv` — **L2**: one routing decision per message
- `outputs/unresolved_references.json` — **L2**: L2 signals that named a subject with no confident
  group match (surfaced, not silently dropped)
- `outputs/benchmark_report.json` / `.md` — written by `python src/benchmark.py`
