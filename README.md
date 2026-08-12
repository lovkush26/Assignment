# Message Intelligence System

A local, rule-based pipeline that takes a chronological CSV of messages and produces:

1. **Classification** into six categories, with confidence + reason (Part 1)
2. **Task/event extraction** — title, description, deadline, time, person, priority (Part 2)
3. **Sensitive-information detection & masking** — type, risk, masked text, recommended action (Part 3)

A Streamlit UI (`app.py`) wraps the same pipeline for interactive/cloud demonstration.

> **The dataset is not included in this repository.** `data/messages.csv` and
> `data/mandatory_demo_ids.csv` are gitignored on purpose (see *Assumptions and
> limitations*). `data/sample_messages.csv` is a small **fictional** file used
> only so the deployed demo has something to process out of the box.

## Quick start

```bash
pip install -r requirements.txt

# CLI: place the real dataset at data/messages.csv, then:
python src/pipeline.py          # writes outputs/*.json and outputs/*.csv

# Interactive UI:
streamlit run app.py            # upload a CSV, or click "Use bundled sample data"
```

No network calls are made anywhere in this codebase — `src/classify.py`,
`src/extract.py`, and `src/sensitive.py` are pure `re`/`datetime` logic over
data already in memory. Nothing is sent to an external AI service, per the
assignment's rules.

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
- **Dataset excluded from the repo** per the assignment rules. `data/`
  contains only a small fictional `sample_messages.csv` for the public demo;
  the real `messages.csv` / `mandatory_demo_ids.csv` are gitignored and must
  be supplied locally to reproduce the full 900-message run.
- **Streamlit Community Cloud** is a static/compute host, not an AI service —
  uploading a CSV to your own deployed instance during a demo does not
  violate the "no external AI services" rule. No data is stored server-side
  beyond the current session; the app never writes uploaded content to disk.

## AI-tool usage declaration

This project was built with **Claude Code** (Anthropic) as a coding
assistant: it helped analyze the dataset's message templates, draft the
classification/extraction/masking rule sets in `src/`, build the Streamlit
UI, and write this README. All rules, regexes, category boundaries, and the
priority/masking heuristics were reviewed, tested against the full
900-message dataset, and are understood and explainable by the author (see
sections above for the exact reasoning behind each design decision). No
message content was sent to any external AI/LLM service as part of the
pipeline itself — Claude Code was used only as a local development tool
during authoring, not as a runtime component that processes messages.

## Project structure

```
message-intel-system/
├── app.py                    # Streamlit demo UI
├── requirements.txt
├── src/
│   ├── classify.py           # Part 1: category rules
│   ├── extract.py            # Part 2: task/event regex extraction
│   ├── sensitive.py          # Part 3: sensitive-value detection + masking
│   ├── prefixes.py           # shared noise-prefix stripping
│   ├── pipeline.py           # orchestration (CLI + shared by app.py)
│   └── analyze_templates.py  # dataset template/frequency analysis used to design the rules
├── data/
│   └── sample_messages.csv   # small fictional demo file (real dataset gitignored)
└── outputs/                  # generated JSON/CSV (git-ignored by default; see below)
```

## Outputs

`python src/pipeline.py` writes, per message/item/finding:

- `outputs/classifications.json` / `.csv` — Part 1
- `outputs/tasks_events.json` / `.csv` — Part 2
- `outputs/sensitive_findings.json` / `.csv` — Part 3 (values pre-masked)
