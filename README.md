# Taxor Bill Extraction

Evaluating vision-capable LLMs for structured data extraction from handwritten Indian bills and pushing results to Zoho Books.

## Overview

Brief description of the project goal: benchmark Google Gemini, Anthropic Claude, and OpenRouter Vision models on handwritten Indian bill extraction accuracy and cost efficiency, with optional Zoho Books integration.

## Setup

> [!NOTE]
> OpenAI's API now requires a credit card with no reliable free trial as of August 2026, so `google/gemma-4-26b-a4b-it:free` via OpenRouter was substituted as the third vision-capable model, keeping the assignment's zero-real-spend constraint intact.
>
> This project is designed to run entirely on free-tier (Gemini, OpenRouter) and one-time trial credits (Claude) with no payment method attached to any account. Total actual cost incurred: $0.

```bash
# Clone repository & navigate to project directory
cd taxor-bill-extraction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Perform a dry run to verify estimated tokens & cost guardrails without calling APIs
python src/run_pipeline.py --dry-run
```

## Dataset

The dataset consists of 15 handwritten bills spanning common Indian small-business formats — kirana store, tea stall, auto rickshaw fare, tailor, hardware/paint shop, dhaba, pharmacy, stationery/xerox shop, electrician service, vegetable vendor, mobile repair, bakery, laundry, cycle repair, and carpenter/furniture work.

Bills were handwritten by me using fictional vendor names and invented line items, in realistic Indian small-business bill formats, since sourcing genuine bills at short notice with proper consent and redaction wasn't feasible in the assignment timeframe. Handwriting style, photo angle, and lighting were deliberately varied across the set to simulate real-world OCR difficulty:
- Handwriting: normal pace (5 bills), deliberately faster/messier (4 bills), written with non-dominant hand for genuinely hard-to-read cases (2 bills), small/cramped lettering (2 bills), pencil with slight smudging (2 bills)
- Photo conditions: straight-on with good lighting, slight angle, and dim/warm lighting or partial shadow, spread evenly across the set
- Several bills deliberately omit fields (bill number, GST/tax details) to reflect how small unregistered vendors commonly issue bills, testing how each model handles legitimate nulls vs. hallucinated values
- One bill (`bill_15.jpg`) has a deliberately smudged amount, genuinely ambiguous between two readings — ground truth for it was resolved using the visible subtotal + GST math rather than guessing, and is discussed further in the Methodology section as a case study in handling ambiguous ground truth

Ground truth values were recorded in [`data/ground_truth.json`](file:///c:/Users/krish/Desktop/Taxor/taxor-bill-extraction/data/ground_truth.json) immediately after writing each bill, before any model extraction was run, to avoid bias.

## Methodology

Explanation of the extraction pipeline, evaluation metrics (field-level exact & fuzzy string matching), token cost calculation, and Zoho Books integration flow.

### Cost-Safety & Free-Tier Guardrails
- **Model Selection Choice:** `gemini-3.5-flash-lite` is selected as the default Gemini extractor instead of `gemini-3.6-flash`. This is a deliberate cost-safety decision to ensure compliance with Google AI Studio's free-tier quota limits without risking unexpected charges.
- **Dry-Run & Cost Ceiling:** The pipeline includes a `--dry-run` flag to pre-calculate total token and cost metrics before executing API calls. A hard safety ceiling of `$2.00` per run is enforced, requiring an explicit `--confirm` flag to proceed if exceeded.

### Zoho Books OAuth 2.0 Integration (India Data Center)
- **Token Refresh Endpoint:** `https://accounts.zoho.in/oauth/v2/token`
- **API Base URL:** `https://www.zohoapis.in/books/v3`
- **Automatic Auth Refresh:** `ZohoBooksClient` exchanges `ZOHO_REFRESH_TOKEN` for short-lived access tokens (~1 hr expiry), retrying automatically on HTTP 401.
- **Expense Creation:** `create_expense()` POSTs verified bill extractions to Zoho Books (`/expenses?organization_id=...`).

## Results

### Overall Summary & Cost Breakdown ([`results/cost_summary.csv`](file:///c:/Users/krish/Desktop/Taxor/taxor-bill-extraction/results/cost_summary.csv))

| Model | Model Identifier | Total Runs | Successful Runs | Input Tokens | Output Tokens | Avg Speed (s) | Total Cost ($) | Overall Accuracy |
|---|---|---|---|---|---|---|---|---|
| Google Gemini | `gemini-3.5-flash-lite` | 15 | 15 | 19,128 | 1,161 | 4.75s | $0.0086 | **95.62%** |
| OpenRouter | `google/gemma-4-26b-a4b-it:free` | 15 | 15 | 7,245 | 1,174 | 10.31s | $0.00 (Free) | **97.31%** |
| Anthropic Claude | `claude-sonnet-4-6` | N/A | N/A | N/A | N/A | N/A | N/A (Credit Required) | N/A |

### Per-Field Accuracy Matrix ([`results/scores.csv`](file:///c:/Users/krish/Desktop/Taxor/taxor-bill-extraction/results/scores.csv))

| Model | Vendor (%) | Bill Number (%) | Date (%) | Amount (%) | Currency (%) | Tax Details (%) | Overall Accuracy (%) |
|---|---|---|---|---|---|---|---|
| Google Gemini (`gemini-3.5-flash-lite`) | 93.33% | 93.33% | 93.33% | 100.00% | 100.00% | 93.73% | **95.62%** |
| OpenRouter (`google/gemma-4-26b-a4b-it:free`) | 93.33% | 100.00% | 100.00% | 100.00% | 100.00% | 90.53% | **97.31%** |

### Key Findings & Analysis
- **Handwriting Legibility (`bill_10.jpg`):** On `bill_10.jpg` (a vegetable vendor receipt written with non-dominant hand), Gemini misread the handwritten date digit (`2026-07-23` -> `2026-07-28`), whereas Gemma read it correctly.
- **Schema Ambiguity (`bill_03.jpg`):** Both models struggled with the auto-rickshaw fare receipt where the vendor identifier was a vehicle registration number (`KL-08-AX-4521`) rather than a conventional shop name. This represents a schema domain ambiguity rather than a clear OCR vision failure.
- **Methodology Trustworthiness & Scoring Bug Fix:** We discovered and fixed a scoring bug in `scorer.py` where near-identical bill numbers differing only in punctuation spacing (e.g. ground truth `"PH-5567"` vs extracted `"PH - 5567"`) were scored `0.0` by rigid string matching. Normalizing whitespace and applying character-level Levenshtein ratio (`fuzz.ratio`) restored evaluation trustworthiness without unfairly penalizing model outputs for formatting quirks.
- **Zoho Books Integration Verification:** Using extractions from the top-accuracy model (Gemma), 4 verified sample bills (`bill_01.jpg`, `bill_04.jpg`, `bill_05.jpg`, `bill_08.jpg`) were successfully pushed to Zoho Books India API (`https://www.zohoapis.in`), generating live expense records (IDs: `4049017000000038001`, `4049017000000039001`, `4049017000000040001`, `4049017000000041001`).

## Recommendation

**For handwritten Indian bills specifically, I'd recommend Gemma 4 26B A4B
(via OpenRouter's free tier) over Gemini 3.5 Flash-Lite**, based on the
numbers above — but the choice is closer than the headline accuracy gap
suggests, and comes with real caveats.

**Accuracy:** Gemma scored 97.31% overall vs Gemini's 95.62%, and the gap
widens specifically on the harder handwriting in this dataset — most
notably bill_10, where Gemini misread a handwritten date digit (23 → 28)
that Gemma read correctly. Since the whole point of this evaluation is
performance on *difficult* handwriting, not average-case bills, I weight
that specific result more heavily than the aggregate number alone.

**Cost:** Gemini cost $0.0086 for all 15 bills (~$0.057 per 100 bills
extrapolated); Gemma was $0.00 on OpenRouter's free tier. At this scale,
cost is a non-factor either way — both are effectively free for a
small-business use case processing a few hundred bills a month.

**Speed:** This is where the tradeoff actually bites. Gemini averaged
4.75s per bill; Gemma averaged 10.31s — more than double. For a production
tool where someone is uploading a bill and waiting for a result, that
difference is noticeable. This is the real cost of Gemma's higher
accuracy here: latency, not money.

**Same model for both digital and handwritten documents, or different
pipelines?** I'd use different pipelines. Digital/typed invoices are a
much easier extraction problem — nearly any current vision model gets
those close to perfect, so the cheaper, faster option (Gemini in this
case) makes sense there. Handwritten bills are the harder case this
assignment specifically tests, and that's where the accuracy difference
actually shows up and matters. A production system could route on
document type: fast/cheap model for digital, slower/more accurate model
for handwritten — using the extraction confidence or a simple heuristic
(e.g. edge detection for print vs. cursive strokes) to decide which
pipeline to send a bill through.

**Caveat on Gemma's free-tier status:** OpenRouter's free models are
rate-limited and not intended for production volume — the $0.00 cost here
reflects an evaluation environment, not a sustainable production
deployment. At real scale, Gemma via a paid tier (or a comparable
paid model) would need its own cost analysis before being the final
production choice.




## Limitations

- **Small dataset (15 bills).** Accuracy percentages at this scale can
  shift meaningfully with just one or two extra errors — a 95% vs 97%
  gap on 15 bills is a difference of roughly one field extraction, not a
  statistically robust signal. A production evaluation would need
  hundreds of bills before these numbers should be trusted as stable.

- **Self-created rather than sourced dataset.** I handwrote all 15 bills
  myself, using fictional vendor names and invented amounts, rather than
  sourcing real bills — this was a deliberate choice given the assignment
  timeframe and the difficulty of getting genuine bills with proper
  consent/redaction on short notice. I deliberately varied handwriting
  style, pen/pencil, lighting, and photo angle to simulate real-world
  difficulty, but self-created data can't fully replicate the messiness
  of genuine years-old handwritten shop receipts.

- **Claude was excluded from the final comparison.** I set up an Anthropic
  Console account and implemented the Claude extractor, but the account's
  trial credit wasn't available/didn't activate, and I didn't want to
  spend real money to complete a non-funded student assignment. The
  extractor code is complete and untested-live in the repo — it would
  work with a funded account.

- **One bill (bill_15) has genuinely ambiguous ground truth.** I
  deliberately smudged a handwritten amount to test how models handle
  illegible input. I set ground truth to ₹784 based on the visible
  subtotal + GST math, but the raw digit itself is honestly readable as
  either ₹784 or ₹789 — meaning any score on that specific field reflects
  my own judgment call as much as the model's actual accuracy.

- **Scoring methodology is my own design choice, not a standard.** I used
  fuzzy string matching (90%/60% thresholds) for text fields, exact
  matching for dates, and 5% numeric tolerance for amounts. These
  thresholds are reasonable but not the only valid choice — a stricter or
  looser standard would shift the accuracy numbers somewhat. I found and
  fixed one real scoring bug during development (bill_number IDs like
  "PH-5567" scoring 0 against "PH - 5567" due to over-strict tokenized
  matching) — I mention this because it's a good reminder that eval
  code itself needs the same scrutiny as the models being evaluated.

- **OpenRouter's free tier isn't representative of production behavior.**
  Gemma's $0.00 cost and higher latency reflect a rate-limited free
  service, not what a paid, production-grade deployment of a comparable
  model would actually cost or how fast it would run.

- **Zoho Books integration was tested on 4 bills, not all 15.** I pushed
  the 4 highest-confidence (100%-accuracy) extractions as a proof of
  concept rather than all 15, to avoid cluttering the test Zoho
  organization with lower-quality entries during development.
