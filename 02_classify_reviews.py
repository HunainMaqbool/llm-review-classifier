# Anthropic SDK — the main client for sending requests to Claude
import anthropic
# Import specific error types so each failure can be handled differently
# (wrong key = crash immediately, rate limit = retry, API error = log and skip)
from anthropic import AuthenticationError, RateLimitError, APIError

# pandas reads the CSV of 800 reviews and writes the classified output CSV
import pandas as pd
# json parses Claude's text response into a Python dict we can store
import json
# os checks whether the checkpoint file exists so we can resume mid-run
import os
# time.sleep adds a small delay between API calls to stay within rate limits
import time
# tqdm draws the progress bar so we can see how many reviews are left
from tqdm import tqdm
# Any is used in type hints because the classification dict has mixed value types
from typing import Any

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# All file paths and model settings in one place — change here, affects everything
INPUT_FILE       = "data/reviews_raw.csv"      # output from Task 1 (raw reviews)
OUTPUT_FILE      = "data/reviews_classified.csv"  # final research dataset
CHECKPOINT_FILE  = "data/checkpoint.csv"       # crash-recovery save file
CHECKPOINT_EVERY = 50                          # save progress every 50 reviews so a crash loses at most 50
MODEL            = "claude-haiku-4-5-20251001" # Haiku is fast and cheap — enough for structured classification
MAX_TOKENS       = 400                         # JSON response is ~200 tokens; 400 gives room for verbose reasoning
TEMPERATURE      = 0.0   # deterministic output for classification tasks
MAX_RETRIES      = 3     # SDK retries with exponential backoff on transient errors
# ──────────────────────────────────────────────────────────────────────────────

# max_retries tells the SDK to automatically retry failed requests (with backoff)
# — this handles temporary Anthropic server blips without us writing retry loops
client = anthropic.Anthropic(max_retries=MAX_RETRIES)

# ─── VALID OUTPUT VALUES ───────────────────────────────────────────────────────
# These sets define the only legal values the LLM is allowed to return.
# Used in validate_classification() to catch hallucinated or misspelled labels
# before they silently corrupt the research CSV.
VALID_TYPES           = {"FUNCTIONAL", "QUALITY", "UX", "EXPERIENCE"}
VALID_EMOTIONAL_NEEDS = {"AUTONOMY", "COMPETENCE", "RELATEDNESS", "SECURITY", "STIMULATION", "POPULARITY"}
VALID_SENTIMENTS      = {"POSITIVE", "NEGATIVE"}
VALID_CONFIDENCE      = {"HIGH", "MEDIUM", "LOW"}
# ──────────────────────────────────────────────────────────────────────────────

# ─── SYSTEM MESSAGE ───────────────────────────────────────────────────────────
# This block is the same for every one of the 800 API calls.
# cache_control: "ephemeral" tells Anthropic's API to cache this exact text after
# the first call — subsequent calls reuse the cached version instead of re-reading
# ~800 tokens every time. This cuts input token cost by ~60% across the full run.
SYSTEM_PROMPT_BLOCKS: list[dict[str, Any]] = [
    {
        "type": "text",
        "text": (
            "You are an expert requirements engineering researcher specializing in "
            "mobile app user feedback analysis.\n\n"
            "Your job is to classify app reviews for an academic research paper. "
            "You must be precise, consistent, and objective. "
            "You always respond with valid JSON only — no explanation, no markdown, "
            "no prose outside the JSON object."
        ),
        # ephemeral cache lives for 5 minutes — long enough to cover the full 800-review run
        "cache_control": {"type": "ephemeral"},
    }
]

# ─── USER PROMPT TEMPLATE ─────────────────────────────────────────────────────
# This is the per-review prompt. The three {placeholders} are filled in at runtime.
# Structure: TASK → DEFINITIONS → FEW-SHOT EXAMPLES → REVIEW → OUTPUT FORMAT
# The few-shot examples are critical — without them the model often invents new
# category names or fills emotional_need even when llm_type is not EXPERIENCE.
USER_PROMPT_TEMPLATE = """## TASK
Classify the app review below along exactly 5 dimensions. Return ONLY a valid JSON object matching the format shown in the examples.

---

## DEFINITIONS

**llm_type** — What kind of need does the review primarily express? Classify based on the DOMINANT theme across the whole review, not isolated sentences.
- FUNCTIONAL: a feature is broken, missing, or not working as expected (bugs, crashes attributed to a specific feature)
- QUALITY: about performance, speed, reliability, battery drain, or general stability
- UX: about usability, interface layout, navigation flow, confusing design, or visual clarity
- EXPERIENCE: the user expresses an emotional or psychological feeling, need, or desire
- IMPORTANT: If most of the review discusses usability or interface design but ends with one emotional sentence, classify as UX. Only use EXPERIENCE when emotional language is the primary message, not just present somewhere in the review.

**llm_emotional_need** — ONLY fill this if llm_type is EXPERIENCE. Set to null for all other types.
- AUTONOMY: user wants control, freedom, or customization over their experience
- COMPETENCE: user wants to feel skilled, track progress, or achieve goals
- RELATEDNESS: user wants social connection, sharing with others, or community belonging
- SECURITY: user wants safety, privacy, data protection, or stability
- STIMULATION: user wants excitement, variety, novelty, or entertainment
- POPULARITY: user wants recognition, status, likes, or leaderboard visibility

**llm_sentiment** — Is the user's overall tone positive or negative about the app overall?
- POSITIVE: satisfied, praising, happy — use this when the review is 4-5 stars and the dominant message is praise, even if it contains one complaint or feature request
- NEGATIVE: dissatisfied, complaining, frustrated — use this when the primary message is frustration or dissatisfaction, regardless of star rating
- IMPORTANT: Do NOT let a single negative sentence override an otherwise positive review. Judge the overall emotional tone across the entire review, weighted by the star rating. A 4-5 star review with one complaint is POSITIVE.

**llm_dark_pattern** — Does the app appear to manipulate user emotions through deceptive or coercive design?
- true ONLY if the review clearly references: fake urgency ("you'll lose your streak!"), guilt-tripping language, forced or excessive notifications designed to create anxiety, social pressure manipulation, or hidden charges
- false if there is no clear evidence of emotional manipulation

**llm_confidence** — How confident are you in this classification?
- HIGH: the review is clear and unambiguous — one category clearly dominates
- MEDIUM: the review could fit more than one category — use this when two llm_type values are equally plausible, OR when two emotional needs are equally present in an EXPERIENCE review (e.g. both COMPETENCE and RELATEDNESS are mentioned with similar weight)
- LOW: the review is too short, vague, or mixed to classify reliably

---

## FEW-SHOT EXAMPLES

**Example 1**
Review: "The GPS tracking keeps stopping mid-run. I've lost three workouts this week because the app just freezes."
App: Strava | Rating: 1/5

```json
{{
  "llm_type": "FUNCTIONAL",
  "llm_emotional_need": null,
  "llm_sentiment": "NEGATIVE",
  "llm_dark_pattern": false,
  "llm_dark_description": "",
  "llm_confidence": "HIGH",
  "llm_reasoning": "User reports a specific feature (GPS tracking) failing repeatedly, which is a functional defect."
}}
```

**Example 2**
Review: "I love how this app makes me feel like I'm part of a real running community. Seeing my friends' activities every morning motivates me."
App: Strava | Rating: 5/5

```json
{{
  "llm_type": "EXPERIENCE",
  "llm_emotional_need": "RELATEDNESS",
  "llm_sentiment": "POSITIVE",
  "llm_dark_pattern": false,
  "llm_dark_description": "",
  "llm_confidence": "HIGH",
  "llm_reasoning": "User expresses a desire for social belonging and community connection, which maps to the RELATEDNESS emotional need."
}}
```

---

## REVIEW TO CLASSIFY

App: {app_name}
Star rating: {score}/5
Review: \"\"\"{review_text}\"\"\"

---

## OUTPUT FORMAT

Respond with ONLY this JSON object — no text before or after it:

{{
  "llm_type": "FUNCTIONAL | QUALITY | UX | EXPERIENCE",
  "llm_emotional_need": "AUTONOMY | COMPETENCE | RELATEDNESS | SECURITY | STIMULATION | POPULARITY | null",
  "llm_sentiment": "POSITIVE | NEGATIVE",
  "llm_dark_pattern": true or false,
  "llm_dark_description": "one sentence if dark pattern is true, otherwise empty string",
  "llm_confidence": "HIGH | MEDIUM | LOW",
  "llm_reasoning": "one sentence explaining the primary reason for your llm_type classification"
}}"""


def validate_classification(data: dict[str, Any]) -> dict[str, Any]:
    """Check parsed JSON values against known valid sets; flag invalid ones."""
    issues: list[str] = []

    # Check llm_type is one of the 4 allowed values
    if data.get("llm_type") not in VALID_TYPES:
        issues.append(f"invalid llm_type: {data.get('llm_type')}")

    emotional_need = data.get("llm_emotional_need")
    if data.get("llm_type") == "EXPERIENCE":
        # When type is EXPERIENCE, emotional_need must be one of the 6 valid options
        if emotional_need not in VALID_EMOTIONAL_NEEDS:
            issues.append(f"invalid llm_emotional_need for EXPERIENCE: {emotional_need}")
    else:
        # For all other types, emotional_need must be null — flag it if the model filled it anyway
        if emotional_need is not None:
            issues.append(f"llm_emotional_need should be null for type {data.get('llm_type')}, got: {emotional_need}")

    # Check remaining fields against their valid sets
    if data.get("llm_sentiment") not in VALID_SENTIMENTS:
        issues.append(f"invalid llm_sentiment: {data.get('llm_sentiment')}")

    if data.get("llm_confidence") not in VALID_CONFIDENCE:
        issues.append(f"invalid llm_confidence: {data.get('llm_confidence')}")

    # Attach warnings to the dict so they land in the output CSV as a dedicated column
    # — this lets the researcher filter and audit bad classifications without re-running everything
    if issues:
        data["validation_warnings"] = "; ".join(issues)
    return data


def classify_review(review_text: str, app_name: str, score: str) -> dict[str, Any]:
    """Send one review to Claude and return a validated, parsed classification."""
    # Fill in the three per-review placeholders — everything else in the prompt is static
    prompt = USER_PROMPT_TEMPLATE.format(
        review_text=review_text,
        app_name=app_name,
        score=score,
    )
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,   # 0.0 = deterministic; same review always gives same label
            system=SYSTEM_PROMPT_BLOCKS,  # passed as a list to enable per-block cache_control
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Claude sometimes wraps JSON in ```json ... ``` fences despite being told not to.
        # Strip them so json.loads() doesn't choke on the non-JSON characters.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        # Validate before returning so bad values are flagged at the source, not discovered later
        return validate_classification(parsed)

    except AuthenticationError as e:
        # Wrong API key — every future call will also fail, so crash immediately
        # rather than letting the script burn through 800 failed attempts
        raise SystemExit(f"Authentication failed — check ANTHROPIC_API_KEY: {e}") from e

    except RateLimitError as e:
        # The SDK already tried MAX_RETRIES times with exponential backoff and all failed.
        # Log and return an error record so this review is marked in the CSV — it can be
        # re-classified in a follow-up run since checkpoint logic will skip already-done rows.
        print(f"\nRATE LIMIT: exhausted retries — skipping review. Error: {e}")
        return {"error": f"RateLimitError after retries: {str(e)}"}

    except APIError as e:
        # Catch-all for server errors (5xx), malformed requests (4xx), etc.
        print(f"\nAPI ERROR {e.status_code}: {e.message}")
        return {"error": f"APIError {e.status_code}: {e.message}"}

    except json.JSONDecodeError as e:
        # Claude returned text that isn't valid JSON — happens on very short or garbled responses.
        # Truncate raw to 200 chars so the error log stays readable in the terminal.
        print(f"\nJSON PARSE ERROR: {e}")
        return {"error": f"JSON parse failed: {raw[:200]}"}


def main() -> None:
    # ── Phase 1: Load the raw reviews CSV produced by Task 1 ──────────────────
    print(f"Loading reviews from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  → {len(df)} reviews loaded.")

    # ── TESTING MODE: uncomment the next line to test with 10 reviews only ──
    # df = df.head(50)  # REMOVE AFTER TESTING

    # ── Phase 2: Resume from checkpoint if a previous run was interrupted ─────
    # already_done tracks review_ids we've classified so we can skip them this run
    already_done: set[str] = set()
    results: list[dict[str, Any]] = []
    if os.path.exists(CHECKPOINT_FILE):
        done_df = pd.read_csv(CHECKPOINT_FILE)
        already_done = set(done_df["review_id"].astype(str))
        # Pre-fill results with the already-classified rows so the final CSV is complete
        results = done_df.to_dict("records")
        print(f"  → Resuming: {len(already_done)} reviews already classified.")

    # ── Phase 3: Classify only the reviews not yet in the checkpoint ──────────
    # The ~ operator inverts the boolean mask — keeps rows whose id is NOT in already_done
    to_process = df[~df["review_id"].astype(str).isin(already_done)]
    print(f"  → {len(to_process)} reviews left to classify.\n")

    for i, row in enumerate(tqdm(to_process.itertuples(), total=len(to_process), desc="Classifying")):
        # Cast every field to str — CSV values can come in as int/float and break .format()
        review_text: str = str(row.content)
        app_name: str    = str(row.app_name)
        score: str       = str(row.score)
        review_id: str   = str(row.review_id)

        classification = classify_review(review_text, app_name, score)

        # Build a flat record that maps directly to one row in the output CSV
        record: dict[str, Any] = {
            "review_id"             : review_id,
            "app_name"              : app_name,
            "score"                 : score,
            "content"               : review_text,
            "llm_type"              : classification.get("llm_type", "ERROR"),
            "llm_emotional_need"    : classification.get("llm_emotional_need"),        # intentionally None if missing
            "llm_sentiment"         : classification.get("llm_sentiment", "ERROR"),
            "llm_dark_pattern"      : classification.get("llm_dark_pattern"),          # intentionally None if missing
            "llm_dark_description"  : classification.get("llm_dark_description", ""),
            "llm_confidence"        : classification.get("llm_confidence", "ERROR"),
            "llm_reasoning"         : classification.get("llm_reasoning", ""),
            "error"                 : classification.get("error", ""),
            "validation_warnings"   : classification.get("validation_warnings", ""),
        }
        results.append(record)

        # Periodic checkpoint — if the script dies at review 237, the next run starts at 201
        # (the last multiple of 50), losing at most CHECKPOINT_EVERY reviews of work
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
            tqdm.write(f"  ✓ Checkpoint saved at {len(results)} reviews.")

        # Small fixed delay to stay well within Anthropic's rate limit for Haiku
        time.sleep(0.3)

    # ── Phase 4: Write the final CSV with UTF-8-sig encoding ──────────────────
    # utf-8-sig adds a BOM character so Excel opens the CSV without garbling special characters
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ Done! Output saved to: {OUTPUT_FILE}")

    # ── Phase 5: Print a distribution summary for a quick sanity check ────────
    # If one category dominates (e.g. 95% FUNCTIONAL), something went wrong with the prompt
    print("\n─── CLASSIFICATION SUMMARY ───────────────────────────────")
    print(f"Total reviews classified  : {len(final_df)}")
    print(f"Errors                    : {(final_df['error'] != '').sum()}")
    print(f"Validation warnings       : {(final_df['validation_warnings'] != '').sum()}")
    print("\nRequirement Type distribution:")
    print(final_df["llm_type"].value_counts().to_string())
    print("\nEmotional Need distribution (EXPERIENCE reviews only):")
    exp = final_df[final_df["llm_type"] == "EXPERIENCE"]
    print(exp["llm_emotional_need"].value_counts().to_string())
    print("\nSentiment distribution:")
    print(final_df["llm_sentiment"].value_counts().to_string())
    print("\nDark Pattern detected:")
    print(final_df["llm_dark_pattern"].value_counts().to_string())
    print("\nConfidence distribution:")
    print(final_df["llm_confidence"].value_counts().to_string())
    print("──────────────────────────────────────────────────────────")


# Standard Python entry point — only runs main() when this file is executed directly,
# not when it is imported as a module by another script or test
if __name__ == "__main__":
    main()
