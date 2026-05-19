import streamlit as st
import pandas as pd
import anthropic
from anthropic import AuthenticationError, RateLimitError, APIError, APIConnectionError
import json
import time
import io
from typing import Any

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Review Classifier",
    page_icon="🔬",
    layout="centered",
)

# ─── VALID OUTPUT VALUES ───────────────────────────────────────────────────────
VALID_TYPES           = {"FUNCTIONAL", "QUALITY", "UX", "EXPERIENCE"}
VALID_EMOTIONAL_NEEDS = {"AUTONOMY", "COMPETENCE", "RELATEDNESS", "SECURITY", "STIMULATION", "POPULARITY"}
VALID_SENTIMENTS      = {"POSITIVE", "NEGATIVE"}
VALID_CONFIDENCE      = {"HIGH", "MEDIUM", "LOW"}

# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 400
TEMPERATURE = 0.0

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
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
        "cache_control": {"type": "ephemeral"},
    }
]

# ─── USER PROMPT TEMPLATE ─────────────────────────────────────────────────────
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

    if data.get("llm_type") not in VALID_TYPES:
        issues.append(f"invalid llm_type: {data.get('llm_type')}")

    emotional_need = data.get("llm_emotional_need")
    if data.get("llm_type") == "EXPERIENCE":
        if emotional_need not in VALID_EMOTIONAL_NEEDS:
            issues.append(f"invalid llm_emotional_need for EXPERIENCE: {emotional_need}")
    else:
        if emotional_need is not None:
            issues.append(f"llm_emotional_need should be null for type {data.get('llm_type')}, got: {emotional_need}")

    if data.get("llm_sentiment") not in VALID_SENTIMENTS:
        issues.append(f"invalid llm_sentiment: {data.get('llm_sentiment')}")

    if data.get("llm_confidence") not in VALID_CONFIDENCE:
        issues.append(f"invalid llm_confidence: {data.get('llm_confidence')}")

    if issues:
        data["validation_warnings"] = "; ".join(issues)
    return data


def classify_review(
    client: anthropic.Anthropic,
    review_text: str,
    app_name: str,
    score: str,
) -> dict[str, Any]:
    """Send one review to Claude and return a validated, parsed classification."""
    prompt = USER_PROMPT_TEMPLATE.format(
        review_text=review_text,
        app_name=app_name,
        score=score,
    )
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT_BLOCKS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        return validate_classification(parsed)

    except RateLimitError as e:
        return {"error": f"RateLimitError after retries: {str(e)}"}
    except APIError as e:
        return {"error": f"APIError {e.status_code}: {e.message}"}
    except APIConnectionError as e:
        return {"error": f"APIConnectionError: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed: {raw[:200]}"}


# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

st.title("🔬 LLM Review Classifier")
st.markdown(
    "Upload your raw reviews CSV, classify each review with Claude, "
    "and download the results."
)

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your key is used only for this session and never stored.",
    )
    st.markdown("---")
    st.caption("Model: claude-haiku-4-5 · Temp: 0.0 · Max tokens: 400")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload `reviews_raw.csv`",
    type=["csv"],
    help="Must contain columns: review_id, app_name, score, content",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = {"review_id", "app_name", "score", "content"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    st.success(f"Loaded **{len(df)} reviews** from `{uploaded_file.name}`")

    with st.expander("Preview raw data (first 5 rows)"):
        st.dataframe(df.head(5), use_container_width=True)

    st.markdown("---")

    # ── Process button ────────────────────────────────────────────────────────
    if st.button("▶ Start Classification", type="primary", use_container_width=True):

        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar first.")
            st.stop()

        try:
            client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        except AuthenticationError:
            st.error("Invalid API key — check it and try again.")
            st.stop()

        total = len(df)
        results: list[dict[str, Any]] = []

        # Live progress UI elements
        st.markdown("### Processing")
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        error_counter = st.empty()
        errors_so_far = 0

        for i, row in enumerate(df.itertuples()):
            review_text = str(row.content)
            app_name    = str(row.app_name)
            score       = str(row.score)
            review_id   = str(row.review_id)

            # Update status before the API call so the user sees what's running
            status_text.markdown(
                f"**{i + 1} / {total}** &nbsp;·&nbsp; `{app_name}` &nbsp;·&nbsp; "
                f"_{review_text[:80].strip()}…_"
            )

            classification = classify_review(client, review_text, app_name, score)

            if classification.get("error"):
                errors_so_far += 1
                error_counter.warning(f"⚠ {errors_so_far} error(s) so far — will be marked in the CSV.")

            record: dict[str, Any] = {
                "review_id"             : review_id,
                "app_name"              : app_name,
                "score"                 : score,
                "content"               : review_text,
                "llm_type"              : classification.get("llm_type", "ERROR"),
                "llm_emotional_need"    : classification.get("llm_emotional_need"),
                "llm_sentiment"         : classification.get("llm_sentiment", "ERROR"),
                "llm_dark_pattern"      : classification.get("llm_dark_pattern"),
                "llm_dark_description"  : classification.get("llm_dark_description", ""),
                "llm_confidence"        : classification.get("llm_confidence", "ERROR"),
                "llm_reasoning"         : classification.get("llm_reasoning", ""),
                "error"                 : classification.get("error", ""),
                "validation_warnings"   : classification.get("validation_warnings", ""),
            }
            results.append(record)

            progress_bar.progress((i + 1) / total)
            time.sleep(0.3)

        # ── Done ──────────────────────────────────────────────────────────────
        status_text.success(f"✅ Done! Classified {total} reviews — {errors_so_far} error(s).")

        final_df = pd.DataFrame(results)

        # Download button — encode to bytes so st.download_button can serve it
        csv_bytes = final_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇ Download reviews_classified.csv",
            data=csv_bytes,
            file_name="reviews_classified.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")

        # ── Summary stats ──────────────────────────────────────────────────────
        st.markdown("### Classification Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", total)
        col2.metric("Errors", errors_so_far)
        col3.metric("Validation Warnings", int((final_df["validation_warnings"] != "").sum()))
        col4.metric("Dark Patterns", int(final_df["llm_dark_pattern"].eq(True).sum()))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Requirement Type**")
            st.dataframe(
                final_df["llm_type"].value_counts().rename_axis("Type").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Sentiment**")
            st.dataframe(
                final_df["llm_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("**Emotional Need** *(EXPERIENCE only)*")
            exp = final_df[final_df["llm_type"] == "EXPERIENCE"]
            st.dataframe(
                exp["llm_emotional_need"].value_counts().rename_axis("Need").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Confidence**")
            st.dataframe(
                final_df["llm_confidence"].value_counts().rename_axis("Confidence").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")
        st.markdown("### Full Results")
        st.dataframe(final_df, use_container_width=True)
