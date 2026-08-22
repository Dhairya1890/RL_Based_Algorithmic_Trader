"""
Step 4 — Score all headlines in combined_raw.csv using Groq (primary)
and Gemini (fallback). Saves progress every 100 batches so it is resumable
if it crashes or hits quota.

Provider strategy:
  - Primary:  Groq llama-3.3-70b (30 RPM, 14,400 req/day — free)
  - Fallback: Gemini 1.5-flash   (15 RPM — free)

Output: data/raw/scored_raw.csv
  Adds two columns to combined_raw.csv:
    sentiment_score : float [-1.0, +1.0]
    magnitude       : float [0.0,  1.0]

Usage: python src/05_score_sentiment.py
Prereq: 04_merge_clean.py must have been run first.
"""

import json
import time
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from groq import Groq
from config import (
    RAW_DIR,
    GEMINI_API_KEYS,
    GEMINI_MODEL,
    GROQ_API_KEYS,
    GROQ_MODEL,
    GROQ_RPM,
    GEMINI_RPM,
    GEMINI_BATCH_SIZE,
)

COMBINED_RAW_PATH = RAW_DIR / "combined_raw.csv"
SCORED_RAW_PATH   = RAW_DIR / "scored_raw.csv"
CHECKPOINT_PATH   = RAW_DIR / "scoring_checkpoint.csv"
FAILED_PATH       = RAW_DIR / "scoring_failed.csv"

BATCH_SIZE        = GEMINI_BATCH_SIZE  # 10 headlines per API call
CHECKPOINT_EVERY  = 100               # save progress every N batches

# ── Scoring rubric prompt (same for both providers) ───────────────────────────
SYSTEM_PROMPT = """You are a financial sentiment scorer for Indian equity markets.

Score news headlines based on their likely impact on the mentioned company's stock price.

SCORING RULES:

sentiment_score (float, -1.0 to +1.0):
  Positive = news likely to push stock UP
  Negative = news likely to push stock DOWN
  Zero     = no clear price impact

  Reference points:
  +0.9 to +1.0 : Record earnings beat, major acquisition, large confirmed contract
  +0.5 to +0.8 : Dividend increase, guidance raised, analyst upgrade, regulatory approval
  +0.1 to +0.4 : Minor positive update, vague partnership, routine positive filing
   0.0          : Board meeting notice, AGM date, no financial implication
  -0.1 to -0.4 : Minor negative, analyst downgrade, vague regulatory concern
  -0.5 to -0.8 : Earnings miss, guidance cut, credit downgrade, leadership departure
  -0.9 to -1.0 : Fraud investigation, major penalty, sudden CEO resignation, catastrophic miss

magnitude (float, 0.0 to 1.0):
  How strong and certain is the signal?
  0.7-1.0 : Specific figures mentioned, confirmed major event (earnings, M&A, fraud)
  0.4-0.6 : Some specifics, partially confirmed
  0.0-0.3 : Vague, speculative, routine, no figures mentioned

Return ONLY a valid JSON array with one object per headline, in the same order.
Each object must have exactly two keys: sentiment_score and magnitude.
No explanation. No markdown. No extra text."""


def build_user_prompt(headlines: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return f"Score these {len(headlines)} headlines:\n\n{numbered}"


def parse_response(text: str, expected_count: int) -> list[dict] | None:
    """Extract and validate JSON array from model response."""
    # Strip markdown fences if present
    text = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return None
        if len(data) != expected_count:
            return None
        for item in data:
            if "sentiment_score" not in item or "magnitude" not in item:
                return None
            # Clamp values to valid range
            item["sentiment_score"] = max(-1.0, min(1.0, float(item["sentiment_score"])))
            item["magnitude"]       = max(0.0,  min(1.0, float(item["magnitude"])))
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

class ScorerClient:
    """Wraps a single API key for either Groq or Gemini."""

    def __init__(self, provider: str, api_key: str):
        self.provider  = provider
        self.api_key   = api_key
        self.last_call = 0.0
        self.cooldown_until = 0.0  # epoch time when this key is usable again

        if provider == "groq":
            self.client = Groq(api_key=api_key)
            self.min_gap = 60.0 / 30   # 30 RPM
        else:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
            )
            self.min_gap = 60.0 / 15   # 15 RPM

    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def score_batch(self, headlines: list[str]) -> list[dict] | None:
        # Respect per-key rate limit
        elapsed = time.time() - self.last_call
        gap = self.min_gap - elapsed
        if gap > 0:
            time.sleep(gap)

        try:
            if self.provider == "groq":
                resp = self.client.chat.completions.create(
                    model       = GROQ_MODEL,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(headlines)},
                    ],
                    temperature = 0.1,
                    max_tokens  = 500,
                )
                self.last_call = time.time()
                return parse_response(resp.choices[0].message.content, len(headlines))

            else:  # gemini
                resp = self.client.generate_content(
                    build_user_prompt(headlines),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=500,
                    ),
                )
                self.last_call = time.time()
                return parse_response(resp.text, len(headlines))

        except Exception as e:
            self.last_call = time.time()
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "quota" in err:
                # Cool this key down for 60 seconds
                self.cooldown_until = time.time() + 60
                tqdm.write(f"  [429] {self.provider} key ...{self.api_key[-6:]} cooling 60s")
            else:
                tqdm.write(f"  [ERR] {self.provider} key ...{self.api_key[-6:]}: {e}")
            return None


class RotatingScorer:
    """Rotates across all Groq and Gemini keys, skipping cooled-down ones."""

    def __init__(self):
        self.clients = []
        for key in GROQ_API_KEYS:
            self.clients.append(ScorerClient("groq", key))
        for key in GEMINI_API_KEYS:
            self.clients.append(ScorerClient("gemini", key))

        self.current = 0
        tqdm.write(f"Loaded {len(GROQ_API_KEYS)} Groq keys + {len(GEMINI_API_KEYS)} Gemini keys")
        tqdm.write(f"Total pool: {len(self.clients)} API clients\n")

    def _next_available(self) -> ScorerClient | None:
        """Find next available client that is not in cooldown."""
        for _ in range(len(self.clients)):
            self.current = (self.current + 1) % len(self.clients)
            client = self.clients[self.current]
            if client.is_available():
                return client
        return None  # all clients in cooldown

    def score_batch(self, headlines: list[str]) -> list[dict] | None:
        # Try current client first
        client = self.clients[self.current]

        if not client.is_available():
            client = self._next_available()

        if client is None:
            # All keys cooling down, find minimum wait time
            wait = min(c.cooldown_until for c in self.clients) - time.time()
            tqdm.write(f"  [ALL COOLING] waiting {wait:.0f}s for a key to free up")
            time.sleep(max(wait, 1))
            client = self._next_available()

        if client is None:
            return None

        result = client.score_batch(headlines)

        # On failure rotate immediately
        if result is None:
            next_client = self._next_available()
            if next_client:
                self.current = self.clients.index(next_client)
                result = next_client.score_batch(headlines)

        return result


# ── Main scoring loop ─────────────────────────────────────────────────────────
def score_all(df: pd.DataFrame) -> pd.DataFrame:
    scorer = RotatingScorer()

    scores     = [None] * len(df)
    magnitudes = [None] * len(df)
    failed_idx = []

    # Check for existing checkpoint and resume
    start_batch = 0
    if CHECKPOINT_PATH.exists():
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        scored_count = checkpoint["sentiment_score"].notna().sum()
        start_batch  = scored_count // BATCH_SIZE
        # Reload existing scores
        for i, row in checkpoint.iterrows():
            if pd.notna(row.get("sentiment_score")):
                scores[i]     = row["sentiment_score"]
                magnitudes[i] = row["magnitude"]
        tqdm.write(f"Resuming from batch {start_batch} ({scored_count} already scored)")

    # Build batches
    indices = list(range(len(df)))
    batches = [indices[i:i+BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]
    batches = batches[start_batch:]  # skip already-scored batches

    print(f"\nScoring {len(df):,} headlines in {len(batches)} batches")
    print(f"Primary: Groq ({GROQ_MODEL}) | Fallback: Gemini ({GEMINI_MODEL})\n")

    for batch_num, batch_idx in enumerate(tqdm(batches, desc="Batches")):
        headlines = df.iloc[batch_idx]["headline"].tolist()

        result = scorer.score_batch(headlines)
        if result is not None:
            for local_i, global_i in enumerate(batch_idx):
                scores[global_i] = result[local_i]["sentiment_score"]
                magnitudes[global_i] = result[local_i]["magnitude"]
                
        if result is None:
            tqdm.write(f"  [FAILED] batch {batch_num + start_batch}, marking as failed")
            failed_idx.extend(batch_idx)
            for global_i in batch_idx:
                scores[global_i]     = 0.0
                magnitudes[global_i] = 0.0

        # Save checkpoint every N batches
        if (batch_num + 1) % CHECKPOINT_EVERY == 0:
            df_checkpoint = df.copy()
            df_checkpoint["sentiment_score"] = scores
            df_checkpoint["magnitude"]       = magnitudes
            df_checkpoint.to_csv(CHECKPOINT_PATH, index=False)
            tqdm.write(f"  [CHECKPOINT] saved at batch {batch_num + start_batch}")

    df["sentiment_score"] = scores
    df["magnitude"]       = magnitudes

    # Save failed batches for inspection
    if failed_idx:
        df.iloc[failed_idx].to_csv(FAILED_PATH, index=False)
        tqdm.write(f"\n[WARN] {len(failed_idx)} headlines failed — saved to {FAILED_PATH}")

    return df


def main():
    if not COMBINED_RAW_PATH.exists():
        raise FileNotFoundError(
            f"combined_raw.csv not found at {COMBINED_RAW_PATH}\n"
            "Run 04_merge_clean.py first."
        )

    print("Loading combined_raw.csv...")
    df = pd.read_csv(COMBINED_RAW_PATH)
    print(f"Loaded {len(df):,} headlines\n")

    df = score_all(df)

    df.to_csv(SCORED_RAW_PATH, index=False)

    # Clean up checkpoint on successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n{'='*60}")
    print(f"Scoring complete")
    print(f"{'='*60}")
    print(f"Total scored        : {df['sentiment_score'].notna().sum():,}")
    print(f"Failed              : {(df['sentiment_score'] == 0.0).sum():,}")
    print(f"\nScore distribution:")
    print(f"  Positive (>0)     : {(df['sentiment_score'] > 0).sum():,}")
    print(f"  Neutral  (=0)     : {(df['sentiment_score'] == 0).sum():,}")
    print(f"  Negative (<0)     : {(df['sentiment_score'] < 0).sum():,}")
    print(f"\nAverage score      : {df['sentiment_score'].mean():.3f}")
    print(f"Average magnitude  : {df['magnitude'].mean():.3f}")
    print(f"\nSaved to: {SCORED_RAW_PATH}")
    print("\nRun 06_aggregate.py next.")


if __name__ == "__main__":
    main()