"""
generate_from_english.py — EdgeSanskrit English-to-Sanskrit-Audio Pipeline
============================================================================
Pipeline:
  1. Take English text (hardcoded or --text argument)
  2. Translate to Sanskrit Devanagari via OpenRouter (with model fallback)
  3. Feed Sanskrit into IndicF5 CPU engine
  4. Save output WAV
"""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# Force UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VAGDHENU_SRC"] = os.path.join(HERE, "vagdhenu", "src")

# ── English text (edit freely) ────────────────────────────────────────────────
ENGLISH_TEXT = """\
Chapter 1: The World That Rejected Miracles

The first thing I heard after dying was someone screaming.

Your Highness! He's breathing!

I opened my eyes. Stone ceiling. Golden chandeliers. The smell of blood.

A young woman in white robes was crying beside the bed while several armored knights
pointed their swords at me.

Impossible. The prince died. So who is he?

I had no answer.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Translate via OpenRouter (Gemini Flash → Claude Haiku → Mistral 7B)
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("google/gemini-flash-1.5",        "Gemini Flash 1.5"),
    ("anthropic/claude-3-haiku",       "Claude 3 Haiku"),
    ("mistralai/mistral-7b-instruct",  "Mistral 7B Instruct"),
]

def translate_to_sanskrit(english: str, context: str = "narrative prose") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not found in .env")
        sys.exit(1)

    prompt = (
        "You are a master Sanskrit scholar specializing in classical narrative literature.\n"
        f"Translate the following English {context} into elegant Sanskrit Devanagari script.\n"
        "Preserve the dramatic tone and narrative flow. Use Anushtubh meter where possible for verse.\n"
        "CRITICAL: Your response must be ONLY the pure Devanagari Sanskrit text.\n"
        "No English, no transliteration, no explanation, no markdown, no quotes.\n\n"
        f"English text:\n{english.strip()}"
    )

    last_err = None
    for model_id, model_name in MODELS:
        try:
            print(f"[OpenRouter] Trying {model_name}...")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "EdgeSanskrit",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
                timeout=40,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip any markdown fencing the model might add
            for tag in ["```devanagari", "```sanskrit", "```"]:
                raw = raw.replace(tag, "")
            sanskrit = raw.strip()

            if not sanskrit:
                raise ValueError("Empty response from model")

            print(f"[OpenRouter] SUCCESS via {model_name}")
            return sanskrit

        except Exception as e:
            print(f"[OpenRouter] {model_name} failed: {e} — trying next...")
            last_err = e
            continue

    raise RuntimeError(f"All translation models failed. Last: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Synthesize Sanskrit audio via IndicF5
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_audio(sanskrit: str, out_path: str, meter: str = "anushtubh"):
    import soundfile as sf
    from generate_sanskrit_v2 import SanskritChantEngine

    print(f"\n[Engine] Loading IndicF5 on CPU (nfe=12)...")
    engine = SanskritChantEngine(voice_model="indicf5", device="cpu", nfe_step=12, speed=0.90)

    print(f"[Engine] Synthesizing ({len(sanskrit)} chars)...")
    t0 = time.perf_counter()
    sr, audio = engine.synthesize(text=sanskrit, meter=meter, seed=42)
    elapsed = time.perf_counter() - t0
    dur = len(audio) / sr
    rtf = elapsed / dur if dur > 0 else 0

    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"[Engine] Done! Duration={dur:.2f}s | Elapsed={elapsed:.1f}s | RTF={rtf:.2f}x")
    print(f"[Engine] Saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeSanskrit: English → Sanskrit → Audio")
    parser.add_argument("--text",   type=str, default=None,
                        help="English text to synthesize (uses built-in chapter text if omitted)")
    parser.add_argument("--meter",  type=str, default="anushtubh",
                        choices=["anushtubh", "vasantatilaka", "malini", "sragdhara", "upajati"])
    parser.add_argument("--output", type=str, default="chapter1_sanskrit.wav",
                        help="Output WAV filename")
    parser.add_argument("--translate-only", action="store_true",
                        help="Only translate, skip audio generation")
    args = parser.parse_args()

    english = args.text or ENGLISH_TEXT

    print("=" * 60)
    print("  EdgeSanskrit: English to Sanskrit Audio")
    print("=" * 60)
    print(f"\n[Input] {len(english)} chars of English text")
    print("-" * 60)

    # Step 1: Translate
    print("\n[Step 1] Translating to Sanskrit via OpenRouter...")
    sanskrit = translate_to_sanskrit(english)

    print("\n" + "=" * 60)
    print("  SANSKRIT TRANSLATION:")
    print("=" * 60)
    print(sanskrit)
    print("=" * 60)

    # Cache it
    cache_path = os.path.join(HERE, "translations.json")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    cache[english.strip()[:80] + "..."] = sanskrit
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    print(f"\n[Cache] Translation saved to translations.json")

    if args.translate_only:
        print("\n[Done] Translation complete (--translate-only mode, skipping audio)")
        sys.exit(0)

    # Step 2: Synthesize
    print(f"\n[Step 2] Generating Sanskrit audio...")
    out_path = os.path.join(HERE, args.output)
    synthesize_audio(sanskrit, out_path, meter=args.meter)

    print("\n" + "=" * 60)
    print(f"  OUTPUT: {out_path}")
    print("=" * 60)
