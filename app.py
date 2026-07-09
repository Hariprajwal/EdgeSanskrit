import os
import sys
import uuid
import json
import requests
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Must be set before importing generate_sanskrit_v2
os.environ["VAGDHENU_SRC"] = os.path.abspath("vagdhenu/src")

try:
    from generate_sanskrit_v2 import SanskritChantEngine
except ImportError as e:
    print(f"Error importing Sanskrit engine: {e}")
    sys.exit(1)

app = FastAPI(title="EdgeSanskrit Web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global engine (loaded once at startup) ────────────────────────────────
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    print("[Web] Initializing SanskritChantEngine (IndicF5) on CPU...")
    engine = SanskritChantEngine(voice_model="indicf5", device="cpu", nfe_step=12, speed=0.90)
    print("[Web] Engine loaded successfully.")

# ── Static file serving ───────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
os.makedirs("generations", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generations", StaticFiles(directory="generations"), name="generations")

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ── OpenRouter Translation ─────────────────────────────────────────────────
def translate_to_sanskrit(english_text: str, meter: str) -> str:
    """
    Translates English → Sanskrit Devanagari using OpenRouter.
    Tries models in order — if one fails, automatically falls back to the next.
    Pipeline: English → LLM (with fallback) → Sanskrit → store in JSON → return
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in .env")

    prompt = (
        f"You are an expert Sanskrit scholar and poet.\n"
        f"Translate the following English text into authentic Sanskrit Devanagari script.\n"
        f"Target meter: {meter} (e.g., Anushtubh = 8 syllables per pada).\n"
        f"CRITICAL: Reply with ONLY the pure Devanagari Sanskrit text. "
        f"No English, no transliteration, no explanation, no markdown, no quotes.\n\n"
        f"English: {english_text}"
    )

    # ── Fallback chain: try each model in order ───────────────────────────
    models = [
        ("google/gemini-flash-1.5",       "Gemini Flash 1.5"),
        ("anthropic/claude-3-haiku",      "Claude 3 Haiku"),
        ("mistralai/mistral-7b-instruct", "Mistral 7B Instruct"),
    ]

    last_error = None
    for model_id, model_name in models:
        try:
            print(f"[OpenRouter] Trying model: {model_name}...")
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
                    "max_tokens": 512,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()

            # Strip any accidental markdown fencing
            for tag in ["```devanagari", "```sanskrit", "```"]:
                raw = raw.replace(tag, "")
            sanskrit = raw.strip()

            if not sanskrit:
                raise ValueError("Model returned empty translation.")

            print(f"[OpenRouter] ✓ Translated via {model_name}: {sanskrit[:80]}")

            # ── Cache result in translations.json ─────────────────────────
            store_path = os.path.join(os.path.dirname(__file__), "translations.json")
            try:
                with open(store_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                cache = {}
            cache[english_text.strip()] = sanskrit
            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)

            return sanskrit

        except Exception as e:
            print(f"[OpenRouter] {model_name} failed: {e} — trying next fallback...")
            last_error = str(e)
            continue

    # All models failed
    raise HTTPException(status_code=502, detail=f"All translation models failed. Last error: {last_error}")


# ── Main synthesis endpoint ───────────────────────────────────────────────
@app.post("/api/synthesize")
def synthesize(
    text: str = Form(...),
    meter: str = Form("anushtubh"),
    voice_model: str = Form("indicf5"),
    input_lang: str = Form("english"),
):
    """
    Full pipeline:
      1. Receive English (or Devanagari) text
      2. If English → translate via OpenRouter → store in translations.json
      3. Pass Sanskrit to IndicF5 engine
      4. Return WAV URL + translated text to browser
    """
    global engine
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not yet initialized. Please wait a moment.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    import soundfile as sf

    print(f"[Web] Synthesis request | lang={input_lang} | meter={meter} | text='{text[:50]}'")

    # Step 1: Translate if English
    final_text = text
    if input_lang == "english":
        final_text = translate_to_sanskrit(text, meter)
        print(f"[Web] Using translated Sanskrit: {final_text[:80]}")

    # Step 2: Run IndicF5 synthesis
    filename = f"chant_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join("generations", filename)

    try:
        sr, audio = engine.synthesize(text=final_text, meter=meter, seed=60)
        sf.write(out_path, audio, sr, subtype="PCM_16")
    except Exception as e:
        print(f"[Web] Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

    print(f"[Web] Done → {out_path}")
    return {
        "audio_url": f"/generations/{filename}",
        "sanskrit_text": final_text,
        "message": "success",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
