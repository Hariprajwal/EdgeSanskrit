"""
generate_sanskrit_v2.py — Authentic Sanskrit Chanting TTS via IndicF5
=====================================================================
Architecture:
    Sanskrit Devanagari
        → Vagdhenu prep_text.py (Devanagari → Kannada script routing)
        → IndicF5 DiT (flow-matching, 337M params, zero-shot voice cloning)
        ← Reference Audio (Vagdhenu reference_bank/*.wav — real Sanskrit chanting)
        → Vocos Vocoder (mel → 24kHz audio)
        → Gate + Stitch (Vagdhenu-style head/tail trimming, inter-pada silence)
        → Output WAV

Why Kannada routing?
    Devanagari triggers Hindi schwa-deletion in IndicF5's frontend.
    Routing through Kannada orthography suppresses that, giving true
    Sanskrit phonetics — every inherent short 'a' is preserved.

Why reference audio?
    IndicF5 is a zero-shot voice cloner. The "authentic Indian chanting touch"
    is entirely sourced from the 5-12 second reference WAV clips in
    vagdhenu/src/reference_bank/, which are recordings of Prof. Prathosh
    (IISc) chanting Sanskrit in traditional pārāyaṇa style. The model
    clones voice timbre, swara (pitch contour), pace, and chanting rhythm.

CPU Optimization:
    Default nfe_step=12 instead of 64 gives ~5x speedup with minimal
    quality loss. At 12 steps, a 10-second verse takes ~40-60s on CPU.

Usage:
    python generate_sanskrit_v2.py "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
    python generate_sanskrit_v2.py --text "..." --meter anushtubh --output out.wav
    python generate_sanskrit_v2.py --text "..." --nfe 16 --voice-model vagdhenu

Credits:
    Vagdhenu (github.com/prathoshap/vagdhenu) — Prof. Prathosh, IISc
    IndicF5   (github.com/ai4bharat/IndicF5) — AI4Bharat
    Kokoro    (github.com/hexgrad/kokoro)     — hexgrad
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError for Devanagari/macrons)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Repo-relative path setup (must come first) ───────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
VAGDHENU_SRC = os.path.join(HERE, "vagdhenu", "src")
BANK_DIR     = os.path.join(VAGDHENU_SRC, "reference_bank")
BANK_JSON    = os.path.join(BANK_DIR, "bank.json")
VOCAB_FILE   = os.path.join(BANK_DIR, "vocab.txt")

# Add vagdhenu/src to path so we can import its modules directly
if VAGDHENU_SRC not in sys.path:
    sys.path.insert(0, VAGDHENU_SRC)

# ── Load .env file (HF token, etc.) ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, '.env'))
except ImportError:
    pass  # python-dotenv optional — set HF_TOKEN as system env var instead

# Inject HF_TOKEN so huggingface_hub picks it up automatically
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEYS")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

# ── Constants ─────────────────────────────────────────────────
SR = 24_000
FALLBACK_METER = "vasantatilaka"

# IndicF5 DiT architecture (must match the pre-trained weights exactly)
DIT_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


# ── Phonology helpers (ported verbatim from Vagdhenu render_core.py) ──────────

def _n_aksharas(s: str) -> int:
    """Count the number of syllables (akṣaras) in a Devanagari or Kannada string."""
    n = 0
    L = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons  = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i + 1] if i + 1 < L else ""
            if nxt not in ("्", "್"):
                n += 1
    return n


def _ends_halant(txt: str) -> bool:
    """True if the text ends with a virama (halant consonant cluster closer)."""
    t = txt.rstrip(" ।॥|.,;:!?​")
    return bool(t) and t[-1] in "्್"


def _gate(au: np.ndarray,
          voice: float = 0.08, sil: float = 0.012,
          fin: float   = 0.015, fout: float = 0.040,
          lead: float  = 0.03,  keep: float = 0.06,
          fric: bool   = False, halant: bool = False) -> np.ndarray:
    """
    Trim head and tail silence from synthesized audio (Vagdhenu gate function).
    Applies a short fade-in and cosine fade-out for clean transitions.
    """
    win = int(0.02 * SR)
    r   = [float(np.sqrt((au[i:i + win] ** 2).mean()))
           for i in range(0, len(au) - win, win)]
    n = len(r)
    if n == 0:
        return au

    vs = next((i for i in range(n - 1) if r[i] > voice and r[i + 1] > sil),
              int(np.argmax(r)))
    s  = vs
    while s > 0 and r[s - 1] > sil:
        s -= 1

    ve_thr = 0.012 if halant else 0.035
    ve     = max((i for i in range(n) if r[i] > ve_thr), default=vs)
    keep_s = 0.12  if halant else keep

    start = max(0, s * win - int(lead * SR))
    end   = min(len(au), ve * win + int(keep_s * SR))
    out   = au[start:end].copy()

    fi = 0 if fric else int(fin * SR)
    fo = int((0.018 if halant else fout) * SR)
    if fi and len(out) > fi:
        out[:fi] *= np.linspace(0, 1, fi)
    if fo and len(out) > fo:
        out[-fo:] *= (np.cos(np.linspace(0, np.pi, fo)) * 0.5 + 0.5)
    return out


def _split_padas(text: str) -> list[str]:
    """
    Split a Sanskrit śloka into hemistich/pāda segments.
    Splits on newlines first, then on daṇḍas (।). Empty segments are dropped.
    """
    pieces = []
    for line in text.replace("॥", "।").replace("|", "।").splitlines():
        for seg in line.split("।"):
            seg = seg.strip()
            if seg:
                pieces.append(seg)
    return pieces or ([text.strip()] if text.strip() else [])


# ── Model class ───────────────────────────────────────────────────────────────

class SanskritChantEngine:
    """
    Full pipeline for authentic Sanskrit chanting TTS.

    On construction, it:
      1. Loads the IndicF5 DiT model (downloaded from HuggingFace or a local path)
      2. Loads the Vocos vocoder
      3. Parses the Vagdhenu reference bank (bank.json)

    synthesize() then:
      1. Splits text into pādas
      2. Converts each pāda from Devanagari → Kannada via prep_text.py
      3. Looks up the per-meter reference WAV + text from bank.json
      4. Runs IndicF5 inference at nfe_step (default 12) per pāda
      5. Gates each segment and stitches with silence gaps
      6. Returns (sample_rate, audio_float32)
    """

    def __init__(
        self,
        voice_model: str  = "indicf5",   # "indicf5" or "vagdhenu"
        device:      str  = "cpu",
        nfe_step:    int  = 12,
        cfg_strength:float= 3.0,
        speed:       float= 0.90,
        gap:         float= 0.55,
        gap_halant:  float= 0.20,
    ):
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
        except ImportError:
            raise ImportError(
                "\n[EdgeSanskrit] IndicF5 is not installed.\n"
                "Run: pip install git+https://github.com/ai4bharat/IndicF5.git"
                "@13f7c4d627cc10111aea8fe9c0039462cacacdc7\n"
            )

        self.device       = device
        self.nfe_step     = nfe_step
        self.cfg_strength = cfg_strength
        self.speed        = speed
        self.gap          = gap
        self.gap_halant   = gap_halant

        # ── 1. Load reference bank ──────────────────────────────────────
        if not os.path.exists(BANK_JSON):
            raise FileNotFoundError(
                f"Vagdhenu reference bank not found at {BANK_JSON}\n"
                "Make sure you have cloned vagdhenu/ beside this script:\n"
                "  git clone https://github.com/prathoshap/vagdhenu vagdhenu"
            )
        with open(BANK_JSON, encoding="utf-8") as f:
            self._bank = json.load(f)

        self._lut: dict = {}
        for k, v in self._bank.items():
            if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                continue
            self._lut[k.lower()] = v
            self._lut[v["wav"].replace(".wav", "").lower()] = v

        print(f"[EdgeSanskrit] Reference bank loaded: {len(self._lut)//2} meters")

        # ── 2. Load DiT (IndicF5 base or Vagdhenu fine-tune) ───────────
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(
                f"IndicF5 vocab.txt not found at {VOCAB_FILE}\n"
                "It ships with the Vagdhenu clone in reference_bank/vocab.txt"
            )

        print(f"[EdgeSanskrit] Loading IndicF5 DiT on device='{device}'...")
        self.cfm = load_model(DiT, DIT_CFG, mel_spec_type="vocos",
                              vocab_file=VOCAB_FILE, device=device)

        if voice_model == "vagdhenu":
            self._load_vagdhenu_weights()
        else:
            self._load_indicf5_weights()

        self.cfm.eval()
        print("[EdgeSanskrit] DiT loaded.")

        # ── 3. Load Vocos vocoder ───────────────────────────────────────
        print("[EdgeSanskrit] Loading Vocos vocoder...")
        self.vocoder  = load_vocoder("vocos")
        self._refcache: dict = {}
        print("[EdgeSanskrit] Vocos loaded. Engine ready.\n")

    # ── Weight loaders ────────────────────────────────────────────────

    def _load_indicf5_weights(self):
        """Download and load base IndicF5 weights from ai4bharat/IndicF5 on HuggingFace."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print("[EdgeSanskrit] Downloading ai4bharat/IndicF5 weights from HuggingFace...")
        ckpt = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="model.safetensors")
        print(f"[EdgeSanskrit] Checkpoint: {ckpt}")

        sd = load_file(ckpt, device="cpu")
        # Strip the EMA module prefix that IndicF5 uses during training
        sd = {
            k.replace("ema_model._orig_mod.", "").replace("ema_model.", ""): v
            for k, v in sd.items()
        }
        missing, unexpected = self.cfm.load_state_dict(sd, strict=False)
        if missing:
            print(f"[EdgeSanskrit] Missing keys (expected if small): {len(missing)}")

    def _load_vagdhenu_weights(self):
        """Download and load Vagdhenu voice-steered checkpoint from prathoshap/vagdhenu."""
        from huggingface_hub import hf_hub_download

        print("[EdgeSanskrit] Downloading prathoshap/vagdhenu voice-steered weights...")
        ckpt = hf_hub_download(repo_id="prathoshap/vagdhenu",
                               filename="voice_steer_ema.pt")
        print(f"[EdgeSanskrit] Checkpoint: {ckpt}")

        ck  = torch.load(ckpt, map_location="cpu", weights_only=True)
        ema = {k.replace("ema_model.", ""): v
               for k, v in ck["ema_model_state_dict"].items()
               if k not in ("initted", "step")}
        self.cfm.load_state_dict(ema, strict=False)

    # ── Reference audio cache ─────────────────────────────────────────

    def _get_ref(self, meter: str):
        """
        Look up reference audio and metadata for a given meter.
        Results are cached after the first load.
        Returns (ref_audio_path, ref_text, sec_per_syll, ref_duration_s)
        """
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        import torchaudio as ta

        key = meter.lower().replace(".wav", "").replace("ā", "a").replace("ṭ", "t").replace("ū", "u")
        # Try exact key first, then normalized version
        matched_key = None
        for k in self._lut:
            if k == key or k.replace("ā", "a") == key or meter.lower() in k:
                matched_key = k
                break

        if matched_key is None:
            print(f"[EdgeSanskrit] Meter '{meter}' not found in bank → fallback '{FALLBACK_METER}'")
            matched_key = FALLBACK_METER

        if matched_key in self._refcache:
            return self._refcache[matched_key]

        e        = self._lut[matched_key]
        wav_path = os.path.join(BANK_DIR, e["wav"])
        ref_text = e["ref_text"]
        sps      = float(e.get("sec_per_syll", 0.26))

        ref_audio, ref_t = preprocess_ref_audio_text(wav_path, ref_text, clip_short=True)
        ra, wav_sr       = ta.load(ref_audio)
        ref_len          = ra.shape[-1] / wav_sr

        val = (ref_audio, ref_t, sps, ref_len)
        self._refcache[matched_key] = val
        print(f"[EdgeSanskrit] Reference: {e['wav']} ({ref_len:.1f}s) — sec/syl={sps}")
        return val

    # ── Main synthesis ────────────────────────────────────────────────

    def synthesize(
        self,
        text:   str,
        meter:  str   = "anushtubh",
        seed:   int   = 60,
        speed:  float = None,
        sps:    float = None,
    ) -> tuple[int, np.ndarray]:
        """
        Synthesize Sanskrit Devanagari text with authentic chanting prosody.

        Args:
            text:   Sanskrit text in Devanagari (supports multi-line, daṇḍas)
            meter:  Vṛtta/meter name from bank.json (e.g. 'anushtubh', 'vasantatilaka')
            seed:   Random seed for reproducibility
            speed:  Speed multiplier override (default: self.speed = 0.90)
            sps:    Seconds-per-syllable override

        Returns:
            (sample_rate, audio_float32_array)
        """
        from f5_tts.infer.utils_infer import infer_process

        # Import Vagdhenu's text converter
        try:
            import prep_text as PT
        except ImportError:
            raise ImportError(
                "Cannot import prep_text. Ensure vagdhenu/src/ is cloned "
                "and on sys.path."
            )

        padas = _split_padas(text)
        if not padas:
            raise ValueError("Input text is empty after splitting.")

        ref_audio, ref_t, ref_sps, ref_len = self._get_ref(meter)
        if sps is not None:
            ref_sps = float(sps)
        spd = float(speed) if speed is not None else self.speed

        print(f"[EdgeSanskrit] Synthesizing {len(padas)} pāda(s)  nfe={self.nfe_step}  device={self.device}")

        # Convert each pada: Devanagari → Kannada routed text
        pieces = [PT.model_text(p) for p in padas]
        nsylls = [_n_aksharas(p) for p in pieces]

        print(f"[EdgeSanskrit] Pādas:")
        for i, (orig, kan) in enumerate(zip(padas, pieces)):
            print(f"  [{i+1}] {orig}")
            print(f"       → {kan}  ({nsylls[i]} akṣaras)")

        bseg = []
        for i, piece in enumerate(pieces):
            print(f"\n[EdgeSanskrit] Generating pāda {i+1}/{len(pieces)}...")
            t0  = time.perf_counter()
            au  = None

            for attempt in range(4):
                torch.manual_seed(seed + attempt)
                fix_dur = (ref_len + nsylls[i] * ref_sps) if ref_sps > 0 else None

                w, sr, _ = infer_process(
                    ref_audio, ref_t, piece, self.cfm, self.vocoder,
                    mel_spec_type  = "vocos",
                    speed          = spd,
                    nfe_step       = self.nfe_step,
                    cfg_strength   = self.cfg_strength,
                    device         = self.device,
                    fix_duration   = fix_dur,
                )
                w = np.array(w, dtype=np.float32)
                if np.abs(w).max() > 1.5:
                    w = w / 32768.0

                rms = float(np.sqrt((w ** 2).mean()))
                elapsed = time.perf_counter() - t0
                dur     = len(w) / SR
                rtf     = elapsed / dur if dur > 0 else float("inf")
                print(f"  attempt {attempt+1}: {len(w)/SR:.2f}s audio, "
                      f"RMS={rms:.4f}, RTF={rtf:.2f}x, elapsed={elapsed:.1f}s")

                if rms > 0.04:
                    au = w
                    break

            if au is None:
                print(f"  [WARN] All attempts had low RMS — using last result anyway.")
                au = w

            # Vagdhenu-style gate: trim head/tail silence
            halant = _ends_halant(pieces[i])
            au     = _gate(au, halant=halant)
            bseg.append(au)

        # Stitch pādas with silence gaps
        gap_samples        = int(self.gap * SR)
        halant_gap_samples = int(self.gap_halant * SR)
        result = []
        for i, seg in enumerate(bseg):
            result.append(seg)
            if i < len(bseg) - 1:
                extra = halant_gap_samples if _ends_halant(pieces[i]) else 0
                result.append(np.zeros(gap_samples + extra, dtype=np.float32))

        final = np.concatenate(result)

        # Normalize to 97% peak
        mx = np.abs(final).max()
        if mx > 0:
            final = final / mx * 0.97

        print(f"\n[EdgeSanskrit] Final audio: {len(final)/SR:.2f}s")
        return SR, final


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EdgeSanskrit v2 — Authentic Sanskrit Chanting TTS via IndicF5",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "text", nargs="?",
        default="धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत संजय ॥",
        help="Sanskrit Devanagari input text (default: BG 1.1)",
    )
    parser.add_argument("-o", "--output",  default="sanskrit_chant_v2.wav",
                        help="Output WAV file path")
    parser.add_argument("-m", "--meter",   default="anushtubh",
                        help="Sanskrit meter/vṛtta (e.g. anushtubh, vasantatilaka)")
    parser.add_argument("--nfe",           type=int,   default=12,
                        help="NFE diffusion steps (lower=faster, default=12)")
    parser.add_argument("--speed",         type=float, default=0.90,
                        help="Speech rate multiplier (default=0.90 for chant pacing)")
    parser.add_argument("--voice-model",   default="indicf5",
                        choices=["indicf5", "vagdhenu"],
                        help="'indicf5' = base zero-shot model (default)\n"
                             "'vagdhenu' = Prof. Prathosh's voice-steered model (~5GB)")
    parser.add_argument("--seed",          type=int,   default=60,
                        help="Random seed (default=60)")
    args = parser.parse_args()

    import soundfile as sf

    engine = SanskritChantEngine(
        voice_model  = args.voice_model,
        device       = "cpu",
        nfe_step     = args.nfe,
        speed        = args.speed,
    )

    t_start = time.perf_counter()
    sr, audio = engine.synthesize(
        text  = args.text,
        meter = args.meter,
        seed  = args.seed,
    )
    total_elapsed = time.perf_counter() - t_start
    audio_dur     = len(audio) / sr
    rtf           = total_elapsed / audio_dur if audio_dur > 0 else float("inf")

    sf.write(args.output, audio, sr, subtype="PCM_16")

    print(f"\n{'='*60}")
    print(f"  Output : {args.output}")
    print(f"  Audio  : {audio_dur:.2f}s at {sr}Hz")
    print(f"  Elapsed: {total_elapsed:.1f}s")
    print(f"  RTF    : {rtf:.2f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
