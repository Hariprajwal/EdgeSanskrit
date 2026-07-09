"""
test_run_v2.py — Verification harness for EdgeSanskrit v2 (IndicF5 engine)
===========================================================================
Tests authentic Sanskrit chanting synthesis via IndicF5 + Vagdhenu reference audio.
Benchmarks CPU inference and compares with Phase 1 (Kokoro) output.
"""

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Test verses ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name"  : "Bhagavad Gita 1.1 (Anushtubh)",
        "text"  : (
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\n"
            "मामकाः पाण्डवाश्चैव किमकुर्वत संजय ॥"
        ),
        "meter" : "anushtubh",
        "output": "v2_gita_1_1.wav",
    },
    {
        "name"  : "Vishnu Sahasranama Invocation (Anushtubh)",
        "text"  : (
            "यस्य स्मरणमात्रेण जन्मसंसारबन्धनात् ।\n"
            "विमुच्यते नमस्तस्मै विष्णवे प्रभविष्णवे ॥"
        ),
        "meter" : "anushtubh",
        "output": "v2_vishnu.wav",
    },
]


def check_prerequisites():
    """Verify all required files and packages are present."""
    print("=" * 60)
    print("  EdgeSanskrit v2 — Prerequisite Check")
    print("=" * 60)

    issues = []

    # Check IndicF5
    try:
        import f5_tts
        print(f"  [OK] f5_tts (IndicF5): {f5_tts.__version__ if hasattr(f5_tts, '__version__') else 'installed'}")
    except ImportError:
        issues.append("IndicF5 not installed. Run: pip install git+https://github.com/ai4bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7")

    # Check vocos
    try:
        import vocos
        print(f"  [OK] vocos: {vocos.__version__}")
    except ImportError:
        issues.append("vocos not installed. Run: pip install vocos>=0.1.0")

    # Check indic-transliteration
    try:
        import indic_transliteration
        print(f"  [OK] indic_transliteration: installed")
    except ImportError:
        issues.append("indic-transliteration not installed. Run: pip install indic-transliteration")

    # Check vagdhenu reference bank
    bank_json = os.path.join(HERE, "vagdhenu", "src", "reference_bank", "bank.json")
    if os.path.exists(bank_json):
        print(f"  [OK] Vagdhenu reference bank: {bank_json}")
    else:
        issues.append(f"Vagdhenu reference bank missing. Clone: git clone https://github.com/prathoshap/vagdhenu vagdhenu")

    # Check vocab.txt
    vocab = os.path.join(HERE, "vagdhenu", "src", "reference_bank", "vocab.txt")
    if os.path.exists(vocab):
        print(f"  [OK] IndicF5 vocab.txt: found")
    else:
        issues.append(f"vocab.txt missing from reference_bank/")

    # Check reference audio files
    import json
    if os.path.exists(bank_json):
        with open(bank_json, encoding="utf-8") as f:
            bank = json.load(f)
        wav_dir = os.path.join(HERE, "vagdhenu", "src", "reference_bank")
        missing_wavs = []
        for k, v in bank.items():
            if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                continue
            wav_path = os.path.join(wav_dir, v["wav"])
            if not os.path.exists(wav_path):
                missing_wavs.append(v["wav"])
        if missing_wavs:
            issues.append(f"Missing reference WAVs: {missing_wavs}")
        else:
            print(f"  [OK] Reference WAVs: all {sum(1 for k,v in bank.items() if isinstance(v, dict) and 'wav' in v)} present")

    print()
    if issues:
        print("[FAIL] The following issues must be resolved before running:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    print("  All prerequisites satisfied.\n")
    return True


def run_tests(nfe: int = 12, voice_model: str = "indicf5"):
    """Run synthesis tests and print RTF benchmarks."""
    sys.path.insert(0, HERE)
    from generate_sanskrit_v2 import SanskritChantEngine
    import soundfile as sf

    print("=" * 60)
    print(f"  EdgeSanskrit v2 — Synthesis Tests")
    print(f"  Engine: IndicF5  |  NFE={nfe}  |  Device=CPU")
    print("=" * 60)

    # Load engine once
    engine = SanskritChantEngine(
        voice_model=voice_model,
        device="cpu",
        nfe_step=nfe,
        speed=0.90,
    )

    results = []
    for tc in TEST_CASES:
        print(f"\n{'─'*60}")
        print(f"  Test: {tc['name']}")
        print(f"  Meter: {tc['meter']}")
        print(f"{'─'*60}")

        t0 = time.perf_counter()
        sr, audio = engine.synthesize(
            text  = tc["text"],
            meter = tc["meter"],
            seed  = 60,
        )
        elapsed   = time.perf_counter() - t0
        audio_dur = len(audio) / sr
        rtf       = elapsed / audio_dur if audio_dur > 0 else float("inf")

        out_path = os.path.join(HERE, tc["output"])
        sf.write(out_path, audio, sr, subtype="PCM_16")

        results.append({
            "name"     : tc["name"],
            "audio_dur": audio_dur,
            "elapsed"  : elapsed,
            "rtf"      : rtf,
            "output"   : out_path,
            "rms"      : float(np.sqrt((audio ** 2).mean())),
        })

        print(f"\n  → Saved: {out_path}")
        print(f"    Duration: {audio_dur:.2f}s  |  Elapsed: {elapsed:.1f}s  |  RTF: {rtf:.2f}x")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Test':<35} {'Dur':>6} {'Elapsed':>8} {'RTF':>6}")
    print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*6}")
    for r in results:
        status = "🔴 slow" if r["rtf"] > 10 else ("🟡" if r["rtf"] > 3 else "🟢")
        print(f"  {r['name'][:35]:<35} {r['audio_dur']:>5.1f}s {r['elapsed']:>7.1f}s {r['rtf']:>5.2f}x {status}")

    print(f"\n  Generated audio files:")
    for r in results:
        print(f"    {r['output']}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="EdgeSanskrit v2 — Test harness"
    )
    parser.add_argument("--nfe",        type=int, default=12,
                        help="NFE diffusion steps (default=12)")
    parser.add_argument("--voice-model", default="indicf5",
                        choices=["indicf5", "vagdhenu"])
    parser.add_argument("--check-only", action="store_true",
                        help="Only run prerequisite checks, skip synthesis")
    args = parser.parse_args()

    ok = check_prerequisites()
    if not args.check_only and ok:
        run_tests(nfe=args.nfe, voice_model=args.voice_model)
    elif not ok:
        print("[INFO] Run prerequisite checks above, then retry.")
        sys.exit(1)
