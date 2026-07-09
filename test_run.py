# test_run.py
import time
import os
from loguru import logger
import soundfile as sf

from generate_sanskrit import SanskritKokoroTTS

def run_test():
    # Canonical Bhagavad Gita Verse 1.1
    gita_verse = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत संजय ॥"
    
    logger.info("Initializing Sanskrit Kokoro TTS on CPU...")
    tts = SanskritKokoroTTS(device='cpu')
    
    # 1. Test with Hindi Male Voice (hm_omega)
    logger.info("--- Test 1: Hindi Male Voice (hm_omega) ---")
    start_time = time.time()
    audio_omega = tts.synthesize(gita_verse, voice_name='hm_omega', speed=1.0)
    end_time = time.time()
    
    if audio_omega is not None:
        duration = len(audio_omega) / 24000
        elapsed = end_time - start_time
        rtf = elapsed / duration
        logger.info(f"Male voice audio generated successfully!")
        logger.info(f"Audio Duration: {duration:.2f} seconds")
        logger.info(f"Inference Time: {elapsed:.2f} seconds")
        logger.info(f"Real-Time Factor (RTF) on CPU: {rtf:.4f}x ({'Faster' if rtf < 1.0 else 'Slower'} than real-time)")
        
        output_file_omega = "sanskrit_gita_omega.wav"
        sf.write(output_file_omega, audio_omega.numpy(), 24000)
        logger.info(f"Saved audio to: {output_file_omega}")
    else:
        logger.error("Failed to generate audio for male voice.")

    # 2. Test with Hindi Female Voice (hf_alpha)
    logger.info("--- Test 2: Hindi Female Voice (hf_alpha) ---")
    start_time = time.time()
    audio_alpha = tts.synthesize(gita_verse, voice_name='hf_alpha', speed=1.0)
    end_time = time.time()
    
    if audio_alpha is not None:
        duration = len(audio_alpha) / 24000
        elapsed = end_time - start_time
        rtf = elapsed / duration
        logger.info(f"Female voice audio generated successfully!")
        logger.info(f"Audio Duration: {duration:.2f} seconds")
        logger.info(f"Inference Time: {elapsed:.2f} seconds")
        logger.info(f"Real-Time Factor (RTF) on CPU: {rtf:.4f}x ({'Faster' if rtf < 1.0 else 'Slower'} than real-time)")
        
        output_file_alpha = "sanskrit_gita_alpha.wav"
        sf.write(output_file_alpha, audio_alpha.numpy(), 24000)
        logger.info(f"Saved audio to: {output_file_alpha}")
    else:
        logger.error("Failed to generate audio for female voice.")

if __name__ == '__main__':
    run_test()
