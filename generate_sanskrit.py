# generate_sanskrit.py
import os
import sys
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from loguru import logger

# Import Sanskrit phonemizer
from sanskrit_phonemizer import devanagari_to_ipa

# Import KModel from the installed kokoro package
try:
    from kokoro.model import KModel
except ImportError:
    # Fallback to local import if needed
    sys.path.append(os.path.join(os.path.dirname(__file__), 'kokoro'))
    from kokoro.model import KModel

class SanskritKokoroTTS:
    def __init__(self, repo_id='hexgrad/Kokoro-82M', device='cpu'):
        logger.info(f"Initializing Kokoro KModel from repo '{repo_id}' on device '{device}'...")
        self.repo_id = repo_id
        self.device = device
        
        # Download/load model config and weights
        self.model = KModel(repo_id=repo_id).to(device).eval()
        self.voices = {}
        
    def get_voice_pack(self, voice_name):
        """
        Loads the voice style pack (e.g. 'hm_omega', 'hf_alpha').
        Downloads from Hugging Face if not cached locally.
        """
        if voice_name in self.voices:
            return self.voices[voice_name]
        
        logger.info(f"Loading voice pack '{voice_name}'...")
        if voice_name.endswith('.pt'):
            f_path = voice_name
        else:
            f_path = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice_name}.pt')
            
        pack = torch.load(f_path, map_location=self.device, weights_only=True)
        self.voices[voice_name] = pack
        return pack

    def synthesize_chunk(self, ipa_text, voice_pack, speed=1.0):
        """
        Synthesizes a single chunk of IPA phonemes (max 510 characters).
        """
        if not ipa_text.strip():
            return None
            
        # Maximum length Kokoro supports is 510 phonemes
        if len(ipa_text) > 510:
            logger.warning(f"Chunk too long ({len(ipa_text)} > 510). Truncating to 510 characters.")
            ipa_text = ipa_text[:510]
            
        # Retrieve the style vector corresponding to the length of the phoneme sequence
        # Index is len(ipa_text) - 1. Clamp index within bounds just in case.
        idx = max(0, min(len(ipa_text) - 1, len(voice_pack) - 1))
        style_vector = voice_pack[idx].unsqueeze(0).to(self.device)
        
        # Forward pass on KModel
        with torch.no_grad():
            audio = self.model(ipa_text, style_vector[0], speed)
        return audio

    def synthesize(self, text, voice_name='hm_omega', speed=1.0, split_pattern=r'[।॥\n\.\?!]+'):
        """
        Synthesizes Sanskrit Devanagari text.
        Automatically splits long texts into clauses or sentences to fit Kokoro's context window,
        synthesizes them on CPU, and concatenates the resulting audio.
        """
        voice_pack = self.get_voice_pack(voice_name)
        
        # Clean text and split it based on punctuation/boundaries
        import re
        segments = re.split(split_pattern, text)
        segments = [s.strip() for s in segments if s.strip()]
        
        if not segments:
            logger.warning("No text to synthesize.")
            return None
            
        audio_chunks = []
        
        for idx, segment in enumerate(segments):
            # Translate Devanagari segment to IPA
            ipa_str = devanagari_to_ipa(segment)
            if not ipa_str.strip():
                continue
                
            logger.info(f"Synthesizing segment {idx+1}/{len(segments)}:")
            logger.info(f"  Deva: {segment}")
            logger.info(f"  IPA:  {ipa_str}")
            
            # Synthesize chunk
            audio = self.synthesize_chunk(ipa_str, voice_pack, speed)
            if audio is not None:
                audio_chunks.append(audio)
                
                # Add a brief pause (silence) after each segment/punctuation boundary
                # 24000 Hz * 0.4 seconds = 9600 samples of zeros
                silence = torch.zeros(int(24000 * 0.4))
                audio_chunks.append(silence)
                
        if not audio_chunks:
            return None
            
        # Concatenate all chunks
        final_audio = torch.cat(audio_chunks[:-1]) # Drop the trailing silence
        return final_audio

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sanskrit Text-to-Speech using Kokoro on CPU")
    parser.add_argument("text", type=str, help="Sanskrit Devanagari text to synthesize")
    parser.add_argument("-o", "--output", type=str, default="sanskrit_output.wav", help="Output WAV file path")
    parser.add_argument("-v", "--voice", type=str, default="hm_omega", help="Voice name (e.g. hm_omega, hf_alpha)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed multiplier")
    
    args = parser.parse_args()
    
    tts = SanskritKokoroTTS()
    audio = tts.synthesize(args.text, voice_name=args.voice, speed=args.speed)
    
    if audio is not None:
        sf.write(args.output, audio.numpy(), 24000)
        logger.info(f"Saved synthesized Sanskrit audio to {args.output}")
    else:
        logger.error("Synthesis failed.")

if __name__ == '__main__':
    main()
