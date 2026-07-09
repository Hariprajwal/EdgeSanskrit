# sanskrit_phonemizer.py
import re

# VOWELS
VOWELS = {
    'अ': 'a',
    'आ': 'aː',
    'इ': 'i',
    'ई': 'iː',
    'उ': 'u',
    'ऊ': 'uː',
    'ऋ': 'ɾɪ',
    'ॠ': 'ɾiː',
    'ऌ': 'lɪ',
    'ए': 'e',
    'ऐ': 'aɪ',
    'ओ': 'o',
    'औ': 'aʊ'
}

# VOWEL SIGNS (DEPENDENT VOWELS)
VOWEL_SIGNS = {
    'ा': 'aː',
    'ि': 'i',
    'ी': 'iː',
    'ु': 'u',
    'ू': 'uː',
    'ृ': 'ɾɪ',
    'ॄ': 'ɾiː',
    'ॢ': 'lɪ',
    'े': 'e',
    'ै': 'aɪ',
    'ो': 'o',
    'ौ': 'aʊ'
}

# CONSONANTS
CONSONANTS = {
    'क': 'k', 'ख': 'kʰ', 'ग': 'ɡ', 'घ': 'ɡʰ', 'ङ': 'ŋ',
    'च': 'tʃ', 'छ': 'tʃʰ', 'ज': 'dʒ', 'झ': 'dʒʰ', 'ञ': 'ɲ',
    'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
    'त': 't', 'थ': 'tʰ', 'द': 'd', 'ध': 'dʰ', 'न': 'n',
    'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
    'य': 'j', 'र': 'ɾ', 'ल': 'l', 'व': 'v',
    'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'h',
    'ळ': 'ɭ'
}

VIRAMA = '्'
ANUSVARA = 'ं'
VISARGA = 'ः'
AVAGRAHA = 'ऽ'

# Consonant vargas for homorganic nasal replacement
VARGAS = {
    'velar': (set('कखगघङ'), 'ŋ'),
    'palatal': (set('चछजझञ'), 'ɲ'),
    'retroflex': (set('टठडढण'), 'ɳ'),
    'dental': (set('तथदधन'), 'n'),
    'labial': (set('पफबभम'), 'm')
}

def get_homorganic_nasal(text, index):
    # Search forward for the next consonant to find its varga
    for i in range(index + 1, len(text)):
        char = text[i]
        if char in CONSONANTS:
            # Check varga
            for varga_name, (chars, nasal) in VARGAS.items():
                if char in chars:
                    return nasal
            break
        elif char in (' ', '\n', '\t', '।', '॥', ',', '.'):
            # If we hit a boundary before a consonant, default to 'm'
            break
    return 'm'

def get_visarga_echo(prev_vowel_ipa):
    if not prev_vowel_ipa:
        return 'ha'
    
    # Strip length markers or features to get core vowel
    core_vowel = prev_vowel_ipa.replace('ː', '').strip()
    if 'a' in core_vowel:
        return 'ha'
    elif 'i' in core_vowel:
        return 'hi'
    elif 'u' in core_vowel:
        return 'hu'
    elif 'e' in core_vowel:
        return 'he'
    elif 'o' in core_vowel:
        return 'ho'
    else:
        return 'ha'

def devanagari_to_ipa(text):
    """
    Converts Devanagari Sanskrit text directly to Kokoro-compatible IPA phonemes,
    preserving short vowels (no schwa deletion), and applying visarga and anusvara sandhi.
    """
    # Clean up whitespace and special characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    ipa_out = []
    i = 0
    n = len(text)
    
    # Track the last generated vowel to resolve visarga echo U+0903
    last_vowel_ipa = 'a' # Default
    
    while i < n:
        char = text[i]
        
        # 1. Independent Vowels
        if char in VOWELS:
            v_ipa = VOWELS[char]
            ipa_out.append(v_ipa)
            last_vowel_ipa = v_ipa
            i += 1
            
        # 2. Consonants
        elif char in CONSONANTS:
            c_ipa = CONSONANTS[char]
            # Look ahead to see if it's followed by a virama or dependent vowel sign
            has_vowel_sign = False
            has_virama = False
            
            # Check next character
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == VIRAMA:
                    has_virama = True
                    ipa_out.append(c_ipa) # Append consonant directly without default vowel
                    i += 2 # Skip both consonant and virama
                elif nxt in VOWEL_SIGNS:
                    has_vowel_sign = True
                    v_ipa = VOWEL_SIGNS[nxt]
                    ipa_out.append(c_ipa + v_ipa)
                    last_vowel_ipa = v_ipa
                    i += 2 # Skip both consonant and vowel sign
                    
            if not has_virama and not has_vowel_sign:
                # No sign or virama means default inherent short vowel 'a'
                ipa_out.append(c_ipa + 'a')
                last_vowel_ipa = 'a'
                i += 1
                
        # 3. Anusvara
        elif char == ANUSVARA:
            # Map to homorganic nasal
            nasal = get_homorganic_nasal(text, i)
            ipa_out.append(nasal)
            i += 1
            
        # 4. Visarga
        elif char == VISARGA:
            # Visarga echo based on last vowel
            echo = get_visarga_echo(last_vowel_ipa)
            ipa_out.append(echo)
            i += 1
            
        # 5. Avagraha U+093D
        elif char == AVAGRAHA:
            ipa_out.append('ː') # Maps to length mark
            i += 1
            
        # 6. Spaces, punctuation, and other non-indic characters
        else:
            if char in (' ', '\n', '\t'):
                ipa_out.append(' ')
            elif char in ('।', '॥'):
                ipa_out.append('.') # Map daṇḍas to periods for natural pausing
            elif char in (',', ';', '.', '!', '?'):
                ipa_out.append(char)
            # Ignore other characters (digits, parentheses, etc.)
            i += 1
            
    # Join into a single clean string, clean up multiple spaces
    ipa_str = "".join(ipa_out)
    # Replace multiple spaces with a single space
    ipa_str = re.sub(r' +', ' ', ipa_str)
    return ipa_str

# Test run
if __name__ == '__main__':
    test_phrases = [
        "नमः शिवाय",
        "धर्मक्षेत्रे कुरुक्षेत्रे",
        "रामः गच्छति",
        "सच्चिदानन्द",
        "श्रीमद्भगवद्गीता"
    ]
    for phrase in test_phrases:
        print(f"Deva: {phrase}")
        print(f"IPA:  {devanagari_to_ipa(phrase)}")
        print("-" * 40)
