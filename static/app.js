document.addEventListener('DOMContentLoaded', () => {
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const errorMessage = document.getElementById('errorMessage');
    const audioPlayer = document.getElementById('audioPlayer');
    const downloadLink = document.getElementById('downloadLink');
    const translatedText = document.getElementById('translatedText');

    // Update placeholder based on selected language
    document.querySelectorAll('input[name="inputLang"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const lang = document.querySelector('input[name="inputLang"]:checked').value;
            document.getElementById('sanskritText').placeholder =
                lang === 'english'
                    ? 'Example: The warrior stands on the battlefield of Kurukshetra...'
                    : 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत संजय ॥';
        });
    });

    synthesizeBtn.addEventListener('click', async () => {
        const text = document.getElementById('sanskritText').value.trim();
        const meter = document.getElementById('meterSelect').value;
        const voiceModel = document.getElementById('voiceSelect').value;
        const inputLang = document.querySelector('input[name="inputLang"]:checked').value;

        if (!text) {
            showError('Please enter some text first.');
            return;
        }

        // Reset UI
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        synthesizeBtn.classList.add('loading');

        // Update button label context
        document.querySelector('.btn-text').textContent =
            inputLang === 'english' ? 'Translating & Generating...' : 'Generating...';

        let data = null;

        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('meter', meter);
            formData.append('voice_model', voiceModel);
            formData.append('input_lang', inputLang);

            const response = await fetch('/api/synthesize', {
                method: 'POST',
                body: formData,
            });

            // Safely parse JSON — never let it crash
            const rawText = await response.text();
            try {
                data = JSON.parse(rawText);
            } catch {
                throw new Error(`Server returned invalid response (status ${response.status}). Check the terminal for logs.`);
            }

            if (!response.ok) {
                throw new Error(data?.detail || `Server error (${response.status})`);
            }

            // ── Success ──────────────────────────────────────────────────
            if (data.sanskrit_text) {
                translatedText.innerHTML =
                    `<strong>Sanskrit:</strong> ` + data.sanskrit_text.replace(/\n/g, '<br>');
            } else {
                translatedText.innerHTML = '';
            }

            audioPlayer.src = data.audio_url + '?t=' + Date.now(); // cache-bust
            downloadLink.href = data.audio_url;
            resultContainer.classList.remove('hidden');

            // Try auto-play
            audioPlayer.play().catch(() => {});

        } catch (err) {
            showError(err.message);
        } finally {
            synthesizeBtn.classList.remove('loading');
            document.querySelector('.btn-text').textContent = 'Generate Chanting';
        }
    });

    function showError(msg) {
        errorMessage.textContent = '⚠️ ' + msg;
        errorContainer.classList.remove('hidden');
    }
});
