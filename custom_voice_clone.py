import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch
from pathlib import Path

# Directories (relative paths)
SPEAKERS_DIR = Path("speakers")
OUTPUT_DIR = Path("out")

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load multilingual Chatterbox model
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

text = "Heute ist ein wunderschöner Tag. Dies ist meine geklonte Stimme."

# speaker audio prompt
audio_prompt = SPEAKERS_DIR / "speaker0_short.wav"

wav = model.generate(
    text,
    language_id="de",
    audio_prompt_path=audio_prompt,
    exaggeration=0.4,
    cfg_weight=0.6,
)

# Output path
output_file = OUTPUT_DIR / "speaker0_clone_de.wav"

ta.save(str(output_file), wav, model.sr)
print(f"Saved: {output_file}")