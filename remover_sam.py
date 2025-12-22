import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SAMAudio.from_pretrained("facebook/sam-audio-small").to(device).eval()
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-small")

print("Model loaded successfully!")

audio_file = "mixed.mp3"
description = "music"

# Process and separate
inputs = processor(audios=[audio_file], descriptions=[description]).to(device)

with torch.inference_mode():
    result = model.separate(inputs)

# Save results
torchaudio.save("target.wav", result.target[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)
torchaudio.save("residual.wav", result.residual[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)

print("Separation complete!")
print("- target.wav: The isolated sound you described")
print("- residual.wav: Everything else")