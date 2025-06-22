from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
import numpy as np
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

# Text to convert to speech
text = "Hello, My nam is Minh Anh, I am 19, I love Dat very much. Um I am very naughty"

inputs = processor(text, voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# Save audio to file
sample_rate = 24000  # Bark uses 24kHz sample rate
wavfile.write("output.wav", sample_rate, audio_array)

print(f"Audio saved to output.wav")
print(f"Text: {text}")
print(f"Voice preset: {voice_preset}")