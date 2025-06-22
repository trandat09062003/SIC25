# ============================================
# CÁC MODEL TEXT-TO-SPEECH TƯƠNG TỰ BARK
# ============================================

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, BarkModel
import os

# ============================================
# 1. BARK MODEL (Suno)
# ============================================
"""Các model gen audio phổ biến"""
def bark_tts(text, filename="bark_output.wav", voice_preset="v2/en_speaker_6"):
    """Bark model từ Suno - Chất lượng cao, hỗ trợ nhiều ngôn ngữ"""
    print("=== BARK MODEL ===")
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    
    inputs = processor(text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    wavfile.write(filename, 24000, audio_array)
    print(f"✅ Bark audio saved: {filename}")
    return filename

# ============================================
# 2. COQUI TTS
# ============================================

def coqui_tts(text, filename="coqui_output.wav"):
    """Coqui TTS - Open source, nhiều voice"""
    print("=== COQUI TTS ===")
    
    try:
        from TTS.api import TTS
        
        # Sử dụng model mặc định
        tts = TTS()
        
        # Sinh audio
        tts.tts_to_file(text=text, file_path=filename)
        
        print(f"✅ Coqui TTS audio saved: {filename}")
        return filename
        
    except ImportError:
        print("❌ Coqui TTS not installed. Run: pip install TTS")
        return None

# ============================================
# 3. TACOTRON 2 + WAVEGLOW
# ============================================

def tacotron_tts(text, filename="tacotron_output.wav"):
    """Tacotron 2 + WaveGlow - Google's model"""
    print("=== TACOTRON 2 ===")
    
    try:
        from transformers import Tacotron2Processor, Tacotron2ForConditionalGeneration
        from transformers import WaveGlowProcessor, WaveGlowForConditionalGeneration
        
        # Load Tacotron 2
        processor = Tacotron2Processor.from_pretrained("microsoft/speechtoc_text_to_speech")
        model = Tacotron2ForConditionalGeneration.from_pretrained("microsoft/speechtoc_text_to_speech")
        
        # Process text
        inputs = processor(text, return_tensors="pt")
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs = model(**inputs).logits
        
        # Convert to audio (simplified)
        audio_array = mel_outputs.squeeze().numpy()
        
        # Save (this is simplified - real implementation needs vocoder)
        wavfile.write(filename, 22050, audio_array)
        
        print(f"✅ Tacotron audio saved: {filename}")
        return filename
        
    except ImportError:
        print("❌ Tacotron dependencies not available")
        return None

# ============================================
# 4. FACEBOOK SEAMLESS M4T
# ============================================

def seamless_tts(text, filename="seamless_output.wav"):
    """Facebook SeamlessM4T - Multilingual TTS"""
    print("=== SEAMLESS M4T ===")
    
    try:
        from transformers import SeamlessM4TProcessor, SeamlessM4TForTextToSpeech
        
        processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-medium")
        model = SeamlessM4TForTextToSpeech.from_pretrained("facebook/seamless-m4t-medium")
        
        # Process text
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate audio
        with torch.no_grad():
            audio_array = model.generate(**inputs).audio.squeeze().numpy()
        
        wavfile.write(filename, 24000, audio_array)
        print(f"✅ Seamless M4T audio saved: {filename}")
        return filename
        
    except ImportError:
        print("❌ Seamless M4T not available")
        return None

# ============================================
# 5. GOOGLE TTS (gTTS)
# ============================================

def gtts_tts(text, filename="gtts_output.wav"):
    """Google Text-to-Speech - Online service"""
    print("=== GOOGLE TTS ===")
    
    try:
        from gtts import gTTS
        import subprocess
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save as MP3 first
        temp_mp3 = "temp.mp3"
        tts.save(temp_mp3)
        
        # Convert to WAV using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_mp3, '-acodec', 'pcm_s16le', 
                '-ar', '22050', filename, '-y'
            ], check=True, capture_output=True)
            
            # Remove temp file
            os.remove(temp_mp3)
            
            print(f"✅ Google TTS audio saved: {filename}")
            return filename
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg not found. Install ffmpeg or use MP3 file")
            os.rename(temp_mp3, filename.replace('.wav', '.mp3'))
            return filename.replace('.wav', '.mp3')
            
    except ImportError:
        print("❌ gTTS not installed. Run: pip install gTTS")
        return None

# ============================================
# 6. ELEVENLABS (API)
# ============================================

def elevenlabs_tts(text, filename="elevenlabs_output.wav", api_key=None):
    """ElevenLabs TTS - High quality voices"""
    print("=== ELEVENLABS TTS ===")
    
    try:
        from elevenlabs import generate, save
        
        if not api_key:
            print("❌ ElevenLabs API key required")
            return None
        
        # Generate audio
        audio = generate(
            text=text,
            voice="Rachel",  # Default voice
            model="eleven_monolingual_v1",
            api_key=api_key
        )
        
        # Save audio
        save(audio, filename)
        
        print(f"✅ ElevenLabs audio saved: {filename}")
        return filename
        
    except ImportError:
        print("❌ ElevenLabs not installed. Run: pip install elevenlabs")
        return None

# ============================================
# 7. COMPARISON FUNCTION
# ============================================

def compare_all_models(text, output_dir="outputs"):
    """So sánh tất cả các model TTS"""
    print("🚀 COMPARING ALL TTS MODELS")
    print("=" * 50)
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    models = [
        ("Bark", lambda: bark_tts(text, f"{output_dir}/bark.wav")),
        ("Coqui TTS", lambda: coqui_tts(text, f"{output_dir}/coqui.wav")),
        ("Tacotron", lambda: tacotron_tts(text, f"{output_dir}/tacotron.wav")),
        ("Seamless M4T", lambda: seamless_tts(text, f"{output_dir}/seamless.wav")),
        ("Google TTS", lambda: gtts_tts(text, f"{output_dir}/gtts.wav")),
    ]
    
    results = {}
    
    for name, func in models:
        print(f"\n🎯 Testing {name}...")
        try:
            result = func()
            results[name] = "✅ Success" if result else "❌ Failed"
        except Exception as e:
            results[name] = f"❌ Error: {str(e)[:50]}"
    
    print("\n📊 RESULTS SUMMARY:")
    print("=" * 30)
    for name, status in results.items():
        print(f"{name}: {status}")
    
    return results

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Test text
    test_text = "Hello, My name is Dat, I am 22, I am a student at Hanoi university of science and technology"
    
    print("🎵 TEXT-TO-SPEECH MODELS COMPARISON")
    print("=" * 60)
    print(f"Text: {test_text}")
    print("=" * 60)
    
    # Chạy so sánh
    results = compare_all_models(test_text)
    
    print("\n🎉 COMPLETED! Check the 'outputs' folder for audio files.") 