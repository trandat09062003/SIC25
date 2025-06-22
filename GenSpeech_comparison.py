# ============================================
# PHƯƠNG PHÁP 1: Sử dụng trực tiếp Bark Model
# ============================================

from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
import numpy as np

def method1_direct_bark():
    """Các phương pháp gen audio bằng bark"""
    """Phương pháp 1: Sử dụng trực tiếp Bark model"""
    print("=== PHƯƠNG PHÁP 1: Direct Bark Model ===")
    
    # Load model và processor
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    
    # Cấu hình voice
    voice_preset = "v2/en_speaker_6"
    
    # Text input
    text = "Hello, My name is Dat, I am 22, I am a student at Hanoi university of science and technology"
    
    # Xử lý text
    inputs = processor(text, voice_preset=voice_preset)
    
    # Sinh audio
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Lưu file
    sample_rate = 24000
    wavfile.write("output_method1.wav", sample_rate, audio_array)
    
    print(f"✅ Audio saved: output_method1.wav")
    print(f"📝 Text: {text}")
    print(f"🎤 Voice: {voice_preset}")
    print(f"📊 Audio shape: {audio_array.shape}")
    print(f"🔊 Sample rate: {sample_rate} Hz")
    print()

# ============================================
# PHƯƠNG PHÁP 2: Sử dụng function wrapper
# ============================================

def generate_audio(text_prompt, voice_preset="v2/en_speaker_6"):
    """Phương pháp 2: Function wrapper cho Bark"""
    print("=== PHƯƠNG PHÁP 2: Function Wrapper ===")
    
    # Load model (có thể cache để tái sử dụng)
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    
    # Xử lý text với music notation
    inputs = processor(text_prompt, voice_preset=voice_preset)
    
    # Sinh audio
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    
    return audio_array

def method2_function_wrapper():
    """Demo phương pháp 2"""
    # Text với music notation
    text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
    """
    
    # Sinh audio
    audio_array = generate_audio(text_prompt)
    
    # Lưu file
    sample_rate = 24000
    wavfile.write("output_method2.wav", sample_rate, audio_array)
    
    print(f"✅ Audio saved: output_method2.wav")
    print(f"📝 Text: {text_prompt.strip()}")
    print(f"🎵 Music notation: ♪ ♪")
    print(f"📊 Audio shape: {audio_array.shape}")
    print(f"🔊 Sample rate: {sample_rate} Hz")
    print()

# ============================================
# PHƯƠNG PHÁP 3: Advanced với nhiều voice
# ============================================

def method3_advanced_features():
    """Phương pháp 3: Advanced features"""
    print("=== PHƯƠNG PHÁP 3: Advanced Features ===")
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    
    # Danh sách voice presets
    voice_presets = [
        "v2/en_speaker_6",    # Female voice
        "v2/en_speaker_9",    # Male voice  
        "v2/en_speaker_0",    # Another voice
    ]
    
    # Text với nhiều loại
    texts = [
        "Hello, this is a normal conversation.",
        "♪ Let's sing a beautiful song together ♪",
        "[LAUGHTER] That was funny! [LAUGHTER]"
    ]
    
    for i, (text, voice) in enumerate(zip(texts, voice_presets)):
        inputs = processor(text, voice_preset=voice)
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        
        filename = f"output_method3_voice_{i+1}.wav"
        wavfile.write(filename, 24000, audio_array)
        
        print(f"✅ {filename} - Voice: {voice}")
        print(f"📝 Text: {text[:50]}...")
        print()

# ============================================
# CHẠY TẤT CẢ PHƯƠNG PHÁP
# ============================================

if __name__ == "__main__":
    print("🚀 BẮT ĐẦU SO SÁNH CÁC PHƯƠNG PHÁP TEXT-TO-SPEECH")
    print("=" * 60)
    
    # Chạy từng phương pháp
    method1_direct_bark()
    method2_function_wrapper()
    method3_advanced_features()
    
    print("\n🎉 HOÀN THÀNH! Kiểm tra các file .wav được tạo ra.") 