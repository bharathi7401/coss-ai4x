# test_asr.py
import os
from services.asr_service import ASRService

def main():
    # Path to your wav file
    wav_path = os.path.join("asr_testing", "tamil_audio.wav")  # change filename if needed
    
    # Read wav file as bytes
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    
    # Create ASR service object
    asr_service = ASRService()
    
    # Call transcription
    result = asr_service.transcribe_audio(audio_bytes, audio_format="wav")
    
    # Print the output
    print("Transcription Result:")
    print(result)

if __name__ == "__main__":
    main()
