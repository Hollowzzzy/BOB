import speech_recognition as sr
from faster_whisper import WhisperModel
import os
import tempfile

model = WhisperModel("base")

recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True

def save_audio_to_file(audio):
    """Write microphone audio to a temporary wav file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.get_wav_data())
        return f.name

while True:
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source)

        file_path = save_audio_to_file(audio)

        segments, info = model.transcribe(file_path)

        text = "".join(segment.text for segment in segments).strip()

        os.remove(file_path)

        if not text:
            continue

        print("You:", text)

        if "exit" in text.lower():
            break

        os.system(f'say "{text}"')

    except KeyboardInterrupt:
        print("\nStopped manually.")
        break

    except Exception as e:
        print("Error:", e)
