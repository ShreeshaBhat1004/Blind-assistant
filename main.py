import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import numpy as np
import json
import subprocess
import time
import requests
import traceback

# ====================================================================
# --- SETTINGS ---
# ====================================================================
# llama.cpp server endpoint
LLM_SERVER_ENDPOINT = "http://127.0.0.1:8080/v1/completions"

# -- Ears Settings --
STT_MODEL_SIZE = "tiny.en"
RECORDING_FILENAME = "user_audio.wav"
RECORDING_DURATION = 5
SAMPLE_RATE = 16000

# -- Mouth Settings --
TTS_MODEL_PATH = "/home/shreesha/en_US-lessac-medium.onnx"
TTS_OUTPUT_FILENAME = "assistant_speech.wav"

# ====================================================================
# --- INITIALIZE COMPONENTS ---
# ====================================================================
print("--- Initializing Assistant ---")
print("✅ Brain is running via llama.cpp server.")

print(f"Loading Ears (Whisper '{STT_MODEL_SIZE}')...")
try:
    # Use CPU for stability on Jetson
    stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")
    print("✅ Ears are ready.")
except Exception as e:
    traceback.print_exc()
    print(f"❌ Failed to load Whisper: {e}")
    exit(1)

print("✅ Mouth is ready (will be called as a command).")

# ====================================================================
# --- HELPER FUNCTIONS ---
# ====================================================================
def run_llm(prompt):
    """
    Send a request to llama.cpp server (/v1/completions) and return the response text.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_SERVER_ENDPOINT, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "").strip()
        return result.get("content", "").strip()

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error communicating with LLM server: {e}")
        return "Error: Could not connect to the brain server."

def speak(text):
    try:
        print(f"Assistant 🗣️: {text}")
        command = ["piper", "--model", TTS_MODEL_PATH, "--output_file", TTS_OUTPUT_FILENAME]
        subprocess.run(command, input=text, text=True, check=True, capture_output=True)
        data, fs = sf.read(TTS_OUTPUT_FILENAME, dtype='float32')
        sd.play(data, fs, blocking=True)  # blocking playback to avoid underruns
        sd.wait()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ TTS failed: {e}")

def listen():
    try:
        print(f"\n🎙️  Listening for {RECORDING_DURATION} seconds...")
        recording = sd.rec(int(RECORDING_DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        sf.write(RECORDING_FILENAME, recording, SAMPLE_RATE)
        return RECORDING_FILENAME
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Recording failed: {e}")
        return None

def transcribe(audio_file):
    try:
        print("🧠 Transcribing...")
        segments, _ = stt_model.transcribe(audio_file, beam_size=5)
        return "".join(segment.text for segment in segments).strip()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Transcription failed: {e}")
        return ""

def execute_command(command):
    print(f"🚀 Executing: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True, timeout=15)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: The command took too long to execute."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Command execution failed: {e}"

# ====================================================================
# --- MAIN LOOP ---
# ====================================================================
if __name__ == "__main__":
    speak("Assistant online and ready.")
    while True:
        try:
            audio_file = listen()
            if not audio_file:
                speak("Recording failed. Please check your microphone.")
                continue

            user_request = transcribe(audio_file)

            if not user_request:
                speak("I didn't hear anything. Please try again.")
                continue

            print(f"User 🎤: {user_request}")

            prompt_for_command = (
                f"Convert the following request to a single, executable Linux shell command. "
                f"Only output the command. Request: '{user_request}'"
            )
            command_to_run = run_llm(prompt_for_command)

            if any(word in command_to_run for word in ["sudo", "rm -rf", "mkfs"]):
                speak("I'm sorry, that seems like a potentially dangerous command, so I won't run it.")
                continue

            if "Error:" in command_to_run or not command_to_run:
                speak("I'm sorry, I couldn't understand that command.")
                continue

            command_output = execute_command(command_to_run)

            prompt_for_summary = (
                f"The user ran a command and got this output: '{command_output}'. "
                f"Describe the result in a brief, natural sentence."
            )
            summary = run_llm(prompt_for_summary)

            speak(summary)

        except KeyboardInterrupt:
            print("\nShutting down assistant.")
            speak("Goodbye.")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"An unexpected error occurred: {e}")
            speak("I've encountered an error and need to restart.")
            break

