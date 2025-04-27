# src/feedback.py
import threading
from gtts import gTTS
from playsound import playsound
import uuid
import os
import streamlit as st

audio_lock = threading.Lock()

# In your speak_async or where audio feedback is triggered

def speak_async(text):
    if st.session_state.get('is_analyze_video_flow', False):
        # Skip audio feedback if we're in "Analyze Video" flow
        print(f"[VOICE] Skipping audio feedback as we're in analyze video flow: {text}")
        return
    
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def speak(text):
    if st.session_state.get('is_analyze_video_flow', False):
        # Skip audio feedback if we're in "Analyze Video" flow
        print(f"[VOICE] Skipping audio feedback as we're in analyze video flow: {text}")
        return
    
    # Normal audio feedback logic
    if not audio_lock.acquire(blocking=False):
        print(f"[VOICE] Skipping (audio already playing): {text}")
        return
    try:
        print(f"[VOICE] Playing: {text}")
        tts = gTTS(text=text, lang='en')
        filename = f"/tmp/{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("🔴 TTS Error:", e)
    finally:
        audio_lock.release()
