import whisper
import streamlit as st

@st.cache_resource
def load_stt_model():
    print("Loading Speech To Text Model...")
    model = whisper.load_model("base.en")
    print("Speech To Text Model Loaded...")
    return model


def SpeechToText(audio_file_path):
    """
        Converts speech from an audio file to text using the Whisper model.

        Args:
            audio_file_path (str): The path to the audio file to be transcribed.

        Returns:
            str: The transcribed text from the audio file.
    """
    STTmodel = load_stt_model()
    result = STTmodel.transcribe(audio_file_path)
    return result['text']

if __name__ == "__main__":
    audio_file_path = input("please enter the audio file path : ")
    transcription = SpeechToText(audio_file_path)
    print("Transcription : ",transcription)
