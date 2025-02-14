import streamlit as st
from TTS.api import TTS

import torch
import os
import warnings
warnings.filterwarnings("ignore")

# Set the device for computation (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_tts_model():
    """

        Loads the XTTS-v2 Text-To-Speech model and caches it for Streamlit.

        Returns:
            TTS: Loaded TTS model.

    """
    print("Loading the Text To Speech Model...")
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    model.to(device) # Device Map
    print("Text To Speech Model Loaded...")
    return model

# Get the current directory of the script or environment
current_dir = os.getcwd()

# Paths relative to the current directory
output_path = os.path.join(current_dir, "output/output_audio.wav")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def TextToSpeech(input_text):
    """

        Converts input text to speech using the XTTS-v2 model.

        Parameters:
        input_text (str): The input text to be converted to speech.

        Returns:
        str: Path to the generated speech audio file.

    """
    tts = load_tts_model() # Get the cached model

    tts.tts_to_file(
        text= input_text,
        file_path=output_path,
        language="en",
        speaker="Ana Florence",
        split_sentences=True
        )

    return output_path # Return the path of the generated audio file

if __name__ == "__main__":
    text = input("Input the Text : ")
    output_path = TextToSpeech(text)
    print("Audio File at : ",output_path)