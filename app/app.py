import streamlit as st
from audio_recorder_streamlit import audio_recorder

import os
import datetime
import time

from rag_workflow import setup_qa_system,process_query

# **Initialize UI**
st.title("🎙️ Voice-Activated RAG Chatbot")
st.write("Upload a document and start asking questions!")

#**Initialize session state**
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None  # Persist qa_chain
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = None  # Track document name

# **File Upload**
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:

    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    # **Check if New File is Uploaded**
    if st.session_state.document_uploaded != uploaded_file.name:
        st.session_state.document_uploaded = uploaded_file.name  # Update the tracked file

        # **Reset and Remove Old Data**
        if st.session_state.qa_chain:
            st.session_state.qa_chain = None  # Reset QA Chain
            for file in os.listdir("uploaded_docs"):
                os.remove(os.path.join("uploaded_docs", file))  # Delete old files
    os.makedirs("uploaded_docs", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # **Initialize RAG System**
    st.write("⚙ Processing document...")
    start_time = time.time()

    st.session_state.qa_chain,doc_pipeline_times = setup_qa_system(file_path)
    end_time = time.time()
    st.success(f"✅ Document uploaded and processed in **{end_time - start_time:.2f} seconds**! You can now ask questions.")

    # **Show Document Pipeline Timing Breakdown**
    st.subheader("⏳ Document Processing Time Breakdown:")
    st.write(f"📖 **Document Loading**: {doc_pipeline_times['document_loading_time']:.2f} seconds")
    st.write(f"✂ **Text Splitting**: {doc_pipeline_times['text_splitting_time']:.2f} seconds")
    st.write(f"🔢 **Embedding Creation**: {doc_pipeline_times['embedding_creation_time']:.2f} seconds")
    st.write(f"🏦 **Vector DB Storage**: {doc_pipeline_times['vector_db_storage_time']:.2f} seconds")
    st.write(f"🔍 **Retriever Setup**: {doc_pipeline_times['retriever_setup_time']:.2f} seconds")
    st.write(f"🧠 **LLM Loading**: {doc_pipeline_times['llm_loading_time']:.2f} seconds")
    st.write(f"📝 **Chain Setup**: {doc_pipeline_times['chain_setup_time']:.2f} seconds")


if st.session_state.qa_chain:
    # ** User Choice : Upload Audio / Record Audio
    query_format = st.radio(
        "How do you want to ask your question ?",
        ("Upload a WAV file","Record Audio"),
    )
    if query_format == "Upload a WAV file":
        # **Audio File Input**
        audio_file = st.file_uploader("Upload a speech query (WAV format)", type=["wav"])
        if audio_file:
            audio_path = os.path.join("uploaded_queries", audio_file.name)
            os.makedirs("uploaded_queries", exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            total_start_time = time.time()
            # **Process Query**
            st.write("🎙️ Processing your query...")
            response_text , response_audio_path , step_times = process_query(audio_path, st.session_state.qa_chain)
            total_end_time = time.time()

            # **Display & Play Response Audio**
            st.subheader("🔊 AI Response:")
            # st.write(response_text)
            with open(response_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")

            # **Show Time Breakdown**
            st.subheader("⏳ Processing Time Breakdown:")
            st.write(f"⏱ **Total Query Processing Time**: {total_end_time - total_start_time:.2f} seconds")
            st.write(f"📌 **Speech Transcription**: {step_times['transcription_time']:.2f} seconds")
            st.write(f"📌 **Sentiment Analysis**: {step_times['sentiment_analysis_time']:.2f} seconds")
            st.write(f"📌 **Response Generation**: {step_times['response_generation_time']:.2f} seconds")
            st.write(f"📌 **Text-to-Speech (TTS)**: {step_times['tts_time']:.2f} seconds")

            os.remove(response_audio_path)

    elif query_format == "Record Audio":
        recorded_audio = audio_recorder("🎤 Click below to record your voice query:",pause_threshold=10.0)

        if recorded_audio:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"recorded_audio_{timestamp}.wav"
            recorded_audio_path = f"recorded_queries/{file_name}"
            os.makedirs("recorded_queries",exist_ok=True)
            st.write("Your Query : ")
            st.audio(recorded_audio)
            with open(recorded_audio_path, "wb") as f:
                f.write(recorded_audio)

            total_start_time = time.time()

            # Process the Query
            st.write("🎙️ Processing your query...")
            response_text , response_audio_path , step_times = process_query(recorded_audio_path,st.session_state.qa_chain)

            total_end_time = time.time()

            # Display & Play Response Audio
            st.subheader("🔊 AI Response:")
            with open(response_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")

            # **Show Time Breakdown**
            st.subheader("⏳ Query Processing Time Breakdown:")
            st.write(f"⏱ **Total Query Processing Time**: {total_end_time - total_start_time:.2f} seconds")
            st.write(f"📌 **Speech Transcription**: {step_times['transcription_time']:.2f} seconds")
            st.write(f"📌 **Sentiment Analysis**: {step_times['sentiment_analysis_time']:.2f} seconds")
            st.write(f"📌 **Response Generation**: {step_times['response_generation_time']:.2f} seconds")
            st.write(f"📌 **Text-to-Speech (TTS)**: {step_times['tts_time']:.2f} seconds")

            # After the response is displayed remove the query and the response audios
            os.remove(recorded_audio_path)
            os.remove(response_audio_path)