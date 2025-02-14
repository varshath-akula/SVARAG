import warnings
warnings.filterwarnings("ignore")

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever


from dotenv import load_dotenv
from operator import itemgetter
import os
import shutil
import time

from speech_to_text import SpeechToText
from sentiment_analysis_module import get_tone_instructions
from text_to_speech import TextToSpeech

# Load the environment variables
load_dotenv()

# Retrieve Hugging Face API key from environment variables
HF_Token = os.getenv("HUGGINGFACE_API_KEY")

def configure():
    """

        Configures the runtime environment.

        - Adds FFmpeg to the system path to enable audio processing.

    """
    ffmpeg_dir = r'C:\ProgramData\chocolatey\bin'
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

configure() # Ensure environment configuration is set

def setup_qa_system(file_path):
    """

        Initializes the RAG (Retrieval-Augmented Generation) system.

        - Clears the existing ChromaDB to ensure fresh document indexing.
        - Loads the provided PDF document and extracts text.
        - Splits the text into manageable chunks.
        - Creates an embedding-based vector database.
        - Sets up a hybrid search retriever (vector + keyword-based).
        - Initializes an LLM (Mistral-7B) to generate responses.

        Args:
            file_path (str): Path to the document to be processed.

        Returns:
            tuple:
                chain (LangChain pipeline): The initialized question-answering chain.
                doc_pipeline_times (dict): Dictionary containing processing times for each document pipeline step.

    """
    doc_pipeline_times = {}
    # Define a fixed path for ChromaDB storage
    chroma_db_path = "chroma_db"

    # Attempt to delete the existing vector database
    if os.path.exists(chroma_db_path):
        try:
            print("Attempting to release ChromaDB resources...")

            # Connect to ChromaDB and clear collections
            chroma_client = Chroma(persist_directory=chroma_db_path)
            chroma_client.delete_collection()  # Ensure collections are removed
            del chroma_client  # Delete the object reference
            time.sleep(2)  # Allow time for file release

            # Now remove the database folder safely
            shutil.rmtree(chroma_db_path, ignore_errors=True)
            print("✅ ChromaDB successfully cleared.")
        except Exception as e:
            print(f"⚠️ Error while clearing ChromaDB: {e}")


    print("Loading the Document...")
    start_time = time.time()
    # Load the document using LangChain's UnstructuredPDFLoader
    Loader = UnstructuredPDFLoader(file_path)
    docs = Loader.load()
    doc_pipeline_times["document_loading_time"] = time.time() - start_time

    print("Processing the Document...")

    start_time = time.time()
    # Split the document into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    doc_pipeline_times["text_splitting_time"] = time.time() - start_time

    # Create Embeddings for document chunks
    start_time = time.time()
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_Token, model_name="BAAI/bge-base-en-v1.5")
    doc_pipeline_times["embedding_creation_time"] = time.time() - start_time

    # Store embeddings in ChromaDB
    start_time = time.time()
    vector_db = Chroma.from_documents(chunks, embeddings,persist_directory=chroma_db_path)
    doc_pipeline_times["vector_db_storage_time"] = time.time() - start_time

    # Hybrid Search retrieval (Semantic Search + Keyword Search)
    start_time = time.time()
    search_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(retrievers=[search_retriever, keyword_retriever], weights=[0.5, 0.5])
    doc_pipeline_times["retriever_setup_time"] = time.time() - start_time

    # Load the Mistral-7B language model via Hugging Face Hub
    start_time = time.time()
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 2048},
        huggingfacehub_api_token=HF_Token,
    )
    doc_pipeline_times["llm_loading_time"] = time.time() - start_time

    # Define the prompt template
    start_time = time.time()
    template = """
        <|system|>>
        You are a helpful AI Assistant that follows instructions extremely well.
        Use the following context and tone_instructions to answer user question.

        Think step by step before answering the question. You will get a $1000 tip if you provide correct answer.
        Your response must be formatted as a single paragraph with no bullet points, bold text, or special symbols 
        that might interfere with text-to-speech output. Ensure smooth readability and coherence.

        CONTEXT: {context}
        Tone Instructions : {tone_instructions}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """
    # Create a LangChain-based prompt and processing pipeline
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = (
            {"context": itemgetter("query") | ensemble_retriever,
             "query": itemgetter("query"),
             "tone_instructions": itemgetter("tone_instructions")}
            | prompt
            | llm
            | output_parser
    )
    doc_pipeline_times["chain_setup_time"] = time.time() - start_time

    return chain , doc_pipeline_times

# **QA System Wrapper Function**
def process_query(audio_file_path, qa_chain):
    """

       Processes a user query by:
       1. Transcribing speech from an audio file.
       2. Analyzing sentiment and tone from vocal characteristics.
       3. Generating a text-based response using a retrieval-augmented model.
       4. Converting the generated response into speech.

       Args:
           audio_file_path (str): Path to the user's audio query.
           qa_chain (LangChain pipeline): The initialized RAG-based question-answering system.

       Returns:
           tuple: A tuple containing:
               - response_text (str): The AI-generated text response.
               - response_audio_path (str): The path to the generated speech output.
               - step_times (dict) : Dictionary containing processing times for each step.
       """
    step_times = {}

    print("Transcribing query...")
    start_time = time.time()
    query_text = SpeechToText(audio_file_path)
    step_times["transcription_time"] = time.time()-start_time
    print("Query Text : ",query_text)

    print("Analyzing sentiment...")
    start_time = time.time()
    tone_instructions = get_tone_instructions(audio_file_path, query_text)
    step_times["sentiment_analysis_time"] = time.time() - start_time
    print("Tone Instructions : ",tone_instructions)

    print("Generating response...")
    start_time = time.time()
    response = qa_chain.invoke(input={"tone_instructions": tone_instructions, "query": query_text})
    step_times["response_generation_time"] = time.time() - start_time
    response_text = response.split('<|assistant|>')[-1].strip()
    print("Response Text : ",response_text)

    print("Converting response to speech...")
    start_time = time.time()
    response_audio_path = TextToSpeech(response_text)
    step_times["tts_time"] = time.time() - start_time

    return response_text , response_audio_path , step_times