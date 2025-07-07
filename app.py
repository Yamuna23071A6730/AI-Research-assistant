import streamlit as st
from st_audiorec import st_audiorec
import os
import time
import requests
import docx
import pytesseract
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import base64

# ------------------ Configuration ------------------
load_dotenv()

# Updated Poppler path for user
poppler_path = r"C:\\Users\\yamun\\Downloads\\poppler-25.07.0\\Library\\bin" if os.name == "nt" else "/usr/bin"

if "TESSERACT_PATH" in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_PATH"]

if "question_answer_history" not in st.session_state:
    st.session_state.question_answer_history = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

language_map = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Spanish": "es", "French": "fr"
}


# ------------------ Text-to-Speech ------------------
def text_to_speech(text, language):
    try:
        lang_code = language_map.get(language, "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_base64 = base64.b64encode(audio_file.read()).decode()
        audio_html = f'''<audio controls autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'''
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {e}")

# ------------------ Helper Functions ------------------
def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

def get_pdf_text(files):
    text = ""
    for f in files:
        pdf_bytes = f.read()
        reader = PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            content = page.extract_text()
            text += content if content else ""
    return text

def get_docx_text(files):
    text = ""
    for f in files:
        doc = docx.Document(f)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_conversational_chain():
    prompt = PromptTemplate(template="""Context:\n{context}\n\nQ: {question}\n\nA:""", input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def search_google(query):
    model = genai.GenerativeModel('gemini-2.0-pro')
    try:
        return model.generate_content(f"Summary of: {query}").text
    except Exception as e:
        return f"Google Search Failed: {e}"

def user_input(question, lang):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return

    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = result["output_text"]

    if "cannot be answered" in answer.lower():
        answer = search_google(question)

    translated = translate_text(answer, lang)
    st.session_state.question_answer_history.append({"question": question, "answer": translated})
    st.write("**Answer:**", translated)

def translate_text(text, target_language):
    model = genai.GenerativeModel('gemini-2.0-pro')
    return model.generate_content(f"Translate to {target_language}:\n{text}").text

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.markdown("""
<style>
body { background-color: #f2f2f2; color: #333; font-family: 'Segoe UI', sans-serif; }
.stButton > button {
  background: linear-gradient(to right, #6a11cb, #2575fc);
  color: white;
  font-weight: bold;
  border-radius: 12px;
  transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
  transform: scale(1.05);
  background: linear-gradient(to right, #ff416c, #ff4b2b);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
h1, h2, h3, .stMarkdown {
  animation: fadeInUp 0.7s ease-in-out;
}
@keyframes fadeInUp {
  0% {opacity: 0; transform: translateY(20px);}
  100% {opacity: 1; transform: translateY(0);}
}
.footer {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Personal AI Research Assistant")
st.write("Upload documents or provide a URL, ask a question, and get answers in your preferred language.")

st.markdown("""
<div class="footer">
  <a href="https://github.com/saivardhan507/AI-Research-Project" target="_blank">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="40" title="GitHub Repository"/>
  </a>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Choose Input Mode:", ["Type", "Speak"], horizontal=True)
language = st.selectbox("Select Language", list(language_map.keys()))

if mode == "Speak":
    st.info("Record your question below.")
    audio_bytes = st_audiorec()
    if audio_bytes:
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(BytesIO(audio_bytes)) as src:
                data = recognizer.record(src)
            st.session_state.transcribed_text = recognizer.recognize_google(data, language=language_map.get(language, "en"))
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

    question = st.text_input("Transcribed Question:", value=st.session_state.transcribed_text)
else:
    question = st.text_input("Type your Question:")

audio_out = st.checkbox("Enable Audio Reply")

if st.button("Submit"):
    if question:
        user_input(question, language)
        if audio_out:
            latest = st.session_state.question_answer_history[-1]["answer"]
            text_to_speech(latest, language)
    else:
        st.error("Please enter a question first.")

st.subheader("📂 Upload Files")
files = st.file_uploader("Upload PDFs or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Process Files"):
    raw = ""
    if files:
        pdfs = [f for f in files if f.name.endswith(".pdf")]
        docs = [f for f in files if f.name.endswith(".docx")]
        if pdfs: raw += get_pdf_text(pdfs)
        if docs: raw += get_docx_text(docs)
        if raw:
            chunks = get_text_chunks(raw)
            get_vector_store(chunks)
            st.success("Documents processed successfully.")
        else:
            st.error("No text found in uploaded files.")

if st.session_state.question_answer_history:
    st.subheader("📝 Conversation History")
    for item in st.session_state.question_answer_history:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
if st.button("Clear History"):
    st.session_state.question_answer_history = []
    st.success("Conversation history cleared.")
