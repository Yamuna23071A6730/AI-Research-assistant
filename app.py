import streamlit as st
from st_audiorec import st_audiorec
import os
from dotenv import load_dotenv
import pytesseract
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import requests
import docx
import base64
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# -------------------- Configuration --------------------
load_dotenv()

# Load API Key securely
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Set it in .env or Streamlit Secrets.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# OCR Setup
OCR_ENABLED = False
tess_path = os.getenv("TESSERACT_PATH")
if tess_path and os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
    OCR_ENABLED = True

# -------------------- Session Init --------------------
st.session_state.setdefault("question_answer_history", [])
st.session_state.setdefault("transcribed_text", "")

language_map = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Spanish": "es", "French": "fr"
}

# -------------------- Utility Functions --------------------
def text_to_speech(text, language):
    try:
        lang_code = language_map.get(language, "en")
        tts = gTTS(text=text, lang=lang_code)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_base64 = base64.b64encode(audio_file.read()).decode()
        st.markdown(f"""<audio controls autoplay>
                         <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>""",
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {e}")

def get_pdf_text(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
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
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300).split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

def get_conversational_chain():
    prompt = PromptTemplate(
        template="""You are a helpful AI assistant. Use the following context to answer the question.\n\nContext: {context}\nQuestion: {question}\n\nHelpful Answer:""",
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def translate_text(text, target_lang):
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(f"Translate to {target_lang}:\n{text}").text

def search_google(query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        return model.generate_content(f"Search and summarize: {query}").text
    except Exception as e:
        return f"Search failed: {e}"

def user_input(question, lang):
    with st.spinner("Thinking..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        try:
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception:
            st.error("No vector store found. Please upload and process documents first.")
            return
        docs = db.similarity_search(question)
        chain = get_conversational_chain()
        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answer = result["output_text"]

        if "cannot be answered" in answer.lower() or len(answer.strip()) < 10:
            answer = search_google(question)

        translated = translate_text(answer, lang)
        st.session_state.question_answer_history.append({"question": question, "answer": translated})
        st.markdown(f"**Answer:** {translated}")

# -------------------- UI Layout --------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown("""
<style>
.stButton > button {
  background: linear-gradient(to right, #6a11cb, #20948B);
  color: white; font-weight: bold; border-radius: 12px;
  transition: transform 0.3s ease-in-out;
}
.stButton > button:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# -------------------- Left Column (Main) --------------------
with col1:
    st.title("🤖 AI Research Assistant")

    mode = st.radio("Choose Input Mode:", ["Type", "Speak"], horizontal=True)
    language = st.selectbox("Select Language", list(language_map.keys()))
    audio_out = st.checkbox("Enable Audio Reply")

    if mode == "Speak":
        st.info("Record your question below.")
        audio_bytes = st_audiorec()
        if audio_bytes:
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(BytesIO(audio_bytes)) as src:
                    audio = recognizer.record(src)
                st.session_state.transcribed_text = recognizer.recognize_google(audio, language=language_map[language])
            except Exception as e:
                st.error(f"Speech Recognition failed: {e}")
        question = st.text_input("Transcribed Question:", st.session_state.transcribed_text)
    else:
        question = st.text_input("Type your Question:")

    if st.button("Submit"):
        if question:
            user_input(question, language)
            if audio_out:
                answer = st.session_state.question_answer_history[-1]["answer"]
                text_to_speech(answer, language)
        else:
            st.warning("Please ask a question.")

    st.subheader("🌐 Provide a URL")
    url = st.text_input("Enter URL to extract and process text from")

    if st.button("Process URL"):
        if url:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                text = "\n".join(p.text for p in paragraphs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("URL processed and vectorized.")
            except Exception as e:
                st.error(f"Failed to process URL: {e}")
        else:
            st.warning("Please enter a valid URL.")

    st.subheader("📂 Upload Files")
    files = st.file_uploader("Upload PDFs or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Process Files"):
        if not files:
            st.warning("Please upload some files.")
        else:
            raw = ""
            pdfs = [f for f in files if f.name.endswith(".pdf")]
            docs = [f for f in files if f.name.endswith(".docx")]
            if pdfs: raw += get_pdf_text(pdfs)
            if docs: raw += get_docx_text(docs)
            if raw:
                chunks = get_text_chunks(raw)
                get_vector_store(chunks)
                st.success("Files processed successfully!")
            else:
                st.error("No extractable text found.")

    if OCR_ENABLED:
        st.subheader("🖼️ Upload Image for OCR (Local Use Only)")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if image_file:
            try:
                img = Image.open(image_file)
                extracted_text = pytesseract.image_to_string(img)
                st.text_area("Extracted Text", extracted_text, height=200)
                if st.button("Vectorize Image Text"):
                    chunks = get_text_chunks(extracted_text)
                    get_vector_store(chunks)
                    st.success("Image text processed and vectorized.")
            except Exception as e:
                st.error(f"OCR failed: {e}")
    else:
        st.info("🛑 OCR (image upload) is disabled on Streamlit Cloud.")

# -------------------- Right Column (History) --------------------
with col2:
    st.subheader("📝 Conversation History")
    if st.session_state.question_answer_history:
        for item in reversed(st.session_state.question_answer_history):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
    else:
        st.info("No history yet.")
    if st.button("Clear History", key="clear"):
        st.session_state.question_answer_history = []
        st.success("History cleared.")
