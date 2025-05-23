import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForteBank RAG Chatbot",
    page_icon="🏦",
    layout="wide",
)

# ─── ENVIRONMENT ────────────────────────────────────────────────────────────────
# Load .env if present
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY не задана.")
    st.stop()

# Path to your committed PDFs
DATA_PATH = "./knowledge_base"

# ─── RAG PROMPT TEMPLATE ────────────────────────────────────────────────────────
RAG_TEMPLATE = """Ты — дружелюбный и профессиональный ассистент направления платежных систем ForteBank. 
Дай ответ на вопрос на основе следующего контекста:
{context}

# Инструкции
- Отвечай на русском.
- Поздоровайся и затем отвечай.
- Если контекст содержит явный ответ — приведи его.
- Если похожий контекст — предложи уточнить.
- Если нет — вежливо переключи на специалиста.

Вопрос: {question}
"""

# ─── CACHED RESOURCES ──────────────────────────────────────────────────────────
@st.cache_resource
def get_embedding_function():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

@st.cache_resource
def build_vector_store(_docs):
    """Split → embed → store entirely in RAM (no disk)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embed_fn = get_embedding_function()
    return FAISS.from_texts(
        texts,
        embed_fn,
        metadatas=metadatas
    )

@st.cache_resource(hash_funcs={FAISS: lambda _: None})
def create_rag_chain(vector_store, 'gpt-4o-mini', temperature=0):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def load_local_docs():
    """Load all PDFs from your committed folder."""
    paths = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
    docs = []
    for p in paths:
        docs.extend(PyPDFLoader(p).load())
    return docs

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.title("🏦 ForteBank RAG Chatbot")
    st.markdown("Задайте вопрос по платежным системам ForteBank.")

    # ─ List indexed PDFs ─
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Индексированные PDF")
    pdfs = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
    if pdfs:
        for pdf in pdfs:
            rel = os.path.relpath(pdf, DATA_PATH)
            st.sidebar.text(f"• {rel}")
    else:
        st.sidebar.text("— Нет PDF-файлов —")

    # ─ Upload new PDFs ─
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Загрузить PDF")
    uploaded = st.sidebar.file_uploader(
        "Выберите PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    if uploaded and not st.session_state.get("upload_processed", False):
        new_docs = []
        for f in uploaded:
            new_docs.extend(PyPDFLoader(f).load())
        combined = load_local_docs() + new_docs
        st.session_state.vector_store = build_vector_store(combined)
        st.session_state.upload_processed = True
        st.sidebar.success("✅ Загружено и проиндексировано!")

    # ─ Initialize vector store ─
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = build_vector_store(load_local_docs())

    # ─ Build RAG chain ─
    vector_store = st.session_state.vector_store
    rag_chain = create_rag_chain(vector_store, model_option, temperature)

    # ─ Chat UI ─
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Обдумываю ответ…"):
                try:
                    answer = rag_chain.invoke(prompt)
                except Exception as e:
                    answer = f"Ошибка: {e}"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()
