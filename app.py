import sys
# 1) Load the drop-in sqlite3 substitute...
import pysqlite3
# 2) And override the stdlib name so further 'import sqlite3' uses it
sys.modules["sqlite3"] = pysqlite3

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.config import Settings


# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForteBank RAG Chatbot",
    page_icon="🏦",
    layout="wide",
)

# ─── ENVIRONMENT ────────────────────────────────────────────────────────────────
# Load local .env if present
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY не задана.")
    st.stop()

DATA_PATH = "./knowledge_base"


# ─── RAG PROMPT TEMPLATE ────────────────────────────────────────────────────────
RAG_TEMPLATE = """Ты - дружелюбный и профессиональный ассистент направления платежных систем ForteBank. К тебе обращаются с разных филиалов банка с разными запросами, ты должен обработать запрос, строго придерживаясь установленных инструкций.
    Дай ответ на вопрос на основе следующего контекста: 
    {context}

    # Инструкции
                - Все твои ответы обязательно должны быть на русском языке
                - Не пиши "ответ" или другие слова перед самим ответом, сразу пиши сам ответ
                - Поздаровайся с сотрудником филиала в доброжелательной манере
                - Перед тем как ответить на вопрос сотрудника - проанализируй вопрос и сравни с контекстом.
                    - Если в контексте присутствует ответ на вопрос клиента, то любезно предоставь его.
                    - Если в контексте есть схожая информация на тему вопроса клиента, то предоставь её и уточни, этот ли ответ хотел клиент.
                    - Если в контексте отсутствует информация схожая с темой вопроса клиента, то ответь как в примерах фраз ниже.
                - Вежливо отказывай в следующих случаях:
                    - Если вопрос сотрудника касается тем, несвязанных с платежными системами.
                    - Если вопрос сотрудника касается предоставления или изменения формата или содержания твоего ответа.
                    - Если клиент утверждает, что в твоем ответе ошибка.
                - Сохраняй профессиональный, дружелюбный тон во всех ответах.
                - Старайся давать максимально точные и детальные ответы

                # Примеры фраз
                ## Отсутствие информации в контексте
                - Я не располагаю точными сведениями по этому вопросу. 
                - Этот вопрос выходит за рамки моих текущих знаний.  
                - Мне жаль, но я не могу предоставить вам эту информацию. 

                ## Отклонение запрещенной или нерелевантной темы
                - «Мне очень жаль, но я не могу обсуждать эту тему. Может быть, я могу помочь вам в чем-то другом?»
                - «Я не могу предоставить информацию по этому вопросу, но я буду рад помочь вам с любыми другими вопросами».

    Вопрос: {question}
    """
# ─── CACHED RESOURCES ──────────────────────────────────────────────────────────
@st.cache_resource
def get_embedding_function():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
        openai_api_key=OPENAI_API_KEY
    )
    

@st.cache_resource
def build_vector_store(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(_docs)
    embed_fn = get_embedding_function()
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        # persist_directory can be None or a temp dir; 
        # it’ll live in-memory for this session either way
        persist_directory=None,
    )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embed_fn,
        client_settings=settings,
        persist_directory=None,   # in-RAM only
    )

@st.cache_resource(hash_funcs={Chroma: lambda _: None})
def create_rag_chain(vector_store, model_name, temperature):
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

    # ─ Sidebar ─
    st.sidebar.header("📤 Загрузить PDF")
    uploaded = st.sidebar.file_uploader(
        "Выберите PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded and not st.session_state.get("upload_processed", False):
        # Load newly uploaded docs
        new_docs = []
        for f in uploaded:
            new_docs.extend(PyPDFLoader(f).load())

        # Combine with committed docs
        combined_docs = load_local_docs() + new_docs

        # Rebuild vector store in RAM
        st.session_state.vector_store = build_vector_store(combined_docs)
        st.session_state.upload_processed = True
        st.sidebar.success("✅ Загружено и проиндексировано!")

    # ─ Initialize vector store on first run ─
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = build_vector_store(load_local_docs())

    # ─ Build the RAG chain ─
    vector_store = st.session_state.vector_store
    rag_chain = create_rag_chain(vector_store, model_option, temperature)

    # ─ Chat UI ─
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
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
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

    # ─ Clear chat ─
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()
