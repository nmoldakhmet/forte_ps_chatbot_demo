import os
import glob
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="ForteBank RAG Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
@st.cache_resource
def load_environment():
    # Try to load local .env file (for development)
    env_path = "/home/nurbek/Forte/payment_systems/code/env_variables.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # For Streamlit Cloud, get from st.secrets or environment
    return {
        "DATA_PATH": st.secrets.get("DATA_PATH", os.getenv("DATA_PATH", "./data")),
        "CHROMA_PATH": st.secrets.get("CHROMA_PATH", os.getenv("CHROMA_PATH", "./chroma_db")),
        "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    }

# RAG Template
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

@st.cache_data
def load_all_documents(data_path):
    """Loads all PDF documents from the specified data path including subdirectories."""
    all_documents = []
    pdf_files = glob.glob(os.path.join(data_path, "**/*.pdf"), recursive=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            status_text.text(f"Loading {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            progress_bar.progress((i + 1) / len(pdf_files))
        except Exception as e:
            st.error(f"Error loading {pdf_path}: {str(e)}")
    
    status_text.text(f"Loaded {len(all_documents)} total pages from {len(pdf_files)} PDF files")
    return all_documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

@st.cache_resource
def get_embedding_function():
    """Initializes OpenAI embeddings using text-embedding-3-small model."""
    env_vars = load_environment()
    if not env_vars["OPENAI_API_KEY"]:
        st.error("OPENAI_API_KEY environment variable not set.")
        st.stop()
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    return embeddings

@st.cache_resource
def get_vector_store(persist_directory):
    """Initializes or loads the Chroma vector store."""
    embedding_function = get_embedding_function()
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        return vectorstore
    else:
        return None

def index_documents(chunks, embedding_function, persist_directory):
    """Indexes document chunks into the Chroma vector store."""
    with st.spinner("Indexing documents... This may take a few minutes."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    return vectorstore

@st.cache_resource(
    hash_funcs={Chroma: lambda _: None}   # ignore hashing for Chroma instances
)
def create_rag_chain(vector_store, openai_model="gpt-3.5-turbo", temperature=0):
    """Creates the RAG chain with OpenAI's ChatGPT."""
    env_vars = load_environment()
    if not env_vars["OPENAI_API_KEY"]:
        st.error("OPENAI_API_KEY environment variable not set.")
        st.stop()
    
    llm = ChatOpenAI(
        model=openai_model,
        temperature=temperature
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    st.title("🏦 ForteBank RAG Chatbot")
    st.markdown("Добро пожаловать в систему поддержки платежных систем ForteBank!")

    # Load environment variables
    env_vars = load_environment()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📊 Статус системы")
        
        # Check if vector store exists
        if env_vars["CHROMA_PATH"] and os.path.exists(env_vars["CHROMA_PATH"]):
            st.success("✅ База знаний загружена")
            vector_store_exists = True
        else:
            st.warning("⚠️ База знаний не найдена")
            vector_store_exists = False
        
        # Rebuild vector store button
        if st.button("🔄 Переиндексировать базу знаний"):
            rebuild_vector_store(env_vars)
        
        st.markdown("---")
        st.markdown("**Информация:**")
        st.markdown(f"📁 Путь к данным: `{env_vars['DATA_PATH']}`")
        st.markdown(f"🗃️ Путь к базе: `{env_vars['CHROMA_PATH']}`")

        st.markdown("---")
        # 1) Show all indexed PDFs
        st.header("📄 Индексированные PDF")
        if env_vars["DATA_PATH"]:
            pdf_files = glob.glob(
                os.path.join(env_vars["DATA_PATH"], "**/*.pdf"),
                recursive=True
            )
            if pdf_files:
                for pdf in pdf_files:
                    # display relative path so it’s not too long
                    rel = os.path.relpath(pdf, env_vars["DATA_PATH"])
                    st.text(f"• {rel}")
            else:
                st.text("— Нет PDF-файлов для отображения —")
        else:
            st.text("Путь к данным не настроен.")

        st.markdown("---")
        # 2) Allow user to upload new PDFs
        st.header("📤 Загрузить PDF для индексации")
        # in your sidebar, after the uploader:
        uploaded = st.file_uploader(
            "Выберите один или несколько PDF", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        # only run once per upload event
        if uploaded and not st.session_state.get("upload_processed", False):
            # save all files
            for file in uploaded:
                save_path = os.path.join(env_vars["DATA_PATH"], file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

            st.success(f"Загружено {len(uploaded)} файл(ов). Перестраиваю индекс…")
            st.session_state.upload_processed = True

            # now rebuild once
            rebuild_vector_store(env_vars)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize RAG system
    if vector_store_exists:
        try:
            vector_store = get_vector_store(env_vars["CHROMA_PATH"])
            rag_chain = create_rag_chain(vector_store, 'gpt-3.5-turbo', temperature=0)
            
            # Accept user input
            if prompt := st.chat_input("Задайте ваш вопрос..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("..."):
                        try:
                            response = rag_chain.invoke(prompt)
                            st.markdown(response)
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Извините, произошла ошибка: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except Exception as e:
            st.error(f"Ошибка инициализации системы: {str(e)}")
    else:
        st.warning("⚠️ База знаний не найдена. Пожалуйста, создайте индекс документов.")
        if st.button("📚 Создать базу знаний"):
            rebuild_vector_store(env_vars)

    # Clear chat button
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()

def rebuild_vector_store(env_vars):
    """Rebuild the vector store from scratch."""
    if not env_vars["DATA_PATH"] or not os.path.exists(env_vars["DATA_PATH"]):
        st.error("Путь к данным не найден. Проверьте переменную DATA_PATH.")
        return
    
    try:
        st.info("🔄 Начинаю переиндексацию базы знаний...")
        
        # Load documents
        st.info("📄 Загрузка PDF документов...")
        docs = load_all_documents(env_vars["DATA_PATH"])
        
        # Split documents
        st.info("✂️ Разделение документов на части...")
        chunks = split_documents(docs)
        st.success(f"Создано {len(chunks)} частей текста")
        
        # Get embedding function
        embedding_function = get_embedding_function()
        
        # Index documents
        st.info("🔍 Создание векторного индекса...")
        vector_store = index_documents(chunks, embedding_function, env_vars["CHROMA_PATH"])
        
        st.success("✅ База знаний успешно создана!")
        st.balloons()
        
        # Clear cache to reload the vector store
        st.cache_resource.clear()
        st.rerun()
        
    except Exception as e:
        st.error(f"Ошибка при создании базы знаний: {str(e)}")

if __name__ == "__main__":
    main()