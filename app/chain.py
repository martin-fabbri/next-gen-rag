import os

from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LLM_MODEL = "gemma:7b-instruct"
# LLM_MODEL = "mistral:instruct"
LLM_MODEL = "mixtral:instruct"
EMBEDDING_MODEL = "nomic-embed-text"
TEMPERATURE = 0.9
ENABLE_TRACING = True
### Gemma
# DOCUMENT_CHUNK_SIZE=5000
###
### Mistral/Mixtral
DOCUMENT_CHUNK_SIZE = 7500
###
CHUNK_OVERLAP = 100

os.environ["LANGCHAIN_TRACING_V2"] = str(ENABLE_TRACING)
if ENABLE_TRACING:
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    load_dotenv()
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

embeddings_nomic = OllamaEmbeddings(model=EMBEDDING_MODEL)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=DOCUMENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
doc_splits = text_splitter.split_documents(doc_list)

vector_store = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings_nomic,
)
retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model_local = ChatOllama(model=LLM_MODEL)

nomic_mixtral_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)

