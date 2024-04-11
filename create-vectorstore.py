from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


from dotenv import load_dotenv

load_dotenv()

token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

file = 'product_cleaned_removed.csv'

loader = CSVLoader(file, encoding='utf-8')
documents = loader.load()

docs = token_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)

db.save_local(f"faiss_products_recursive")